//! Production-oriented ONNX Runtime integration for RSMF graph payloads.
//!
//! `rsmf-runtime` keeps RSMF's storage/container boundary intact: graph bytes
//! remain opaque ONNX / ORT payloads, while this crate owns session lifecycle,
//! typed request values, runtime options, and session caching.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::{self, Debug};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ndarray::{ArrayD, ArrayViewD};
use ort::ep::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel as OrtGraphOptimizationLevel;
use ort::session::{RunOptions, Session, SessionOutputs};
use ort::value::{DynValue, Outlet, Tensor, TensorElementType};
use rsmf_core::manifest::GraphKind;
use rsmf_core::tensor::variant::{EncodingKind, LayoutTag};
use rsmf_core::{LogicalDtype, RsmfError, RsmfFile, StorageDtype, TensorView};
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::info_span;

/// Result alias for the runtime crate.
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Errors returned by the RSMF runtime layer.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Error propagated from `rsmf-core`.
    #[error(transparent)]
    Core(#[from] RsmfError),
    /// The requested graph payload is not present.
    #[error("graph payload {graph_idx} not found; file has {graph_count} graph payloads")]
    GraphNotFound {
        /// Requested graph index.
        graph_idx: usize,
        /// Number of graph payloads available in the file.
        graph_count: usize,
    },
    /// Error propagated from ONNX Runtime.
    #[error("onnx runtime error during {stage}: {message}")]
    Ort {
        /// Runtime stage that failed.
        stage: &'static str,
        /// Original ORT error text.
        message: String,
    },
    /// The runtime session cache was poisoned by a panic in another caller.
    #[error("runtime session cache lock poisoned")]
    CachePoisoned,
    /// A cached runtime session lock was poisoned by a panic in another caller.
    #[error("runtime session lock poisoned")]
    SessionPoisoned,
    /// The caller provided a tensor shape that cannot be represented safely.
    #[error("invalid runtime tensor shape: {0}")]
    Shape(String),
    /// A runtime value has a dtype this milestone does not materialize.
    #[error("unsupported runtime tensor dtype: {0}")]
    UnsupportedDtype(String),
    /// A requested RSMF tensor initializer is not present in the file.
    #[error("initializer {initializer_name} references missing RSMF tensor {tensor_name}")]
    InitializerTensorNotFound {
        /// ONNX initializer name.
        initializer_name: String,
        /// RSMF tensor name.
        tensor_name: String,
    },
    /// A requested RSMF tensor initializer cannot be bound by this runtime.
    #[error("initializer {initializer_name} cannot bind RSMF tensor {tensor_name}: {reason}")]
    UnsupportedInitializer {
        /// ONNX initializer name.
        initializer_name: String,
        /// RSMF tensor name.
        tensor_name: String,
        /// Reason the binding is unsupported.
        reason: String,
    },
    /// The runtime executor queue has reached its configured capacity.
    #[error("runtime executor queue is full; capacity is {capacity}")]
    ExecutorQueueFull {
        /// Maximum number of queued requests.
        capacity: usize,
    },
    /// The runtime executor queued tensor byte budget would be exceeded.
    #[error(
        "runtime executor queued tensor byte budget exceeded; requested {requested_bytes} bytes with {queued_bytes}/{capacity_bytes} bytes queued"
    )]
    ExecutorQueueBytesExceeded {
        /// Bytes in the rejected request's owned inputs.
        requested_bytes: usize,
        /// Bytes already queued before the rejected request.
        queued_bytes: usize,
        /// Maximum configured queued input bytes.
        capacity_bytes: usize,
    },
    /// The runtime executor has been closed.
    #[error("runtime executor is closed")]
    ExecutorClosed,
    /// The runtime executor lock was poisoned by a panic in another caller.
    #[error("runtime executor lock poisoned")]
    ExecutorPoisoned,
    /// The runtime executor request sequence counter overflowed.
    #[error("runtime executor request sequence overflow")]
    ExecutorSequenceOverflow,
    /// A request deadline expired before runtime dispatch.
    #[error("runtime request {request_id} deadline expired before dispatch")]
    RequestDeadlineExceeded {
        /// Caller-provided request identifier.
        request_id: String,
    },
    /// A queued request was cancelled before runtime dispatch.
    #[error("runtime request {request_id} was cancelled before dispatch")]
    RequestCancelled {
        /// Caller-provided request identifier.
        request_id: String,
    },
    /// A network serving operation failed.
    #[error("runtime network I/O error during {operation}: {message}")]
    NetworkIo {
        /// Operation that failed.
        operation: &'static str,
        /// Original I/O error text.
        message: String,
    },
    /// A network request was malformed or unsupported.
    #[error("invalid runtime network request: {reason}")]
    NetworkProtocol {
        /// Human-readable protocol error.
        reason: String,
    },
    /// A network request id is already active.
    #[error("runtime network request {request_id} is already active")]
    NetworkRequestAlreadyActive {
        /// Caller-provided request identifier.
        request_id: String,
    },
    /// A network request id is not currently active.
    #[error("runtime network request {request_id} is not active")]
    NetworkRequestNotFound {
        /// Caller-provided request identifier.
        request_id: String,
    },
}

/// Dynamic batching policy for [`RuntimeExecutor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicBatchingConfig {
    /// Maximum number of requests to combine into one runtime invocation.
    pub max_batch_size: usize,
    /// Maximum time a background worker waits after receiving the first request
    /// to allow more compatible requests to arrive.
    pub max_queue_delay: Duration,
}

impl Default for DynamicBatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_queue_delay: Duration::from_millis(1),
        }
    }
}

/// Admission-control policy for [`RuntimeExecutor`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RuntimeAdmissionConfig {
    /// Maximum bytes of owned input tensor data allowed to wait in the queue.
    /// `None` disables queued tensor byte admission.
    pub max_queued_tensor_bytes: Option<usize>,
}

/// ONNX Runtime graph optimization level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphOptimizationLevel {
    /// Disable all graph optimizations.
    Disable,
    /// Enable basic semantics-preserving optimizations.
    Level1,
    /// Enable extended graph optimizations.
    Level2,
    /// Enable layout optimizations.
    Level3,
    /// Enable all optimizations supported by the ORT build.
    #[default]
    All,
}

impl From<GraphOptimizationLevel> for OrtGraphOptimizationLevel {
    fn from(value: GraphOptimizationLevel) -> Self {
        match value {
            GraphOptimizationLevel::Disable => Self::Disable,
            GraphOptimizationLevel::Level1 => Self::Level1,
            GraphOptimizationLevel::Level2 => Self::Level2,
            GraphOptimizationLevel::Level3 => Self::Level3,
            GraphOptimizationLevel::All => Self::All,
        }
    }
}

/// Execution provider selection for R1.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionProvider {
    /// Portable CPU execution provider. This is always available in ORT.
    Cpu {
        /// Enable ORT's CPU memory arena.
        arena: bool,
    },
}

impl Default for ExecutionProvider {
    fn default() -> Self {
        Self::Cpu { arena: true }
    }
}

/// Mapping from an ONNX initializer name to an RSMF tensor.
///
/// R2 binds canonical, row-major RSMF tensors as ORT external initializers at
/// session build time. This keeps graph payloads from needing embedded weight
/// bytes. The initial CPU implementation materializes the RSMF tensor into an
/// ORT-owned value; true mmap/device zero-copy remains a later residency step.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InitializerBinding {
    /// Name of the initializer as referenced by the ONNX / ORT graph.
    pub initializer_name: String,
    /// Name of the tensor in the RSMF manifest.
    pub tensor_name: String,
    /// Optional global RSMF variant index to bind instead of the canonical
    /// variant.
    pub variant_idx: Option<u32>,
}

impl InitializerBinding {
    /// Build an initializer binding from a graph initializer name and an RSMF
    /// tensor name.
    #[must_use]
    pub fn new(initializer_name: impl Into<String>, tensor_name: impl Into<String>) -> Self {
        Self {
            initializer_name: initializer_name.into(),
            tensor_name: tensor_name.into(),
            variant_idx: None,
        }
    }

    /// Bind a specific global RSMF variant index instead of the canonical
    /// tensor variant.
    #[must_use]
    pub fn with_variant(mut self, variant_idx: u32) -> Self {
        self.variant_idx = Some(variant_idx);
        self
    }
}

/// Options used to create and cache an ONNX Runtime session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionOptions {
    /// Graph optimization level.
    pub graph_optimization: GraphOptimizationLevel,
    /// Optional number of intra-op threads.
    pub intra_threads: Option<usize>,
    /// Optional number of inter-op threads.
    pub inter_threads: Option<usize>,
    /// Enable ORT parallel graph execution.
    pub parallel_execution: bool,
    /// Enable ORT memory pattern optimization.
    pub memory_pattern: bool,
    /// Enable deterministic compute where ORT supports it.
    pub deterministic_compute: bool,
    /// Execution providers registered in priority order.
    pub execution_providers: Vec<ExecutionProvider>,
    /// ONNX initializer names to bind from RSMF tensors.
    pub initializers: Vec<InitializerBinding>,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            graph_optimization: GraphOptimizationLevel::All,
            intra_threads: None,
            inter_threads: None,
            parallel_execution: false,
            memory_pattern: true,
            deterministic_compute: false,
            execution_providers: vec![ExecutionProvider::default()],
            initializers: Vec::new(),
        }
    }
}

/// Cache key for a runtime session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    /// Graph payload index.
    pub graph_idx: usize,
    /// Session options used to build the ORT session.
    pub options: SessionOptions,
}

impl SessionKey {
    /// Build a cache key from a graph index and options.
    #[must_use]
    pub fn new(graph_idx: usize, options: SessionOptions) -> Self {
        Self { graph_idx, options }
    }
}

/// Metadata for one graph input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueInfo {
    /// Input/output name.
    pub name: String,
    /// Human-readable ORT value type.
    pub value_type: String,
}

impl ValueInfo {
    fn from_outlet(outlet: &Outlet) -> Self {
        Self {
            name: outlet.name().to_string(),
            value_type: outlet.dtype().to_string(),
        }
    }
}

/// Owned tensor value accepted by and returned from `rsmf-runtime`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dtype", rename_all = "snake_case")]
pub enum RuntimeTensor {
    /// F32 tensor.
    F32 { shape: Vec<usize>, data: Vec<f32> },
    /// F64 tensor.
    F64 { shape: Vec<usize>, data: Vec<f64> },
    /// I64 tensor.
    I64 { shape: Vec<usize>, data: Vec<i64> },
    /// I32 tensor.
    I32 { shape: Vec<usize>, data: Vec<i32> },
    /// U8 tensor.
    U8 { shape: Vec<usize>, data: Vec<u8> },
    /// I8 tensor.
    I8 { shape: Vec<usize>, data: Vec<i8> },
    /// Boolean tensor.
    Bool { shape: Vec<usize>, data: Vec<bool> },
}

impl RuntimeTensor {
    fn into_ort_value(self) -> Result<DynValue> {
        match self {
            RuntimeTensor::F32 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::F64 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::I64 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::I32 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::U8 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::I8 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::Bool { shape, data } => tensor_from_vec(shape, data),
        }
    }
}

/// Named input tensor map.
pub type RuntimeInputs = HashMap<String, RuntimeTensor>;

/// Named output tensor map.
pub type RuntimeOutputs = HashMap<String, RuntimeTensor>;

/// Configuration for the in-process runtime executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeExecutorConfig {
    /// Number of background worker threads. Use `0` to drive execution
    /// manually with [`RuntimeExecutor::execute_next`].
    pub worker_threads: usize,
    /// Maximum number of requests waiting in the queue.
    pub queue_capacity: usize,
    /// Optional dynamic batching policy. When enabled, compatible requests are
    /// concatenated along their leading tensor dimension.
    pub dynamic_batching: Option<DynamicBatchingConfig>,
    /// Optional admission limits beyond request count.
    pub admission: RuntimeAdmissionConfig,
}

impl Default for RuntimeExecutorConfig {
    fn default() -> Self {
        Self {
            worker_threads: 1,
            queue_capacity: 1024,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        }
    }
}

/// One graph-runtime request submitted to [`RuntimeExecutor`].
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeRequest {
    /// Caller-provided request identifier.
    pub request_id: String,
    /// Graph payload index.
    pub graph_idx: usize,
    /// Session options used to select or build the runtime session.
    pub options: SessionOptions,
    /// Owned runtime inputs.
    pub inputs: RuntimeInputs,
    /// Optional deadline. Expired requests fail before graph dispatch.
    pub deadline: Option<Instant>,
    /// Request priority. Higher values run before lower values.
    pub priority: i32,
}

impl RuntimeRequest {
    /// Build a request with default session options, no deadline, and priority
    /// `0`.
    #[must_use]
    pub fn new(request_id: impl Into<String>, graph_idx: usize, inputs: RuntimeInputs) -> Self {
        Self {
            request_id: request_id.into(),
            graph_idx,
            options: SessionOptions::default(),
            inputs,
            deadline: None,
            priority: 0,
        }
    }

    /// Set session options for this request.
    #[must_use]
    pub fn with_options(mut self, options: SessionOptions) -> Self {
        self.options = options;
        self
    }

    /// Set the absolute deadline for this request.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set a timeout relative to the current instant.
    #[must_use]
    pub fn with_timeout(self, timeout: Duration) -> Self {
        self.with_deadline(Instant::now() + timeout)
    }

    /// Set request priority. Higher values run first.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// Per-request timing metadata captured by [`RuntimeExecutor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeRequestTimings {
    /// Time when the request was accepted into the queue.
    pub queued_at: Instant,
    /// Time when the request left the queue.
    pub started_at: Instant,
    /// Time when execution finished.
    pub finished_at: Instant,
    /// Time spent waiting in the executor queue.
    pub queue_time: Duration,
    /// Time spent inside graph runtime execution.
    pub run_time: Duration,
}

/// Successful runtime executor response.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeResponse {
    /// Caller-provided request identifier.
    pub request_id: String,
    /// Owned graph outputs.
    pub outputs: RuntimeOutputs,
    /// Per-request timing metadata.
    pub timings: RuntimeRequestTimings,
}

/// Cumulative in-process executor metrics.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RuntimeExecutorMetrics {
    /// Requests accepted into the queue.
    pub submitted: u64,
    /// Requests completed successfully.
    pub completed: u64,
    /// Requests completed with an error.
    pub failed: u64,
    /// Requests rejected because their deadline expired before dispatch.
    pub deadline_expired: u64,
    /// Requests cancelled before dispatch.
    pub cancelled: u64,
    /// Requests rejected because the request queue was full.
    pub rejected_by_capacity: u64,
    /// Requests rejected because queued tensor bytes would exceed the configured
    /// budget.
    pub rejected_by_memory: u64,
    /// Requests currently waiting in the queue.
    pub current_queue_depth: usize,
    /// Maximum queue depth observed by this executor.
    pub max_observed_queue_depth: usize,
    /// Owned input tensor bytes currently waiting in the queue.
    pub current_queued_tensor_bytes: usize,
    /// Maximum queued owned input tensor bytes observed by this executor.
    pub max_observed_queued_tensor_bytes: usize,
    /// Requests currently executing inside graph runtime calls.
    pub active_requests: usize,
    /// Maximum concurrently active runtime requests observed by this executor.
    pub max_active_requests: usize,
    /// Runtime invocations currently executing.
    pub active_runtime_invocations: usize,
    /// Maximum concurrent runtime invocations observed by this executor.
    pub max_active_runtime_invocations: usize,
    /// Total batch slots currently executing across active runtime invocations.
    pub active_batch_size: usize,
    /// Largest total active batch size observed by this executor.
    pub max_active_batch_size: usize,
    /// Runtime invocations issued to the graph engine.
    pub runtime_invocations: u64,
    /// Runtime invocations that carried more than one request.
    pub batches_executed: u64,
    /// Requests completed through a runtime invocation carrying more than one
    /// request.
    pub batched_requests: u64,
    /// Attempted batches that fell back to individual request execution.
    pub batch_fallbacks: u64,
    /// Dynamic scheduler flushes because a batch reached configured capacity.
    pub batch_flushes_full: u64,
    /// Dynamic scheduler flushes because the max queue delay elapsed.
    pub batch_flushes_delay: u64,
    /// Dynamic scheduler flushes because queued input bytes reached the
    /// configured admission pressure point.
    pub batch_flushes_memory_pressure: u64,
    /// Dynamic scheduler flushes triggered by manual [`RuntimeExecutor::execute_next`].
    pub batch_flushes_manual: u64,
    /// Dynamic scheduler flushes triggered while closing the executor.
    pub batch_flushes_shutdown: u64,
    /// Cumulative queue time for completed and failed dispatched requests.
    pub total_queue_time: Duration,
    /// Cumulative run time for completed and failed dispatched requests.
    pub total_run_time: Duration,
}

/// Result of a request cancellation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeCancellationResult {
    /// The request was queued and is now marked for cancellation.
    Cancelled,
    /// The request had already been marked for cancellation.
    AlreadyCancelled,
    /// The request has already started runtime execution but no interrupt
    /// handle was available.
    AlreadyRunning,
    /// The request has started runtime execution and an ORT termination signal
    /// was requested.
    RunningCancellationRequested,
    /// The request already completed.
    AlreadyCompleted,
}

/// Shared cancellation token for a submitted runtime request.
#[derive(Clone)]
pub struct RuntimeCancellationToken {
    state: Arc<AtomicU8>,
    interrupt_requested: Arc<AtomicBool>,
    active_run: Arc<Mutex<Option<Arc<RunOptions>>>>,
}

impl Debug for RuntimeCancellationToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeCancellationToken")
            .field("state", &self.state.load(AtomicOrdering::Acquire))
            .field(
                "interrupt_requested",
                &self.interrupt_requested.load(AtomicOrdering::Acquire),
            )
            .finish_non_exhaustive()
    }
}

impl RuntimeCancellationToken {
    fn new() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(REQUEST_QUEUED)),
            interrupt_requested: Arc::new(AtomicBool::new(false)),
            active_run: Arc::new(Mutex::new(None)),
        }
    }

    /// Attempt to cancel the request before runtime dispatch.
    ///
    /// Queued requests are cancelled before dispatch. Already-running ORT
    /// executions receive a best-effort `RunOptions::terminate` signal and may
    /// stop once ORT observes that signal.
    #[must_use]
    pub fn cancel(&self) -> RuntimeCancellationResult {
        match self.state.compare_exchange(
            REQUEST_QUEUED,
            REQUEST_CANCELLED,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => RuntimeCancellationResult::Cancelled,
            Err(REQUEST_CANCELLED) => RuntimeCancellationResult::AlreadyCancelled,
            Err(REQUEST_RUNNING) => self.request_running_interrupt(),
            Err(REQUEST_COMPLETED) => RuntimeCancellationResult::AlreadyCompleted,
            Err(_) => RuntimeCancellationResult::AlreadyCompleted,
        }
    }

    /// Return `true` if the request is currently marked cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state.load(AtomicOrdering::Acquire) == REQUEST_CANCELLED
    }

    /// Return `true` if cancellation or ORT termination has been requested.
    #[must_use]
    pub fn is_cancellation_requested(&self) -> bool {
        self.is_cancelled() || self.interrupt_requested.load(AtomicOrdering::Acquire)
    }

    fn try_mark_running(&self) -> std::result::Result<(), RuntimeCancellationResult> {
        match self.state.compare_exchange(
            REQUEST_QUEUED,
            REQUEST_RUNNING,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(REQUEST_CANCELLED) => Err(RuntimeCancellationResult::AlreadyCancelled),
            Err(REQUEST_RUNNING) => Err(RuntimeCancellationResult::AlreadyRunning),
            Err(REQUEST_COMPLETED) => Err(RuntimeCancellationResult::AlreadyCompleted),
            Err(_) => Err(RuntimeCancellationResult::AlreadyCompleted),
        }
    }

    fn request_running_interrupt(&self) -> RuntimeCancellationResult {
        self.interrupt_requested
            .store(true, AtomicOrdering::Release);
        let run_options = self
            .active_run
            .lock()
            .ok()
            .and_then(|active| active.clone());
        if let Some(run_options) = run_options {
            let _ = run_options.terminate();
            RuntimeCancellationResult::RunningCancellationRequested
        } else {
            RuntimeCancellationResult::AlreadyRunning
        }
    }

    fn attach_run_options(&self, run_options: Arc<RunOptions>) -> Result<()> {
        {
            let mut active = self
                .active_run
                .lock()
                .map_err(|_| RuntimeError::ExecutorPoisoned)?;
            *active = Some(Arc::clone(&run_options));
        }
        if self.interrupt_requested.load(AtomicOrdering::Acquire) {
            run_options
                .terminate()
                .map_err(|e| ort_error("terminate run options", e))?;
        }
        Ok(())
    }

    fn clear_run_options(&self) {
        if let Ok(mut active) = self.active_run.lock() {
            *active = None;
        }
    }

    fn mark_completed(&self) {
        self.clear_run_options();
        self.state.store(REQUEST_COMPLETED, AtomicOrdering::Release);
    }
}

const REQUEST_QUEUED: u8 = 0;
const REQUEST_CANCELLED: u8 = 1;
const REQUEST_RUNNING: u8 = 2;
const REQUEST_COMPLETED: u8 = 3;

/// Completion handle returned by [`RuntimeExecutor::submit`].
#[derive(Debug)]
pub struct RuntimeRequestHandle {
    request_id: String,
    cancellation: RuntimeCancellationToken,
    receiver: Receiver<Result<RuntimeResponse>>,
}

impl RuntimeRequestHandle {
    /// Caller-provided request identifier for this handle.
    #[must_use]
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Shared cancellation token for this request.
    #[must_use]
    pub fn cancellation_token(&self) -> RuntimeCancellationToken {
        self.cancellation.clone()
    }

    /// Attempt to cancel this request before runtime dispatch.
    #[must_use]
    pub fn cancel(&self) -> RuntimeCancellationResult {
        self.cancellation.cancel()
    }

    /// Wait for request completion.
    pub fn wait(self) -> Result<RuntimeResponse> {
        self.receiver
            .recv()
            .map_err(|_| RuntimeError::ExecutorClosed)?
    }
}

/// Per-initializer memory materialization report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitializerMemoryReport {
    /// ONNX initializer name.
    pub initializer_name: String,
    /// RSMF tensor name used as the initializer source.
    pub tensor_name: String,
    /// Global RSMF variant index used as the initializer source, or `None`
    /// when the canonical variant was used.
    pub variant_idx: Option<u32>,
    /// Bytes materialized into an ORT-owned initializer value.
    pub materialized_bytes: usize,
}

/// Memory accounting captured when a session is built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMemoryReport {
    /// Embedded graph payload size in bytes.
    pub graph_payload_bytes: usize,
    /// Per-initializer materialization records.
    pub initializers: Vec<InitializerMemoryReport>,
    /// Total initializer bytes materialized into ORT-owned values.
    pub initializer_materialized_bytes: usize,
}

impl SessionMemoryReport {
    /// Number of initializer bindings materialized for this session.
    #[must_use]
    pub fn initializer_count(&self) -> usize {
        self.initializers.len()
    }
}

/// Cached ORT session handle.
#[derive(Clone, Debug)]
pub struct SessionHandle {
    key: SessionKey,
    session: Arc<Mutex<Session>>,
    inputs: Vec<ValueInfo>,
    outputs: Vec<ValueInfo>,
    memory_report: SessionMemoryReport,
}

impl SessionHandle {
    /// Cache key used to create this session.
    #[must_use]
    pub fn key(&self) -> &SessionKey {
        &self.key
    }

    /// Graph input metadata captured when the session was built.
    #[must_use]
    pub fn inputs(&self) -> &[ValueInfo] {
        &self.inputs
    }

    /// Graph output metadata captured when the session was built.
    #[must_use]
    pub fn outputs(&self) -> &[ValueInfo] {
        &self.outputs
    }

    /// Memory accounting captured when this session was built.
    #[must_use]
    pub fn memory_report(&self) -> &SessionMemoryReport {
        &self.memory_report
    }

    /// Run this cached session with owned runtime tensors.
    pub fn run(&self, inputs: RuntimeInputs) -> Result<RuntimeOutputs> {
        self.run_with_cancellation(inputs, None)
    }

    fn run_with_cancellation(
        &self,
        inputs: RuntimeInputs,
        cancellation: Option<&RuntimeCancellationToken>,
    ) -> Result<RuntimeOutputs> {
        match cancellation {
            Some(cancellation) => self.run_with_cancellations(inputs, &[cancellation]),
            None => self.run_with_cancellations(inputs, &[]),
        }
    }

    fn run_with_cancellations(
        &self,
        inputs: RuntimeInputs,
        cancellations: &[&RuntimeCancellationToken],
    ) -> Result<RuntimeOutputs> {
        let ort_inputs = inputs
            .into_iter()
            .map(|(name, tensor)| Ok((name, tensor.into_ort_value()?)))
            .collect::<Result<Vec<_>>>()?;
        let mut session = self.lock_session()?;
        if !cancellations.is_empty() {
            let run_options =
                Arc::new(RunOptions::new().map_err(|e| ort_error("create run options", e))?);
            for cancellation in cancellations {
                if let Err(error) = cancellation.attach_run_options(Arc::clone(&run_options)) {
                    for cancellation in cancellations {
                        cancellation.clear_run_options();
                    }
                    return Err(error);
                }
            }
            let result = session
                .run_with_options(ort_inputs, &*run_options)
                .map_err(|e| ort_error("session run", e));
            for cancellation in cancellations {
                cancellation.clear_run_options();
            }
            materialize_outputs(result?)
        } else {
            let outputs = session
                .run(ort_inputs)
                .map_err(|e| ort_error("session run", e))?;
            materialize_outputs(outputs)
        }
    }

    fn lock_session(&self) -> Result<MutexGuard<'_, Session>> {
        self.session
            .lock()
            .map_err(|_| RuntimeError::SessionPoisoned)
    }
}

/// High-level inference engine for RSMF files.
pub struct Engine {
    file: RsmfFile,
    cache: Mutex<HashMap<SessionKey, SessionHandle>>,
}

struct BuiltSession {
    session: Session,
    memory_report: SessionMemoryReport,
}

struct InitializerValue {
    value: Arc<DynValue>,
    memory_report: InitializerMemoryReport,
}

impl Engine {
    /// Create a new engine from an opened RSMF file.
    pub fn new(file: RsmfFile) -> Result<Self> {
        Ok(Self {
            file,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Return the underlying RSMF file.
    #[must_use]
    pub fn file(&self) -> &RsmfFile {
        &self.file
    }

    /// Create a fresh default ONNX Runtime session for compatibility with the
    /// original runtime API.
    ///
    /// Prefer [`Self::session_handle`] for production callers that want cached
    /// sessions and metadata.
    pub fn session(&self, graph_idx: usize) -> Result<Session> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::session", graph_idx).entered();

        self.build_session(graph_idx, &SessionOptions::default())
            .map(|built| built.session)
    }

    /// Return a cached session handle for `graph_idx` and `options`.
    pub fn session_handle(
        &self,
        graph_idx: usize,
        options: SessionOptions,
    ) -> Result<SessionHandle> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::session_handle", graph_idx).entered();

        let key = SessionKey::new(graph_idx, options);
        let mut cache = self.cache.lock().map_err(|_| RuntimeError::CachePoisoned)?;
        if let Some(handle) = cache.get(&key) {
            return Ok(handle.clone());
        }

        let built = self.build_session(key.graph_idx, &key.options)?;
        let session = built.session;
        let handle = SessionHandle {
            inputs: session
                .inputs()
                .iter()
                .map(ValueInfo::from_outlet)
                .collect(),
            outputs: session
                .outputs()
                .iter()
                .map(ValueInfo::from_outlet)
                .collect(),
            session: Arc::new(Mutex::new(session)),
            memory_report: built.memory_report,
            key: key.clone(),
        };
        cache.insert(key, handle.clone());
        Ok(handle)
    }

    /// Run a graph using a cached session.
    pub fn run(
        &self,
        graph_idx: usize,
        options: SessionOptions,
        inputs: RuntimeInputs,
    ) -> Result<RuntimeOutputs> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::run", graph_idx).entered();

        self.session_handle(graph_idx, options)?.run(inputs)
    }

    fn run_with_cancellations(
        &self,
        graph_idx: usize,
        options: SessionOptions,
        inputs: RuntimeInputs,
        cancellations: &[&RuntimeCancellationToken],
    ) -> Result<RuntimeOutputs> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::run_with_cancellations", graph_idx).entered();

        self.session_handle(graph_idx, options)?
            .run_with_cancellations(inputs, cancellations)
    }

    /// Convenience helper for F32 models.
    pub fn run_f32(
        &self,
        graph_idx: usize,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        self.run_f32_with_options(graph_idx, SessionOptions::default(), inputs)
    }

    /// Convenience helper for F32 models with explicit session options.
    pub fn run_f32_with_options(
        &self,
        graph_idx: usize,
        options: SessionOptions,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        let runtime_inputs = inputs
            .into_iter()
            .map(|(name, array)| {
                let shape = array.shape().to_vec();
                let data = array.iter().copied().collect();
                (name, RuntimeTensor::F32 { shape, data })
            })
            .collect();
        let outputs = self.run(graph_idx, options, runtime_inputs)?;
        outputs
            .into_iter()
            .map(|(name, tensor)| match tensor {
                RuntimeTensor::F32 { shape, data } => {
                    let array = ArrayD::from_shape_vec(shape, data).map_err(|e| {
                        RuntimeError::Shape(format!("F32 output {name} shape mismatch: {e}"))
                    })?;
                    Ok((name, array))
                }
                other => Err(RuntimeError::UnsupportedDtype(format!(
                    "run_f32 expected F32 output, got {}",
                    runtime_tensor_kind(&other)
                ))),
            })
            .collect()
    }

    fn build_session(&self, graph_idx: usize, options: &SessionOptions) -> Result<BuiltSession> {
        let payloads = self.file.graph_payloads();
        let payload = payloads.get(graph_idx).ok_or(RuntimeError::GraphNotFound {
            graph_idx,
            graph_count: payloads.len(),
        })?;
        self.validate_initializer_bindings(payload.kind, payload.bytes, &options.initializers)?;

        let mut builder = Session::builder().map_err(|e| ort_error("session builder", e))?;
        builder = builder
            .with_optimization_level(options.graph_optimization.into())
            .map_err(|e| ort_error("set graph optimization level", e))?;
        builder = builder
            .with_parallel_execution(options.parallel_execution)
            .map_err(|e| ort_error("set execution mode", e))?;
        builder = builder
            .with_memory_pattern(options.memory_pattern)
            .map_err(|e| ort_error("set memory pattern", e))?;
        builder = builder
            .with_deterministic_compute(options.deterministic_compute)
            .map_err(|e| ort_error("set deterministic compute", e))?;

        if let Some(threads) = options.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|e| ort_error("set intra-op threads", e))?;
        }
        if let Some(threads) = options.inter_threads {
            builder = builder
                .with_inter_threads(threads)
                .map_err(|e| ort_error("set inter-op threads", e))?;
        }

        let execution_providers = options
            .execution_providers
            .iter()
            .map(|provider| match provider {
                ExecutionProvider::Cpu { arena } => CPUExecutionProvider::default()
                    .with_arena_allocator(*arena)
                    .build(),
            })
            .collect::<Vec<_>>();
        if !execution_providers.is_empty() {
            builder = builder
                .with_execution_providers(execution_providers)
                .map_err(|e| ort_error("register execution providers", e))?;
        }

        let mut initializer_reports = Vec::with_capacity(options.initializers.len());
        for binding in &options.initializers {
            let initializer = self.initializer_value(binding)?;
            builder = builder
                .with_external_initializer(&binding.initializer_name, initializer.value)
                .map_err(|e| ort_error("register external initializer", e))?;
            initializer_reports.push(initializer.memory_report);
        }

        let initializer_materialized_bytes =
            initializer_reports.iter().try_fold(0usize, |acc, report| {
                acc.checked_add(report.materialized_bytes).ok_or_else(|| {
                    RuntimeError::Shape("initializer materialized byte count overflow".to_string())
                })
            })?;
        let memory_report = SessionMemoryReport {
            graph_payload_bytes: payload.bytes.len(),
            initializer_materialized_bytes,
            initializers: initializer_reports,
        };
        let session = builder
            .commit_from_memory(payload.bytes)
            .map_err(|e| ort_error("session creation", e))?;
        Ok(BuiltSession {
            session,
            memory_report,
        })
    }

    fn initializer_value(&self, binding: &InitializerBinding) -> Result<InitializerValue> {
        let view = self.initializer_view(binding)?;
        validate_initializer_view(binding, &view)?;
        let shape = shape_u64_to_usize(view.shape()).map_err(|error| {
            RuntimeError::UnsupportedInitializer {
                initializer_name: binding.initializer_name.clone(),
                tensor_name: binding.tensor_name.clone(),
                reason: error,
            }
        })?;
        let (value, materialized_bytes) = initializer_tensor_from_view(binding, &view, shape)?;
        Ok(InitializerValue {
            value,
            memory_report: InitializerMemoryReport {
                initializer_name: binding.initializer_name.clone(),
                tensor_name: binding.tensor_name.clone(),
                variant_idx: binding.variant_idx,
                materialized_bytes,
            },
        })
    }

    fn validate_initializer_bindings(
        &self,
        graph_kind: GraphKind,
        graph_bytes: &[u8],
        bindings: &[InitializerBinding],
    ) -> Result<()> {
        if bindings.is_empty() || graph_kind != GraphKind::Onnx {
            return Ok(());
        }

        let initializers = onnx_initializers(graph_bytes).map_err(|reason| {
            RuntimeError::UnsupportedInitializer {
                initializer_name: "<onnx>".to_string(),
                tensor_name: "<graph>".to_string(),
                reason,
            }
        })?;
        for binding in bindings {
            let Some(info) = initializers.get(&binding.initializer_name) else {
                return Err(RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason: "initializer is not declared by the ONNX graph".to_string(),
                });
            };
            let view = self.initializer_view(binding)?;
            validate_initializer_view(binding, &view)?;
            let expected_data_type = OnnxTensorDataType::from_logical_dtype(view.dtype())
                .ok_or_else(|| RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason: format!("unsupported RSMF initializer dtype {:?}", view.dtype()),
                })?;
            if info.data_type != Some(expected_data_type) {
                return Err(RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason: format!(
                        "ONNX initializer dtype is {}, but RSMF tensor dtype is {}",
                        info.data_type.map_or_else(
                            || "missing".to_string(),
                            |data_type| data_type.name().to_string()
                        ),
                        expected_data_type.name()
                    ),
                });
            }

            let actual_shape = shape_u64_to_i64(view.shape()).map_err(|reason| {
                RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason,
                }
            })?;
            if info.shape != actual_shape {
                return Err(RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason: format!(
                        "ONNX initializer shape {:?} does not match RSMF tensor shape {:?}",
                        info.shape, actual_shape
                    ),
                });
            }
        }

        Ok(())
    }

    fn initializer_view<'a>(&'a self, binding: &InitializerBinding) -> Result<TensorView<'a>> {
        let result = if let Some(variant_idx) = binding.variant_idx {
            self.file
                .tensor_view_variant(&binding.tensor_name, variant_idx)
        } else {
            self.file.tensor_view(&binding.tensor_name)
        };
        result.map_err(|error| match error {
            RsmfError::NotFound { what } if what == format!("tensor {}", binding.tensor_name) => {
                RuntimeError::InitializerTensorNotFound {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                }
            }
            RsmfError::NotFound { what } => RuntimeError::UnsupportedInitializer {
                initializer_name: binding.initializer_name.clone(),
                tensor_name: binding.tensor_name.clone(),
                reason: what,
            },
            other => RuntimeError::Core(other),
        })
    }
}

/// In-process, graph-agnostic runtime request executor.
///
/// The executor owns or shares an [`Engine`], accepts bounded queued requests,
/// orders them by priority and FIFO sequence, and runs requests on background
/// workers or through [`Self::execute_next`].
pub struct RuntimeExecutor {
    inner: Arc<RuntimeExecutorInner>,
    workers: Vec<JoinHandle<()>>,
}

impl RuntimeExecutor {
    /// Create an executor that owns `engine`.
    #[must_use]
    pub fn new(engine: Engine, config: RuntimeExecutorConfig) -> Self {
        Self::from_shared(Arc::new(engine), config)
    }

    /// Create an executor from a shared engine.
    #[must_use]
    pub fn from_shared(engine: Arc<Engine>, config: RuntimeExecutorConfig) -> Self {
        let inner = Arc::new(RuntimeExecutorInner {
            engine,
            state: Mutex::new(RuntimeExecutorState {
                queue: BinaryHeap::new(),
                closed: false,
                next_sequence: 0,
                metrics: RuntimeExecutorMetrics::default(),
            }),
            available: Condvar::new(),
            queue_capacity: config.queue_capacity,
            dynamic_batching: config.dynamic_batching,
            admission: config.admission,
        });
        let workers = (0..config.worker_threads)
            .map(|_| {
                let inner = Arc::clone(&inner);
                std::thread::spawn(move || worker_loop(inner))
            })
            .collect();
        Self { inner, workers }
    }

    /// Submit a request to the bounded priority queue.
    pub fn submit(&self, request: RuntimeRequest) -> Result<RuntimeRequestHandle> {
        let (sender, receiver) = mpsc::channel();
        let request_id = request.request_id.clone();
        let input_bytes = runtime_inputs_data_bytes(&request.inputs)?;
        let cancellation = RuntimeCancellationToken::new();
        let mut state = self.inner.lock_state()?;
        if state.closed {
            return Err(RuntimeError::ExecutorClosed);
        }
        if state.queue.len() >= self.inner.queue_capacity {
            state.metrics.rejected_by_capacity =
                state.metrics.rejected_by_capacity.saturating_add(1);
            return Err(RuntimeError::ExecutorQueueFull {
                capacity: self.inner.queue_capacity,
            });
        }
        if let Some(capacity_bytes) = self.inner.admission.max_queued_tensor_bytes {
            let queued_bytes = state.metrics.current_queued_tensor_bytes;
            let would_queue = queued_bytes.checked_add(input_bytes).ok_or_else(|| {
                RuntimeError::Shape("queued tensor byte count overflow".to_string())
            })?;
            if would_queue > capacity_bytes {
                state.metrics.rejected_by_memory =
                    state.metrics.rejected_by_memory.saturating_add(1);
                return Err(RuntimeError::ExecutorQueueBytesExceeded {
                    requested_bytes: input_bytes,
                    queued_bytes,
                    capacity_bytes,
                });
            }
        }

        let sequence = state.next_sequence;
        state.next_sequence = state
            .next_sequence
            .checked_add(1)
            .ok_or(RuntimeError::ExecutorSequenceOverflow)?;
        let priority = request.priority;
        state.push_queued(QueuedRuntimeRequest {
            priority,
            sequence,
            queued_at: Instant::now(),
            request,
            cancellation: cancellation.clone(),
            input_bytes,
            sender,
        });
        state.metrics.submitted = state.metrics.submitted.saturating_add(1);
        drop(state);
        self.inner.available.notify_one();

        Ok(RuntimeRequestHandle {
            request_id,
            cancellation,
            receiver,
        })
    }

    /// Execute the next queued request on the caller's thread.
    ///
    /// Returns `Ok(true)` when a request was executed and `Ok(false)` when no
    /// request is currently queued.
    pub fn execute_next(&self) -> Result<bool> {
        let Some(batch) = self.inner.pop_ready_batch()? else {
            return Ok(false);
        };
        execute_queued_batch(&self.inner, batch);
        Ok(true)
    }

    /// Stop accepting new requests and wake sleeping workers.
    ///
    /// Already queued requests are still drained by workers or
    /// [`Self::execute_next`].
    pub fn close(&self) -> Result<()> {
        let mut state = self.inner.lock_state()?;
        state.closed = true;
        drop(state);
        self.inner.available.notify_all();
        Ok(())
    }

    /// Snapshot cumulative executor metrics.
    pub fn metrics(&self) -> Result<RuntimeExecutorMetrics> {
        self.inner.lock_state().map(|state| state.metrics.clone())
    }

    /// Current number of queued requests.
    pub fn queued_len(&self) -> Result<usize> {
        self.inner.lock_state().map(|state| state.queue.len())
    }
}

impl Drop for RuntimeExecutor {
    fn drop(&mut self) {
        let _ = self.close();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Configuration for the dependency-light HTTP/1.1 network serving API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeNetworkServerConfig {
    /// Address to bind. Use port `0` to request an OS-assigned port.
    pub bind_addr: SocketAddr,
    /// Maximum accepted request body size in bytes.
    pub max_body_bytes: usize,
}

impl Default for RuntimeNetworkServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            max_body_bytes: 16 * 1024 * 1024,
        }
    }
}

/// JSON request accepted by `POST /v1/run`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeNetworkRunRequest {
    /// Caller-provided request identifier.
    pub request_id: String,
    /// Graph payload index.
    pub graph_idx: usize,
    /// Optional full session options. Defaults to [`SessionOptions::default`]
    /// when omitted.
    pub options: Option<SessionOptions>,
    /// Owned runtime inputs.
    pub inputs: RuntimeInputs,
    /// Optional request priority. Higher values run first.
    pub priority: Option<i32>,
    /// Optional timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

impl RuntimeNetworkRunRequest {
    fn into_runtime_request(self) -> RuntimeRequest {
        let mut request = RuntimeRequest::new(self.request_id, self.graph_idx, self.inputs)
            .with_options(self.options.unwrap_or_default());
        if let Some(priority) = self.priority {
            request = request.with_priority(priority);
        }
        if let Some(timeout_ms) = self.timeout_ms {
            request = request.with_timeout(Duration::from_millis(timeout_ms));
        }
        request
    }
}

/// Duration-only timings returned by the network serving API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeNetworkTimings {
    /// Time spent waiting in the executor queue, in milliseconds.
    pub queue_time_ms: u128,
    /// Time spent inside graph runtime execution, in milliseconds.
    pub run_time_ms: u128,
}

impl RuntimeNetworkTimings {
    fn from_timings(timings: &RuntimeRequestTimings) -> Self {
        Self {
            queue_time_ms: timings.queue_time.as_millis(),
            run_time_ms: timings.run_time.as_millis(),
        }
    }
}

/// JSON response returned by `POST /v1/run`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeNetworkRunResponse {
    /// Caller-provided request identifier.
    pub request_id: String,
    /// Owned graph outputs.
    pub outputs: RuntimeOutputs,
    /// Duration-only request timing metadata.
    pub timings: RuntimeNetworkTimings,
}

impl RuntimeNetworkRunResponse {
    fn from_response(response: RuntimeResponse) -> Self {
        Self {
            request_id: response.request_id,
            outputs: response.outputs,
            timings: RuntimeNetworkTimings::from_timings(&response.timings),
        }
    }
}

/// JSON metrics snapshot returned by `GET /metrics`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeNetworkMetrics {
    /// Requests accepted into the queue.
    pub submitted: u64,
    /// Requests completed successfully.
    pub completed: u64,
    /// Requests completed with an error.
    pub failed: u64,
    /// Requests rejected because their deadline expired before dispatch.
    pub deadline_expired: u64,
    /// Requests cancelled before dispatch.
    pub cancelled: u64,
    /// Requests rejected because the request queue was full.
    pub rejected_by_capacity: u64,
    /// Requests rejected because queued tensor bytes would exceed the configured
    /// budget.
    pub rejected_by_memory: u64,
    /// Requests currently waiting in the queue.
    pub current_queue_depth: usize,
    /// Maximum queue depth observed by this executor.
    pub max_observed_queue_depth: usize,
    /// Owned input tensor bytes currently waiting in the queue.
    pub current_queued_tensor_bytes: usize,
    /// Maximum queued owned input tensor bytes observed by this executor.
    pub max_observed_queued_tensor_bytes: usize,
    /// Requests currently executing inside graph runtime calls.
    pub active_requests: usize,
    /// Maximum concurrently active runtime requests observed by this executor.
    pub max_active_requests: usize,
    /// Runtime invocations currently executing.
    pub active_runtime_invocations: usize,
    /// Maximum concurrent runtime invocations observed by this executor.
    pub max_active_runtime_invocations: usize,
    /// Total batch slots currently executing across active runtime invocations.
    pub active_batch_size: usize,
    /// Largest total active batch size observed by this executor.
    pub max_active_batch_size: usize,
    /// Runtime invocations issued to the graph engine.
    pub runtime_invocations: u64,
    /// Runtime invocations that carried more than one request.
    pub batches_executed: u64,
    /// Requests completed through a runtime invocation carrying more than one
    /// request.
    pub batched_requests: u64,
    /// Attempted batches that fell back to individual request execution.
    pub batch_fallbacks: u64,
    /// Dynamic scheduler flushes because a batch reached configured capacity.
    pub batch_flushes_full: u64,
    /// Dynamic scheduler flushes because the max queue delay elapsed.
    pub batch_flushes_delay: u64,
    /// Dynamic scheduler flushes because queued input bytes reached the
    /// configured admission pressure point.
    pub batch_flushes_memory_pressure: u64,
    /// Dynamic scheduler flushes triggered by manual dispatch.
    pub batch_flushes_manual: u64,
    /// Dynamic scheduler flushes triggered while closing the executor.
    pub batch_flushes_shutdown: u64,
    /// Cumulative queue time, in milliseconds.
    pub total_queue_time_ms: u128,
    /// Cumulative run time, in milliseconds.
    pub total_run_time_ms: u128,
}

impl From<RuntimeExecutorMetrics> for RuntimeNetworkMetrics {
    fn from(metrics: RuntimeExecutorMetrics) -> Self {
        Self {
            submitted: metrics.submitted,
            completed: metrics.completed,
            failed: metrics.failed,
            deadline_expired: metrics.deadline_expired,
            cancelled: metrics.cancelled,
            rejected_by_capacity: metrics.rejected_by_capacity,
            rejected_by_memory: metrics.rejected_by_memory,
            current_queue_depth: metrics.current_queue_depth,
            max_observed_queue_depth: metrics.max_observed_queue_depth,
            current_queued_tensor_bytes: metrics.current_queued_tensor_bytes,
            max_observed_queued_tensor_bytes: metrics.max_observed_queued_tensor_bytes,
            active_requests: metrics.active_requests,
            max_active_requests: metrics.max_active_requests,
            active_runtime_invocations: metrics.active_runtime_invocations,
            max_active_runtime_invocations: metrics.max_active_runtime_invocations,
            active_batch_size: metrics.active_batch_size,
            max_active_batch_size: metrics.max_active_batch_size,
            runtime_invocations: metrics.runtime_invocations,
            batches_executed: metrics.batches_executed,
            batched_requests: metrics.batched_requests,
            batch_fallbacks: metrics.batch_fallbacks,
            batch_flushes_full: metrics.batch_flushes_full,
            batch_flushes_delay: metrics.batch_flushes_delay,
            batch_flushes_memory_pressure: metrics.batch_flushes_memory_pressure,
            batch_flushes_manual: metrics.batch_flushes_manual,
            batch_flushes_shutdown: metrics.batch_flushes_shutdown,
            total_queue_time_ms: metrics.total_queue_time.as_millis(),
            total_run_time_ms: metrics.total_run_time.as_millis(),
        }
    }
}

/// Handle for a background network serving loop.
#[derive(Debug)]
pub struct RuntimeNetworkServerHandle {
    local_addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl RuntimeNetworkServerHandle {
    /// Address the server is bound to.
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Ask the serving loop to stop and wait for its listener thread.
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown.store(true, AtomicOrdering::Release);
        let _ = TcpStream::connect(self.local_addr);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        Ok(())
    }
}

impl Drop for RuntimeNetworkServerHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, AtomicOrdering::Release);
        let _ = TcpStream::connect(self.local_addr);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Dependency-light HTTP/1.1 JSON server for [`RuntimeExecutor`].
pub struct RuntimeNetworkServer {
    executor: Arc<RuntimeExecutor>,
    config: RuntimeNetworkServerConfig,
}

impl RuntimeNetworkServer {
    /// Create a network server wrapper around an executor.
    #[must_use]
    pub fn new(executor: RuntimeExecutor, config: RuntimeNetworkServerConfig) -> Self {
        Self::from_shared(Arc::new(executor), config)
    }

    /// Create a network server wrapper around a shared executor.
    #[must_use]
    pub fn from_shared(executor: Arc<RuntimeExecutor>, config: RuntimeNetworkServerConfig) -> Self {
        Self { executor, config }
    }

    /// Bind and start the background serving loop.
    pub fn start(self) -> Result<RuntimeNetworkServerHandle> {
        let listener = TcpListener::bind(self.config.bind_addr)
            .map_err(|e| network_io_error("bind network server", e))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| network_io_error("configure network listener", e))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| network_io_error("read network listener address", e))?;
        let state = Arc::new(RuntimeNetworkServerState::default());
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = Arc::clone(&shutdown);
        let executor = Arc::clone(&self.executor);
        let max_body_bytes = self.config.max_body_bytes;
        let thread = std::thread::spawn(move || {
            network_accept_loop(listener, executor, state, thread_shutdown, max_body_bytes);
        });
        Ok(RuntimeNetworkServerHandle {
            local_addr,
            shutdown,
            thread: Some(thread),
        })
    }
}

#[derive(Default)]
struct RuntimeNetworkServerState {
    inflight: Mutex<HashMap<String, Option<RuntimeCancellationToken>>>,
}

impl RuntimeNetworkServerState {
    fn reserve(&self, request_id: &str) -> Result<()> {
        let mut inflight = self
            .inflight
            .lock()
            .map_err(|_| RuntimeError::ExecutorPoisoned)?;
        if inflight.contains_key(request_id) {
            return Err(RuntimeError::NetworkRequestAlreadyActive {
                request_id: request_id.to_string(),
            });
        }
        inflight.insert(request_id.to_string(), None);
        Ok(())
    }

    fn attach(&self, request_id: &str, cancellation: RuntimeCancellationToken) -> Result<()> {
        let mut inflight = self
            .inflight
            .lock()
            .map_err(|_| RuntimeError::ExecutorPoisoned)?;
        inflight.insert(request_id.to_string(), Some(cancellation));
        Ok(())
    }

    fn remove(&self, request_id: &str) {
        if let Ok(mut inflight) = self.inflight.lock() {
            inflight.remove(request_id);
        }
    }

    fn cancellation(&self, request_id: &str) -> Result<Option<RuntimeCancellationToken>> {
        let inflight = self
            .inflight
            .lock()
            .map_err(|_| RuntimeError::ExecutorPoisoned)?;
        inflight
            .get(request_id)
            .cloned()
            .ok_or_else(|| RuntimeError::NetworkRequestNotFound {
                request_id: request_id.to_string(),
            })
    }
}

#[derive(Debug)]
struct RuntimeHttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn network_accept_loop(
    listener: TcpListener,
    executor: Arc<RuntimeExecutor>,
    state: Arc<RuntimeNetworkServerState>,
    shutdown: Arc<AtomicBool>,
    max_body_bytes: usize,
) {
    while !shutdown.load(AtomicOrdering::Acquire) {
        match listener.accept() {
            Ok((stream, _)) => {
                let executor = Arc::clone(&executor);
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    handle_network_connection(stream, executor, state, max_body_bytes);
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break,
        }
    }
}

fn handle_network_connection(
    mut stream: TcpStream,
    executor: Arc<RuntimeExecutor>,
    state: Arc<RuntimeNetworkServerState>,
    max_body_bytes: usize,
) {
    let response = read_http_request(&mut stream, max_body_bytes)
        .and_then(|request| route_network_request(request, &executor, &state));
    match response {
        Ok(body) => {
            let _ = write_json_response(&mut stream, 200, "OK", &body);
        }
        Err(error) => {
            let (status, reason, code) = network_error_status(&error);
            let body = serde_json::json!({
                "error": {
                    "code": code,
                    "message": error.to_string()
                }
            });
            let _ = write_json_response(&mut stream, status, reason, &body);
        }
    }
}

fn route_network_request(
    request: RuntimeHttpRequest,
    executor: &RuntimeExecutor,
    state: &RuntimeNetworkServerState,
) -> Result<serde_json::Value> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => Ok(serde_json::json!({ "status": "ok" })),
        ("GET", "/metrics") => {
            let metrics = RuntimeNetworkMetrics::from(executor.metrics()?);
            serde_json::to_value(metrics).map_err(network_json_error)
        }
        ("POST", "/v1/run") => handle_network_run(request.body, executor, state),
        _ if request.method == "GET" && request.path.starts_with("/v1/requests/") => {
            let request_id = network_request_id_from_path(&request.path)?;
            let cancellation = state.cancellation(request_id)?;
            let status = if let Some(cancellation) = cancellation {
                if cancellation.is_cancellation_requested() {
                    "cancellation_requested"
                } else {
                    "inflight"
                }
            } else {
                "submitting"
            };
            Ok(serde_json::json!({
                "request_id": request_id,
                "status": status
            }))
        }
        _ if request.method == "DELETE" && request.path.starts_with("/v1/requests/") => {
            let request_id = network_request_id_from_path(&request.path)?;
            let cancellation = state.cancellation(request_id)?;
            let Some(cancellation) = cancellation else {
                return Ok(serde_json::json!({
                    "request_id": request_id,
                    "cancellation": "submitting"
                }));
            };
            Ok(serde_json::json!({
                "request_id": request_id,
                "cancellation": format!("{:?}", cancellation.cancel())
            }))
        }
        _ => Err(RuntimeError::NetworkProtocol {
            reason: format!("unsupported route {} {}", request.method, request.path),
        }),
    }
}

fn handle_network_run(
    body: Vec<u8>,
    executor: &RuntimeExecutor,
    state: &RuntimeNetworkServerState,
) -> Result<serde_json::Value> {
    let request: RuntimeNetworkRunRequest =
        serde_json::from_slice(&body).map_err(network_json_error)?;
    let request_id = request.request_id.clone();
    state.reserve(&request_id)?;
    let runtime_request = request.into_runtime_request();
    let handle = match executor.submit(runtime_request) {
        Ok(handle) => handle,
        Err(error) => {
            state.remove(&request_id);
            return Err(error);
        }
    };
    state.attach(&request_id, handle.cancellation_token())?;
    let response = handle.wait();
    state.remove(&request_id);
    response
        .map(RuntimeNetworkRunResponse::from_response)
        .and_then(|response| serde_json::to_value(response).map_err(network_json_error))
}

fn network_request_id_from_path(path: &str) -> Result<&str> {
    let request_id =
        path.strip_prefix("/v1/requests/")
            .ok_or_else(|| RuntimeError::NetworkProtocol {
                reason: format!("invalid request path {path}"),
            })?;
    if request_id.is_empty() || request_id.contains('/') {
        return Err(RuntimeError::NetworkProtocol {
            reason: format!("invalid request id path {path}"),
        });
    }
    Ok(request_id)
}

fn read_http_request(stream: &mut TcpStream, max_body_bytes: usize) -> Result<RuntimeHttpRequest> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| network_io_error("configure connection read timeout", e))?;
    let mut buffer = Vec::new();
    let header_end = loop {
        if let Some(header_end) = find_header_end(&buffer) {
            break header_end;
        }
        if buffer.len() > 16 * 1024 {
            return Err(RuntimeError::NetworkProtocol {
                reason: "HTTP request headers exceed 16 KiB".to_string(),
            });
        }
        let mut chunk = [0u8; 1024];
        let read = stream
            .read(&mut chunk)
            .map_err(|e| network_io_error("read network request", e))?;
        if read == 0 {
            return Err(RuntimeError::NetworkProtocol {
                reason: "connection closed before HTTP headers completed".to_string(),
            });
        }
        buffer.extend_from_slice(&chunk[..read]);
    };

    let header_bytes = &buffer[..header_end];
    let headers =
        std::str::from_utf8(header_bytes).map_err(|error| RuntimeError::NetworkProtocol {
            reason: format!("HTTP headers are not UTF-8: {error}"),
        })?;
    let mut lines = headers.split("\r\n");
    let request_line = lines.next().ok_or_else(|| RuntimeError::NetworkProtocol {
        reason: "missing HTTP request line".to_string(),
    })?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| RuntimeError::NetworkProtocol {
            reason: "missing HTTP method".to_string(),
        })?
        .to_string();
    let path = request_parts
        .next()
        .ok_or_else(|| RuntimeError::NetworkProtocol {
            reason: "missing HTTP path".to_string(),
        })?
        .to_string();
    let _version = request_parts
        .next()
        .ok_or_else(|| RuntimeError::NetworkProtocol {
            reason: "missing HTTP version".to_string(),
        })?;

    let mut content_length = 0usize;
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("content-length")
        {
            content_length =
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|error| RuntimeError::NetworkProtocol {
                        reason: format!("invalid content-length: {error}"),
                    })?;
        }
    }
    if content_length > max_body_bytes {
        return Err(RuntimeError::NetworkProtocol {
            reason: format!("request body is {content_length} bytes, limit is {max_body_bytes}"),
        });
    }

    let body_start = header_end + 4;
    while buffer.len() < body_start + content_length {
        let mut chunk = [0u8; 8192];
        let read = stream
            .read(&mut chunk)
            .map_err(|e| network_io_error("read network request body", e))?;
        if read == 0 {
            return Err(RuntimeError::NetworkProtocol {
                reason: "connection closed before HTTP body completed".to_string(),
            });
        }
        buffer.extend_from_slice(&chunk[..read]);
    }

    Ok(RuntimeHttpRequest {
        method,
        path,
        body: buffer[body_start..body_start + content_length].to_vec(),
    })
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn write_json_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    body: &serde_json::Value,
) -> Result<()> {
    let body = serde_json::to_vec(body).map_err(network_json_error)?;
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|()| stream.write_all(&body))
        .map_err(|e| network_io_error("write network response", e))
}

fn network_error_status(error: &RuntimeError) -> (u16, &'static str, &'static str) {
    match error {
        RuntimeError::NetworkRequestNotFound { .. } => (404, "Not Found", "not_found"),
        RuntimeError::NetworkRequestAlreadyActive { .. } => (409, "Conflict", "conflict"),
        RuntimeError::ExecutorQueueFull { .. }
        | RuntimeError::ExecutorQueueBytesExceeded { .. } => {
            (429, "Too Many Requests", "admission_rejected")
        }
        RuntimeError::NetworkProtocol { .. } => (400, "Bad Request", "bad_request"),
        RuntimeError::RequestDeadlineExceeded { .. } => {
            (408, "Request Timeout", "deadline_expired")
        }
        RuntimeError::RequestCancelled { .. } => (499, "Client Closed Request", "cancelled"),
        _ => (500, "Internal Server Error", "runtime_error"),
    }
}

fn network_io_error(operation: &'static str, error: std::io::Error) -> RuntimeError {
    RuntimeError::NetworkIo {
        operation,
        message: error.to_string(),
    }
}

fn network_json_error(error: serde_json::Error) -> RuntimeError {
    RuntimeError::NetworkProtocol {
        reason: error.to_string(),
    }
}

struct RuntimeExecutorInner {
    engine: Arc<Engine>,
    state: Mutex<RuntimeExecutorState>,
    available: Condvar,
    queue_capacity: usize,
    dynamic_batching: Option<DynamicBatchingConfig>,
    admission: RuntimeAdmissionConfig,
}

impl RuntimeExecutorInner {
    fn lock_state(&self) -> Result<MutexGuard<'_, RuntimeExecutorState>> {
        self.state
            .lock()
            .map_err(|_| RuntimeError::ExecutorPoisoned)
    }

    fn pop_ready_batch(&self) -> Result<Option<Vec<QueuedRuntimeRequest>>> {
        let mut state = self.lock_state()?;
        Ok(pop_ready_batch_from_state(
            &mut state,
            self.dynamic_batching.as_ref(),
            &self.admission,
        ))
    }

    fn pop_blocking_batch(&self) -> Option<Vec<QueuedRuntimeRequest>> {
        let mut state = self.state.lock().ok()?;
        loop {
            if let Some(queued) = state.pop_queued() {
                let Some(config) = &self.dynamic_batching else {
                    return Some(vec![queued]);
                };
                return collect_blocking_dynamic_batch(self, state, config, vec![queued]);
            }
            if state.closed {
                return None;
            }
            state = self.available.wait(state).ok()?;
        }
    }

    fn record_completed_batch(
        &self,
        queue_times: &[Duration],
        run_time: Duration,
        batch_size: usize,
    ) {
        if let Ok(mut state) = self.state.lock() {
            state.metrics.completed = state
                .metrics
                .completed
                .saturating_add(queue_times.len() as u64);
            state.metrics.runtime_invocations = state.metrics.runtime_invocations.saturating_add(1);
            if batch_size > 1 {
                state.metrics.batches_executed = state.metrics.batches_executed.saturating_add(1);
                state.metrics.batched_requests = state
                    .metrics
                    .batched_requests
                    .saturating_add(queue_times.len() as u64);
            }
            for &queue_time in queue_times {
                state.metrics.total_queue_time += queue_time;
                state.metrics.total_run_time += run_time;
            }
        }
    }

    fn record_failed(
        &self,
        queue_time: Duration,
        run_time: Duration,
        deadline_expired: bool,
        runtime_invoked: bool,
    ) {
        if let Ok(mut state) = self.state.lock() {
            state.metrics.failed = state.metrics.failed.saturating_add(1);
            if deadline_expired {
                state.metrics.deadline_expired = state.metrics.deadline_expired.saturating_add(1);
            }
            if runtime_invoked {
                state.metrics.runtime_invocations =
                    state.metrics.runtime_invocations.saturating_add(1);
            }
            state.metrics.total_queue_time += queue_time;
            state.metrics.total_run_time += run_time;
        }
    }

    fn record_batch_fallback(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.metrics.batch_fallbacks = state.metrics.batch_fallbacks.saturating_add(1);
        }
    }

    fn record_runtime_start(&self, batch_size: usize) {
        if let Ok(mut state) = self.state.lock() {
            state.metrics.active_runtime_invocations =
                state.metrics.active_runtime_invocations.saturating_add(1);
            state.metrics.max_active_runtime_invocations = state
                .metrics
                .max_active_runtime_invocations
                .max(state.metrics.active_runtime_invocations);
            state.metrics.active_requests =
                state.metrics.active_requests.saturating_add(batch_size);
            state.metrics.max_active_requests = state
                .metrics
                .max_active_requests
                .max(state.metrics.active_requests);
            state.metrics.active_batch_size =
                state.metrics.active_batch_size.saturating_add(batch_size);
            state.metrics.max_active_batch_size = state
                .metrics
                .max_active_batch_size
                .max(state.metrics.active_batch_size);
        }
    }

    fn record_runtime_finish(&self, batch_size: usize) {
        if let Ok(mut state) = self.state.lock() {
            state.metrics.active_runtime_invocations =
                state.metrics.active_runtime_invocations.saturating_sub(1);
            state.metrics.active_requests =
                state.metrics.active_requests.saturating_sub(batch_size);
            state.metrics.active_batch_size =
                state.metrics.active_batch_size.saturating_sub(batch_size);
        }
    }

    fn record_cancelled(&self, queue_time: Duration) {
        if let Ok(mut state) = self.state.lock() {
            state.metrics.failed = state.metrics.failed.saturating_add(1);
            state.metrics.cancelled = state.metrics.cancelled.saturating_add(1);
            state.metrics.total_queue_time += queue_time;
        }
    }
}

struct RuntimeExecutorState {
    queue: BinaryHeap<QueuedRuntimeRequest>,
    closed: bool,
    next_sequence: u64,
    metrics: RuntimeExecutorMetrics,
}

impl RuntimeExecutorState {
    fn push_queued(&mut self, queued: QueuedRuntimeRequest) {
        self.metrics.current_queue_depth = self.metrics.current_queue_depth.saturating_add(1);
        self.metrics.max_observed_queue_depth = self
            .metrics
            .max_observed_queue_depth
            .max(self.metrics.current_queue_depth);
        self.metrics.current_queued_tensor_bytes = self
            .metrics
            .current_queued_tensor_bytes
            .saturating_add(queued.input_bytes);
        self.metrics.max_observed_queued_tensor_bytes = self
            .metrics
            .max_observed_queued_tensor_bytes
            .max(self.metrics.current_queued_tensor_bytes);
        self.queue.push(queued);
    }

    fn pop_queued(&mut self) -> Option<QueuedRuntimeRequest> {
        let queued = self.queue.pop()?;
        self.metrics.current_queue_depth = self.metrics.current_queue_depth.saturating_sub(1);
        self.metrics.current_queued_tensor_bytes = self
            .metrics
            .current_queued_tensor_bytes
            .saturating_sub(queued.input_bytes);
        Some(queued)
    }

    fn record_batch_flush(&mut self, reason: BatchFlushReason) {
        match reason {
            BatchFlushReason::Full => {
                self.metrics.batch_flushes_full = self.metrics.batch_flushes_full.saturating_add(1);
            }
            BatchFlushReason::Delay => {
                self.metrics.batch_flushes_delay =
                    self.metrics.batch_flushes_delay.saturating_add(1);
            }
            BatchFlushReason::MemoryPressure => {
                self.metrics.batch_flushes_memory_pressure =
                    self.metrics.batch_flushes_memory_pressure.saturating_add(1);
            }
            BatchFlushReason::Manual => {
                self.metrics.batch_flushes_manual =
                    self.metrics.batch_flushes_manual.saturating_add(1);
            }
            BatchFlushReason::Shutdown => {
                self.metrics.batch_flushes_shutdown =
                    self.metrics.batch_flushes_shutdown.saturating_add(1);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchFlushReason {
    Full,
    Delay,
    MemoryPressure,
    Manual,
    Shutdown,
}

struct QueuedRuntimeRequest {
    priority: i32,
    sequence: u64,
    queued_at: Instant,
    request: RuntimeRequest,
    cancellation: RuntimeCancellationToken,
    input_bytes: usize,
    sender: Sender<Result<RuntimeResponse>>,
}

impl PartialEq for QueuedRuntimeRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for QueuedRuntimeRequest {}

impl PartialOrd for QueuedRuntimeRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedRuntimeRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

fn worker_loop(inner: Arc<RuntimeExecutorInner>) {
    while let Some(batch) = inner.pop_blocking_batch() {
        execute_queued_batch(&inner, batch);
    }
}

fn pop_ready_batch_from_state(
    state: &mut RuntimeExecutorState,
    config: Option<&DynamicBatchingConfig>,
    admission: &RuntimeAdmissionConfig,
) -> Option<Vec<QueuedRuntimeRequest>> {
    let queued = state.pop_queued()?;
    let mut batch = vec![queued];
    if let Some(config) = config {
        extend_batch_from_state(state, Some(config), &mut batch);
        let reason =
            dynamic_batch_flush_reason(state, &batch, config, admission, BatchFlushReason::Manual);
        state.record_batch_flush(reason);
    }
    Some(batch)
}

fn collect_blocking_dynamic_batch(
    inner: &RuntimeExecutorInner,
    mut state: MutexGuard<'_, RuntimeExecutorState>,
    config: &DynamicBatchingConfig,
    mut batch: Vec<QueuedRuntimeRequest>,
) -> Option<Vec<QueuedRuntimeRequest>> {
    let opened_at = Instant::now();
    loop {
        extend_batch_from_state(&mut state, Some(config), &mut batch);
        if batch.len() >= config.max_batch_size {
            state.record_batch_flush(BatchFlushReason::Full);
            return Some(batch);
        }
        if dynamic_batch_memory_pressure(&state, &batch, &inner.admission) {
            state.record_batch_flush(BatchFlushReason::MemoryPressure);
            return Some(batch);
        }
        if state.closed {
            state.record_batch_flush(BatchFlushReason::Shutdown);
            return Some(batch);
        }
        if config.max_queue_delay.is_zero() {
            state.record_batch_flush(BatchFlushReason::Delay);
            return Some(batch);
        }
        let elapsed = opened_at.elapsed();
        if elapsed >= config.max_queue_delay {
            state.record_batch_flush(BatchFlushReason::Delay);
            return Some(batch);
        }
        let wait_for = config.max_queue_delay - elapsed;
        let (next_state, timed_out) = match inner.available.wait_timeout(state, wait_for) {
            Ok((state, timeout)) => (state, timeout.timed_out()),
            Err(poisoned) => {
                let (state, timeout) = poisoned.into_inner();
                (state, timeout.timed_out())
            }
        };
        state = next_state;
        if timed_out {
            extend_batch_from_state(&mut state, Some(config), &mut batch);
            let reason = dynamic_batch_flush_reason(
                &state,
                &batch,
                config,
                &inner.admission,
                BatchFlushReason::Delay,
            );
            state.record_batch_flush(reason);
            return Some(batch);
        }
    }
}

fn dynamic_batch_flush_reason(
    state: &RuntimeExecutorState,
    batch: &[QueuedRuntimeRequest],
    config: &DynamicBatchingConfig,
    admission: &RuntimeAdmissionConfig,
    default_reason: BatchFlushReason,
) -> BatchFlushReason {
    if batch.len() >= config.max_batch_size {
        BatchFlushReason::Full
    } else if dynamic_batch_memory_pressure(state, batch, admission) {
        BatchFlushReason::MemoryPressure
    } else {
        default_reason
    }
}

fn dynamic_batch_memory_pressure(
    state: &RuntimeExecutorState,
    batch: &[QueuedRuntimeRequest],
    admission: &RuntimeAdmissionConfig,
) -> bool {
    let Some(capacity_bytes) = admission.max_queued_tensor_bytes else {
        return false;
    };
    let batch_bytes = batch.iter().fold(0usize, |total, queued| {
        total.saturating_add(queued.input_bytes)
    });
    state
        .metrics
        .current_queued_tensor_bytes
        .saturating_add(batch_bytes)
        >= capacity_bytes
}

fn extend_batch_from_state(
    state: &mut RuntimeExecutorState,
    config: Option<&DynamicBatchingConfig>,
    batch: &mut Vec<QueuedRuntimeRequest>,
) {
    let Some(config) = config else {
        return;
    };
    if config.max_batch_size <= 1 {
        return;
    }

    let mut skipped = Vec::new();
    while batch.len() < config.max_batch_size {
        let Some(candidate) = state.pop_queued() else {
            break;
        };
        if can_batch_queued_requests(&batch[0], &candidate) {
            batch.push(candidate);
        } else {
            skipped.push(candidate);
        }
    }
    for skipped in skipped {
        state.push_queued(skipped);
    }
}

fn execute_queued_batch(inner: &RuntimeExecutorInner, batch: Vec<QueuedRuntimeRequest>) {
    let prepared = prepare_queued_batch(inner, batch);
    if prepared.is_empty() {
        return;
    }
    if prepared.len() == 1 {
        if let Some(prepared) = prepared.into_iter().next() {
            execute_prepared_single(inner, prepared);
        }
        return;
    }

    match run_prepared_batch(inner, &prepared) {
        Ok(()) => {}
        Err(()) => {
            inner.record_batch_fallback();
            for prepared in prepared {
                execute_prepared_single(inner, prepared);
            }
        }
    }
}

fn prepare_queued_batch(
    inner: &RuntimeExecutorInner,
    batch: Vec<QueuedRuntimeRequest>,
) -> Vec<PreparedRuntimeRequest> {
    let started_at = Instant::now();
    batch
        .into_iter()
        .filter_map(|queued| prepare_queued_request(inner, queued, started_at))
        .collect()
}

fn prepare_queued_request(
    inner: &RuntimeExecutorInner,
    queued: QueuedRuntimeRequest,
    started_at: Instant,
) -> Option<PreparedRuntimeRequest> {
    let queue_time = started_at.saturating_duration_since(queued.queued_at);
    if queued.cancellation.try_mark_running().is_err() {
        queued.cancellation.mark_completed();
        inner.record_cancelled(queue_time);
        let _ = queued.sender.send(Err(RuntimeError::RequestCancelled {
            request_id: queued.request.request_id,
        }));
        return None;
    }

    if queued
        .request
        .deadline
        .is_some_and(|deadline| deadline <= started_at)
    {
        queued.cancellation.mark_completed();
        inner.record_failed(queue_time, Duration::ZERO, true, false);
        let _ = queued
            .sender
            .send(Err(RuntimeError::RequestDeadlineExceeded {
                request_id: queued.request.request_id,
            }));
        return None;
    }

    Some(PreparedRuntimeRequest {
        queued_at: queued.queued_at,
        request: queued.request,
        cancellation: queued.cancellation,
        sender: queued.sender,
    })
}

struct PreparedRuntimeRequest {
    queued_at: Instant,
    request: RuntimeRequest,
    cancellation: RuntimeCancellationToken,
    sender: Sender<Result<RuntimeResponse>>,
}

fn execute_prepared_single(inner: &RuntimeExecutorInner, prepared: PreparedRuntimeRequest) {
    let started_at = Instant::now();
    let queue_time = started_at.saturating_duration_since(prepared.queued_at);
    let request_id = prepared.request.request_id;
    inner.record_runtime_start(1);
    let result = inner.engine.run_with_cancellations(
        prepared.request.graph_idx,
        prepared.request.options,
        prepared.request.inputs,
        &[&prepared.cancellation],
    );
    inner.record_runtime_finish(1);
    let finished_at = Instant::now();
    let run_time = finished_at.saturating_duration_since(started_at);
    prepared.cancellation.mark_completed();
    match result {
        Ok(outputs) => {
            inner.record_completed_batch(&[queue_time], run_time, 1);
            let _ = prepared.sender.send(Ok(RuntimeResponse {
                request_id,
                outputs,
                timings: RuntimeRequestTimings {
                    queued_at: prepared.queued_at,
                    started_at,
                    finished_at,
                    queue_time,
                    run_time,
                },
            }));
        }
        Err(error) => {
            inner.record_failed(queue_time, run_time, false, true);
            let _ = prepared.sender.send(Err(error));
        }
    }
}

fn can_batch_queued_requests(
    first: &QueuedRuntimeRequest,
    candidate: &QueuedRuntimeRequest,
) -> bool {
    first.priority == candidate.priority
        && first.request.graph_idx == candidate.request.graph_idx
        && first.request.options == candidate.request.options
        && inputs_are_batch_compatible(&first.request.inputs, &candidate.request.inputs)
}

fn inputs_are_batch_compatible(first: &RuntimeInputs, candidate: &RuntimeInputs) -> bool {
    if first.len() != candidate.len()
        || request_leading_batch_size(first).is_none()
        || request_leading_batch_size(candidate).is_none()
    {
        return false;
    }
    first.iter().all(|(name, tensor)| {
        candidate
            .get(name)
            .is_some_and(|other| tensors_are_batch_compatible(tensor, other))
    })
}

fn request_leading_batch_size(inputs: &RuntimeInputs) -> Option<usize> {
    let mut leading = None;
    for tensor in inputs.values() {
        let shape = runtime_tensor_shape(tensor);
        let &first_dim = shape.first()?;
        if leading.is_some_and(|existing| existing != first_dim) {
            return None;
        }
        leading = Some(first_dim);
    }
    leading
}

fn tensors_are_batch_compatible(first: &RuntimeTensor, candidate: &RuntimeTensor) -> bool {
    match (first, candidate) {
        (
            RuntimeTensor::F32 {
                shape: first_shape, ..
            },
            RuntimeTensor::F32 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::F64 {
                shape: first_shape, ..
            },
            RuntimeTensor::F64 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::I64 {
                shape: first_shape, ..
            },
            RuntimeTensor::I64 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::I32 {
                shape: first_shape, ..
            },
            RuntimeTensor::I32 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::U8 {
                shape: first_shape, ..
            },
            RuntimeTensor::U8 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::I8 {
                shape: first_shape, ..
            },
            RuntimeTensor::I8 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::Bool {
                shape: first_shape, ..
            },
            RuntimeTensor::Bool {
                shape: candidate_shape,
                ..
            },
        ) => {
            !first_shape.is_empty()
                && first_shape.len() == candidate_shape.len()
                && first_shape[1..] == candidate_shape[1..]
        }
        _ => false,
    }
}

fn run_prepared_batch(
    inner: &RuntimeExecutorInner,
    prepared: &[PreparedRuntimeRequest],
) -> std::result::Result<(), ()> {
    let (inputs, batch_sizes) = merge_prepared_inputs(prepared).map_err(|_| ())?;
    let graph_idx = prepared[0].request.graph_idx;
    let options = prepared[0].request.options.clone();
    let cancellations = prepared
        .iter()
        .map(|request| &request.cancellation)
        .collect::<Vec<_>>();
    let started_at = Instant::now();
    inner.record_runtime_start(prepared.len());
    let outputs = inner
        .engine
        .run_with_cancellations(graph_idx, options, inputs, &cancellations);
    inner.record_runtime_finish(prepared.len());
    let outputs = outputs.map_err(|_| ())?;
    let finished_at = Instant::now();
    let run_time = finished_at.saturating_duration_since(started_at);
    let split_outputs = split_runtime_outputs(outputs, &batch_sizes).map_err(|_| ())?;
    let queue_times = prepared
        .iter()
        .map(|request| started_at.saturating_duration_since(request.queued_at))
        .collect::<Vec<_>>();
    inner.record_completed_batch(&queue_times, run_time, prepared.len());

    for ((prepared, outputs), queue_time) in prepared.iter().zip(split_outputs).zip(queue_times) {
        prepared.cancellation.mark_completed();
        let _ = prepared.sender.send(Ok(RuntimeResponse {
            request_id: prepared.request.request_id.clone(),
            outputs,
            timings: RuntimeRequestTimings {
                queued_at: prepared.queued_at,
                started_at,
                finished_at,
                queue_time,
                run_time,
            },
        }));
    }
    Ok(())
}

fn merge_prepared_inputs(
    prepared: &[PreparedRuntimeRequest],
) -> Result<(RuntimeInputs, Vec<usize>)> {
    let first = prepared
        .first()
        .ok_or_else(|| RuntimeError::Shape("cannot batch zero requests".to_string()))?;
    let batch_sizes = prepared
        .iter()
        .map(|request| {
            request_leading_batch_size(&request.request.inputs).ok_or_else(|| {
                RuntimeError::Shape(format!(
                    "request {} has no leading batch dimension",
                    request.request.request_id
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut inputs = RuntimeInputs::with_capacity(first.request.inputs.len());
    for name in first.request.inputs.keys() {
        let tensors = prepared
            .iter()
            .map(|request| {
                request.request.inputs.get(name).ok_or_else(|| {
                    RuntimeError::Shape(format!(
                        "request {} is missing batched input {name}",
                        request.request.request_id
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        inputs.insert(name.clone(), merge_runtime_tensors(name, &tensors)?);
    }
    Ok((inputs, batch_sizes))
}

fn split_runtime_outputs(
    outputs: RuntimeOutputs,
    batch_sizes: &[usize],
) -> Result<Vec<RuntimeOutputs>> {
    let mut per_request = (0..batch_sizes.len())
        .map(|_| RuntimeOutputs::new())
        .collect::<Vec<_>>();
    for (name, tensor) in outputs {
        let split = split_runtime_tensor(&name, tensor, batch_sizes)?;
        for (request_outputs, tensor) in per_request.iter_mut().zip(split) {
            request_outputs.insert(name.clone(), tensor);
        }
    }
    Ok(per_request)
}

fn runtime_tensor_shape(tensor: &RuntimeTensor) -> &[usize] {
    match tensor {
        RuntimeTensor::F32 { shape, .. }
        | RuntimeTensor::F64 { shape, .. }
        | RuntimeTensor::I64 { shape, .. }
        | RuntimeTensor::I32 { shape, .. }
        | RuntimeTensor::U8 { shape, .. }
        | RuntimeTensor::I8 { shape, .. }
        | RuntimeTensor::Bool { shape, .. } => shape,
    }
}

fn runtime_inputs_data_bytes(inputs: &RuntimeInputs) -> Result<usize> {
    inputs.values().try_fold(0usize, |acc, tensor| {
        acc.checked_add(runtime_tensor_data_bytes(tensor)?)
            .ok_or_else(|| RuntimeError::Shape("runtime input byte count overflow".to_string()))
    })
}

fn runtime_tensor_data_bytes(tensor: &RuntimeTensor) -> Result<usize> {
    match tensor {
        RuntimeTensor::F32 { data, .. } => runtime_tensor_data_bytes_for::<f32>(data.len()),
        RuntimeTensor::F64 { data, .. } => runtime_tensor_data_bytes_for::<f64>(data.len()),
        RuntimeTensor::I64 { data, .. } => runtime_tensor_data_bytes_for::<i64>(data.len()),
        RuntimeTensor::I32 { data, .. } => runtime_tensor_data_bytes_for::<i32>(data.len()),
        RuntimeTensor::U8 { data, .. } => runtime_tensor_data_bytes_for::<u8>(data.len()),
        RuntimeTensor::I8 { data, .. } => runtime_tensor_data_bytes_for::<i8>(data.len()),
        RuntimeTensor::Bool { data, .. } => runtime_tensor_data_bytes_for::<bool>(data.len()),
    }
}

fn runtime_tensor_data_bytes_for<T>(len: usize) -> Result<usize> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| RuntimeError::Shape("runtime tensor byte count overflow".to_string()))
}

fn leading_batch_parts(name: &str, shape: &[usize], data_len: usize) -> Result<(usize, usize)> {
    let Some((&batch_size, trailing_shape)) = shape.split_first() else {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} has no leading batch dimension"
        )));
    };
    let trailing_elements = trailing_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| RuntimeError::Shape(format!("tensor {name} shape overflows usize")))
    })?;
    let expected = batch_size.checked_mul(trailing_elements).ok_or_else(|| {
        RuntimeError::Shape(format!("tensor {name} element count overflows usize"))
    })?;
    if expected != data_len {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} implies {expected} elements, got {data_len}",
            shape
        )));
    }
    Ok((batch_size, trailing_elements))
}

macro_rules! merge_runtime_tensor_variant {
    ($name:expr, $tensors:expr, $variant:ident) => {{
        let mut merged_shape = Vec::new();
        let mut merged_data = Vec::new();
        for (idx, tensor) in $tensors.iter().enumerate() {
            let RuntimeTensor::$variant { shape, data } = tensor else {
                return Err(RuntimeError::Shape(format!(
                    "input {} has mixed tensor dtypes for batching",
                    $name
                )));
            };
            let (batch_size, _) = leading_batch_parts($name, shape, data.len())?;
            if idx == 0 {
                merged_shape = shape.clone();
                merged_shape[0] = 0;
            } else if shape[1..] != merged_shape[1..] {
                return Err(RuntimeError::Shape(format!(
                    "input {} has incompatible trailing dimensions for batching",
                    $name
                )));
            }
            merged_shape[0] = merged_shape[0].checked_add(batch_size).ok_or_else(|| {
                RuntimeError::Shape(format!("input {} batch dimension overflows usize", $name))
            })?;
            merged_data.extend_from_slice(data);
        }
        Ok(RuntimeTensor::$variant {
            shape: merged_shape,
            data: merged_data,
        })
    }};
}

fn merge_runtime_tensors(name: &str, tensors: &[&RuntimeTensor]) -> Result<RuntimeTensor> {
    let Some(first) = tensors.first() else {
        return Err(RuntimeError::Shape(format!(
            "input {name} has no tensors to batch"
        )));
    };
    match first {
        RuntimeTensor::F32 { .. } => merge_runtime_tensor_variant!(name, tensors, F32),
        RuntimeTensor::F64 { .. } => merge_runtime_tensor_variant!(name, tensors, F64),
        RuntimeTensor::I64 { .. } => merge_runtime_tensor_variant!(name, tensors, I64),
        RuntimeTensor::I32 { .. } => merge_runtime_tensor_variant!(name, tensors, I32),
        RuntimeTensor::U8 { .. } => merge_runtime_tensor_variant!(name, tensors, U8),
        RuntimeTensor::I8 { .. } => merge_runtime_tensor_variant!(name, tensors, I8),
        RuntimeTensor::Bool { .. } => merge_runtime_tensor_variant!(name, tensors, Bool),
    }
}

macro_rules! split_runtime_tensor_variant {
    ($name:expr, $shape:expr, $data:expr, $batch_sizes:expr, $variant:ident) => {{
        let total_batch = $batch_sizes.iter().try_fold(0usize, |acc, &batch_size| {
            acc.checked_add(batch_size).ok_or_else(|| {
                RuntimeError::Shape(format!("output {} batch dimension overflows usize", $name))
            })
        })?;
        let (batch_size, trailing_elements) = leading_batch_parts($name, &$shape, $data.len())?;
        if batch_size != total_batch {
            return Err(RuntimeError::Shape(format!(
                "output {} leading batch dimension is {batch_size}, expected {total_batch}",
                $name
            )));
        }
        let mut offset = 0usize;
        let mut split = Vec::with_capacity($batch_sizes.len());
        for &request_batch_size in $batch_sizes {
            let len = request_batch_size
                .checked_mul(trailing_elements)
                .ok_or_else(|| {
                    RuntimeError::Shape(format!(
                        "output {} split element count overflows usize",
                        $name
                    ))
                })?;
            let end = offset.checked_add(len).ok_or_else(|| {
                RuntimeError::Shape(format!("output {} split range overflows usize", $name))
            })?;
            let mut shape = $shape.clone();
            shape[0] = request_batch_size;
            split.push(RuntimeTensor::$variant {
                shape,
                data: $data[offset..end].to_vec(),
            });
            offset = end;
        }
        Ok(split)
    }};
}

fn split_runtime_tensor(
    name: &str,
    tensor: RuntimeTensor,
    batch_sizes: &[usize],
) -> Result<Vec<RuntimeTensor>> {
    match tensor {
        RuntimeTensor::F32 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, F32)
        }
        RuntimeTensor::F64 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, F64)
        }
        RuntimeTensor::I64 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, I64)
        }
        RuntimeTensor::I32 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, I32)
        }
        RuntimeTensor::U8 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, U8)
        }
        RuntimeTensor::I8 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, I8)
        }
        RuntimeTensor::Bool { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, Bool)
        }
    }
}

/// Helper to convert an RSMF `TensorView` into an `ndarray::ArrayViewD`.
pub fn tensor_view_to_ndarray<'a>(view: &'a TensorView<'a>) -> Result<ArrayViewD<'a, f32>> {
    if view.dtype() != LogicalDtype::F32 {
        return Err(RuntimeError::UnsupportedDtype(format!(
            "only F32 tensors support zero-copy ndarray conversion, got {:?}",
            view.dtype()
        )));
    }

    let shape: Vec<usize> = view
        .shape()
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                RuntimeError::Shape(format!("tensor dimension {dim} exceeds usize::MAX"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let data = view.as_slice::<f32>()?;

    ArrayViewD::from_shape(shape, data)
        .map_err(|e| RuntimeError::Shape(format!("ndarray shape mismatch: {e}")))
}

fn tensor_from_vec<T>(shape: Vec<usize>, data: Vec<T>) -> Result<DynValue>
where
    T: ort::value::PrimitiveTensorElementType + Clone + Debug + 'static,
{
    let expected = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| RuntimeError::Shape("runtime tensor element count overflow".to_string()))
    })?;
    if expected != data.len() {
        return Err(RuntimeError::Shape(format!(
            "shape {:?} implies {expected} elements, got {}",
            shape,
            data.len()
        )));
    }
    Tensor::<T>::from_array((shape, data.into_boxed_slice()))
        .map(|value| value.into_dyn())
        .map_err(|e| ort_error("tensor value creation", e))
}

fn materialize_outputs(outputs: SessionOutputs<'_>) -> Result<RuntimeOutputs> {
    outputs
        .into_iter()
        .map(|(name, value)| {
            let tensor = match value.dtype().tensor_type() {
                Some(TensorElementType::Float32) => materialize_tensor_f32(&value)?,
                Some(TensorElementType::Float64) => materialize_tensor_f64(&value)?,
                Some(TensorElementType::Int64) => materialize_tensor_i64(&value)?,
                Some(TensorElementType::Int32) => materialize_tensor_i32(&value)?,
                Some(TensorElementType::Uint8) => materialize_tensor_u8(&value)?,
                Some(TensorElementType::Int8) => materialize_tensor_i8(&value)?,
                Some(TensorElementType::Bool) => materialize_tensor_bool(&value)?,
                Some(other) => {
                    return Err(RuntimeError::UnsupportedDtype(other.to_string()));
                }
                None => {
                    return Err(RuntimeError::UnsupportedDtype(value.dtype().to_string()));
                }
            };
            Ok((name.to_string(), tensor))
        })
        .collect()
}

fn shape_to_usize(shape: &[i64]) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                RuntimeError::Shape(format!("output dimension {dim} cannot convert to usize"))
            })
        })
        .collect()
}

fn shape_u64_to_usize(shape: &[u64]) -> std::result::Result<Vec<usize>, String> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| format!("dimension {dim} cannot convert to usize"))
        })
        .collect()
}

fn shape_u64_to_i64(shape: &[u64]) -> std::result::Result<Vec<i64>, String> {
    shape
        .iter()
        .map(|&dim| {
            i64::try_from(dim).map_err(|_| format!("dimension {dim} cannot convert to i64"))
        })
        .collect()
}

fn validate_initializer_view(binding: &InitializerBinding, view: &TensorView<'_>) -> Result<()> {
    let reason = if !is_supported_initializer_dtype(view.dtype()) {
        Some(format!("unsupported initializer dtype {:?}", view.dtype()))
    } else if view.encoding != EncodingKind::Raw {
        Some(format!(
            "only raw canonical initializers are supported, got {:?}",
            view.encoding
        ))
    } else if view.layout != LayoutTag::RowMajor {
        Some(format!(
            "only row-major initializers are supported, got {:?}",
            view.layout
        ))
    } else if view.storage_dtype != StorageDtype::Logical(view.dtype()) {
        Some(format!(
            "initializer storage {:?} does not match logical dtype {:?}",
            view.storage_dtype,
            view.dtype()
        ))
    } else {
        None
    };

    if let Some(reason) = reason {
        Err(RuntimeError::UnsupportedInitializer {
            initializer_name: binding.initializer_name.clone(),
            tensor_name: binding.tensor_name.clone(),
            reason,
        })
    } else {
        Ok(())
    }
}

fn is_supported_initializer_dtype(dtype: LogicalDtype) -> bool {
    matches!(
        dtype,
        LogicalDtype::F32
            | LogicalDtype::F64
            | LogicalDtype::I64
            | LogicalDtype::I32
            | LogicalDtype::U8
            | LogicalDtype::I8
            | LogicalDtype::Bool
    )
}

fn initializer_tensor_from_view(
    binding: &InitializerBinding,
    view: &TensorView<'_>,
    shape: Vec<usize>,
) -> Result<(Arc<DynValue>, usize)> {
    match view.dtype() {
        LogicalDtype::F32 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<f32>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::F64 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<f64>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::I64 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<i64>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::I32 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<i32>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::U8 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<u8>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::I8 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<i8>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::Bool => {
            let data = view
                .bytes()
                .iter()
                .map(|&byte| byte != 0)
                .collect::<Vec<_>>();
            initializer_tensor_from_vec(binding, &shape, data)
        }
        other => Err(RuntimeError::UnsupportedInitializer {
            initializer_name: binding.initializer_name.clone(),
            tensor_name: binding.tensor_name.clone(),
            reason: format!("unsupported initializer dtype {other:?}"),
        }),
    }
}

fn unsupported_initializer(binding: &InitializerBinding, reason: String) -> RuntimeError {
    RuntimeError::UnsupportedInitializer {
        initializer_name: binding.initializer_name.clone(),
        tensor_name: binding.tensor_name.clone(),
        reason,
    }
}

fn initializer_tensor_from_vec<T>(
    binding: &InitializerBinding,
    shape: &[usize],
    data: Vec<T>,
) -> Result<(Arc<DynValue>, usize)>
where
    T: ort::value::PrimitiveTensorElementType + Clone + Debug + 'static,
{
    let materialized_bytes = data
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| RuntimeError::UnsupportedInitializer {
            initializer_name: binding.initializer_name.clone(),
            tensor_name: binding.tensor_name.clone(),
            reason: "initializer materialized byte count overflow".to_string(),
        })?;
    let value = tensor_from_vec(shape.to_vec(), data).map(Arc::new)?;
    Ok((value, materialized_bytes))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OnnxInitializerInfo {
    data_type: Option<OnnxTensorDataType>,
    shape: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnnxTensorDataType {
    Float,
    Uint8,
    Int8,
    Int32,
    Int64,
    Bool,
    Double,
    Other(i32),
}

impl OnnxTensorDataType {
    fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            6 => Self::Int32,
            7 => Self::Int64,
            9 => Self::Bool,
            11 => Self::Double,
            other => Self::Other(other),
        }
    }

    fn from_logical_dtype(dtype: LogicalDtype) -> Option<Self> {
        match dtype {
            LogicalDtype::F32 => Some(Self::Float),
            LogicalDtype::F64 => Some(Self::Double),
            LogicalDtype::I64 => Some(Self::Int64),
            LogicalDtype::I32 => Some(Self::Int32),
            LogicalDtype::U8 => Some(Self::Uint8),
            LogicalDtype::I8 => Some(Self::Int8),
            LogicalDtype::Bool => Some(Self::Bool),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Float => "Float/F32",
            Self::Uint8 => "Uint8",
            Self::Int8 => "Int8",
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::Bool => "Bool",
            Self::Double => "Double/F64",
            Self::Other(4) => "Uint16",
            Self::Other(5) => "Int16",
            Self::Other(10) => "Float16",
            Self::Other(16) => "BFloat16",
            Self::Other(_) => "unknown",
        }
    }
}

fn onnx_initializers(
    graph_bytes: &[u8],
) -> std::result::Result<HashMap<String, OnnxInitializerInfo>, String> {
    let graph = onnx_graph(graph_bytes)?
        .ok_or_else(|| "ONNX model does not contain a graph".to_string())?;
    graph_initializers(graph)
}

fn onnx_graph(model: &[u8]) -> std::result::Result<Option<&[u8]>, String> {
    let mut reader = ProtoReader::new(model);
    let mut graph = None;
    while let Some(field) = reader.next_field()? {
        if field.number == 7 && field.wire_type == PROTO_LEN {
            graph = Some(field.bytes);
        }
    }
    Ok(graph)
}

fn graph_initializers(
    graph: &[u8],
) -> std::result::Result<HashMap<String, OnnxInitializerInfo>, String> {
    let mut reader = ProtoReader::new(graph);
    let mut initializers = HashMap::new();
    while let Some(field) = reader.next_field()? {
        if field.number == 5 && field.wire_type == PROTO_LEN {
            let Some((name, info)) = tensor_proto_initializer(field.bytes)? else {
                continue;
            };
            initializers.insert(name, info);
        }
    }
    Ok(initializers)
}

fn tensor_proto_initializer(
    tensor: &[u8],
) -> std::result::Result<Option<(String, OnnxInitializerInfo)>, String> {
    let mut reader = ProtoReader::new(tensor);
    let mut name = None;
    let mut shape = Vec::new();
    let mut data_type = None;

    while let Some(field) = reader.next_field()? {
        match (field.number, field.wire_type) {
            (1, PROTO_VARINT) => {
                let dim = i64::try_from(field.varint).map_err(|_| {
                    format!(
                        "ONNX initializer dimension {} exceeds i64::MAX",
                        field.varint
                    )
                })?;
                if dim < 0 {
                    return Err(format!("ONNX initializer dimension {dim} is negative"));
                }
                shape.push(dim);
            }
            (2, PROTO_VARINT) => {
                let raw = i32::try_from(field.varint).map_err(|_| {
                    format!("ONNX initializer dtype {} exceeds i32::MAX", field.varint)
                })?;
                data_type = Some(OnnxTensorDataType::from_raw(raw));
            }
            (8, PROTO_LEN) => {
                name = Some(
                    std::str::from_utf8(field.bytes)
                        .map_err(|e| format!("ONNX initializer name is not UTF-8: {e}"))?
                        .to_string(),
                );
            }
            _ => {}
        }
    }

    Ok(name.map(|name| (name, OnnxInitializerInfo { data_type, shape })))
}

const PROTO_VARINT: u8 = 0;
const PROTO_LEN: u8 = 2;

#[derive(Debug)]
struct ProtoField<'a> {
    number: u32,
    wire_type: u8,
    varint: u64,
    bytes: &'a [u8],
}

struct ProtoReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn next_field(&mut self) -> std::result::Result<Option<ProtoField<'a>>, String> {
        if self.pos == self.bytes.len() {
            return Ok(None);
        }

        let tag = self.read_varint()?;
        let number = u32::try_from(tag >> 3)
            .map_err(|_| format!("protobuf field number {} exceeds u32::MAX", tag >> 3))?;
        let wire_type = (tag & 0x07) as u8;
        if number == 0 {
            return Err("protobuf field number 0 is invalid".to_string());
        }

        match wire_type {
            PROTO_VARINT => {
                let varint = self.read_varint()?;
                Ok(Some(ProtoField {
                    number,
                    wire_type,
                    varint,
                    bytes: &[],
                }))
            }
            PROTO_LEN => {
                let len = usize::try_from(self.read_varint()?)
                    .map_err(|_| "protobuf length exceeds usize::MAX".to_string())?;
                let end = self
                    .pos
                    .checked_add(len)
                    .ok_or_else(|| "protobuf length overflow".to_string())?;
                if end > self.bytes.len() {
                    return Err("protobuf length-delimited field extends past input".to_string());
                }
                let bytes = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(Some(ProtoField {
                    number,
                    wire_type,
                    varint: 0,
                    bytes,
                }))
            }
            1 => {
                self.skip(8)?;
                Ok(Some(ProtoField {
                    number,
                    wire_type,
                    varint: 0,
                    bytes: &[],
                }))
            }
            5 => {
                self.skip(4)?;
                Ok(Some(ProtoField {
                    number,
                    wire_type,
                    varint: 0,
                    bytes: &[],
                }))
            }
            other => Err(format!("unsupported protobuf wire type {other}")),
        }
    }

    fn read_varint(&mut self) -> std::result::Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let Some(&byte) = self.bytes.get(self.pos) else {
                return Err("unterminated protobuf varint".to_string());
            };
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("protobuf varint exceeds 64 bits".to_string())
    }

    fn skip(&mut self, len: usize) -> std::result::Result<(), String> {
        self.pos = self
            .pos
            .checked_add(len)
            .ok_or_else(|| "protobuf skip overflow".to_string())?;
        if self.pos > self.bytes.len() {
            return Err("protobuf fixed-width field extends past input".to_string());
        }
        Ok(())
    }
}

fn materialize_tensor_f32(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| ort_error("extract f32 tensor", e))?;
    Ok(RuntimeTensor::F32 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_f64(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<f64>()
        .map_err(|e| ort_error("extract f64 tensor", e))?;
    Ok(RuntimeTensor::F64 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_i64(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<i64>()
        .map_err(|e| ort_error("extract i64 tensor", e))?;
    Ok(RuntimeTensor::I64 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_i32(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<i32>()
        .map_err(|e| ort_error("extract i32 tensor", e))?;
    Ok(RuntimeTensor::I32 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_u8(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<u8>()
        .map_err(|e| ort_error("extract u8 tensor", e))?;
    Ok(RuntimeTensor::U8 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_i8(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<i8>()
        .map_err(|e| ort_error("extract i8 tensor", e))?;
    Ok(RuntimeTensor::I8 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_bool(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<bool>()
        .map_err(|e| ort_error("extract bool tensor", e))?;
    Ok(RuntimeTensor::Bool {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn runtime_tensor_kind(tensor: &RuntimeTensor) -> &'static str {
    match tensor {
        RuntimeTensor::F32 { .. } => "F32",
        RuntimeTensor::F64 { .. } => "F64",
        RuntimeTensor::I64 { .. } => "I64",
        RuntimeTensor::I32 { .. } => "I32",
        RuntimeTensor::U8 { .. } => "U8",
        RuntimeTensor::I8 { .. } => "I8",
        RuntimeTensor::Bool { .. } => "Bool",
    }
}

fn ort_error(stage: &'static str, error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Ort {
        stage,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
    use rsmf_core::{
        EncodingKind, GraphInput, LayoutTag, LogicalDtype, StorageDtype, TargetTag, VariantMeta,
    };
    use tempfile::tempdir;

    #[test]
    fn session_options_participate_in_cache_key() {
        let default_key = SessionKey::new(0, SessionOptions::default());
        let tuned_key = SessionKey::new(
            0,
            SessionOptions {
                intra_threads: Some(1),
                ..SessionOptions::default()
            },
        );
        assert_ne!(default_key, tuned_key);
    }

    #[test]
    fn tensor_value_shape_mismatch_is_rejected() {
        let err = RuntimeTensor::F32 {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0],
        }
        .into_ort_value()
        .unwrap_err();
        assert!(
            matches!(err, RuntimeError::Shape(message) if message.contains("implies 4 elements"))
        );
    }

    #[test]
    fn missing_graph_index_is_typed_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("no-graph.rsmf");
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "x".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![1],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(1.0f32.to_le_bytes().to_vec()),
                packed: Vec::new(),
            })
            .write_to_path(&path)
            .unwrap();
        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();

        let err = engine
            .session_handle(0, SessionOptions::default())
            .unwrap_err();

        assert!(matches!(
            err,
            RuntimeError::GraphNotFound {
                graph_idx: 0,
                graph_count: 0
            }
        ));
    }

    #[test]
    fn runs_embedded_onnx_add_graph_from_rsmf() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("add.onnx.rsmf");
        let graph = tiny_add_onnx_model();
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "fixture.weight".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![1],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
                packed: Vec::new(),
            })
            .with_graph(GraphInput::onnx(graph.clone()))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let handle = engine.session_handle(0, SessionOptions::default()).unwrap();
        assert_eq!(handle.memory_report().graph_payload_bytes, graph.len());
        assert_eq!(handle.memory_report().initializer_count(), 0);
        assert_eq!(handle.memory_report().initializer_materialized_bytes, 0);
        let input_names = handle
            .inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>();
        let output_names = handle
            .outputs()
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(input_names, vec!["x", "y"]);
        assert_eq!(output_names, vec!["z"]);

        let outputs = engine
            .run_f32(
                0,
                HashMap::from([
                    (
                        "x".to_string(),
                        ArrayD::from_shape_vec(vec![2], vec![1.5, -2.0]).unwrap(),
                    ),
                    (
                        "y".to_string(),
                        ArrayD::from_shape_vec(vec![2], vec![2.5, 3.0]).unwrap(),
                    ),
                ]),
            )
            .unwrap();

        let z = outputs.get("z").unwrap();
        assert_eq!(z.shape(), &[2]);
        assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![4.0, 1.0]);
    }

    #[test]
    fn runs_onnx_graph_with_rsmf_external_initializer() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("add-initializer.onnx.rsmf");
        let graph = tiny_add_external_initializer_onnx_model();
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "bias.tensor".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![2],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
                packed: Vec::new(),
            })
            .with_graph(GraphInput::onnx(graph.clone()))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let options = SessionOptions {
            initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
            ..SessionOptions::default()
        };
        let handle = engine.session_handle(0, options.clone()).unwrap();
        let memory_report = handle.memory_report().clone();
        assert_eq!(memory_report.graph_payload_bytes, graph.len());
        assert_eq!(memory_report.initializer_count(), 1);
        assert_eq!(memory_report.initializer_materialized_bytes, 8);
        assert_eq!(
            memory_report.initializers,
            vec![InitializerMemoryReport {
                initializer_name: "bias".to_string(),
                tensor_name: "bias.tensor".to_string(),
                variant_idx: None,
                materialized_bytes: 8,
            }]
        );
        let cached_handle = engine.session_handle(0, options.clone()).unwrap();
        assert_eq!(cached_handle.memory_report(), &memory_report);
        let input_names = handle
            .inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>();
        let output_names = handle
            .outputs()
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(input_names, vec!["x"]);
        assert_eq!(output_names, vec!["z"]);

        let outputs = engine
            .run_f32_with_options(
                0,
                options,
                HashMap::from([(
                    "x".to_string(),
                    ArrayD::from_shape_vec(vec![2], vec![1.5, 2.0]).unwrap(),
                )]),
            )
            .unwrap();

        let z = outputs.get("z").unwrap();
        assert_eq!(z.shape(), &[2]);
        assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![11.5, -2.0]);
    }

    #[test]
    fn runs_onnx_graph_with_selected_raw_rsmf_initializer_variant() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("add-selected-variant.onnx.rsmf");
        let graph = tiny_add_external_initializer_onnx_model();
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "bias.tensor".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![2],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(f32_bytes(&[0.0, 0.0])),
                packed: vec![raw_variant(
                    StorageDtype::Logical(LogicalDtype::F32),
                    LayoutTag::RowMajor,
                    f32_bytes(&[10.0, -4.0]),
                )],
            })
            .with_graph(GraphInput::onnx(graph.clone()))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let options = SessionOptions {
            initializers: vec![InitializerBinding::new("bias", "bias.tensor").with_variant(1)],
            ..SessionOptions::default()
        };
        let handle = engine.session_handle(0, options.clone()).unwrap();
        assert_eq!(handle.memory_report().initializer_materialized_bytes, 8);
        assert_eq!(
            handle.memory_report().initializers,
            vec![InitializerMemoryReport {
                initializer_name: "bias".to_string(),
                tensor_name: "bias.tensor".to_string(),
                variant_idx: Some(1),
                materialized_bytes: 8,
            }]
        );

        let outputs = engine
            .run_f32_with_options(
                0,
                options,
                HashMap::from([(
                    "x".to_string(),
                    ArrayD::from_shape_vec(vec![2], vec![1.5, 2.0]).unwrap(),
                )]),
            )
            .unwrap();

        let z = outputs.get("z").unwrap();
        assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![11.5, -2.0]);
    }

    #[test]
    fn blocked_initializer_variant_is_typed_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("blocked-initializer-variant.rsmf");
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "bias.tensor".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![2],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(f32_bytes(&[0.0, 0.0])),
                packed: vec![raw_variant(
                    StorageDtype::Logical(LogicalDtype::F32),
                    LayoutTag::Blocked,
                    f32_bytes(&[10.0, -4.0]),
                )],
            })
            .with_graph(GraphInput::onnx(tiny_add_external_initializer_onnx_model()))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let err = engine
            .session_handle(
                0,
                SessionOptions {
                    initializers: vec![
                        InitializerBinding::new("bias", "bias.tensor").with_variant(1),
                    ],
                    ..SessionOptions::default()
                },
            )
            .unwrap_err();

        assert!(
            matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("row-major"))
        );
    }

    #[test]
    fn runs_onnx_graph_with_i64_rsmf_external_initializer() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("add-i64-initializer.onnx.rsmf");
        let graph = tiny_add_external_initializer_onnx_model_with_dtype_shape(7, &[2]);
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "bias.tensor".to_string(),
                dtype: LogicalDtype::I64,
                shape: vec![2],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(i64_bytes(&[10, -4])),
                packed: Vec::new(),
            })
            .with_graph(GraphInput::onnx(graph.clone()))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let options = SessionOptions {
            initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
            ..SessionOptions::default()
        };
        let handle = engine.session_handle(0, options.clone()).unwrap();
        assert_eq!(handle.memory_report().graph_payload_bytes, graph.len());
        assert_eq!(handle.memory_report().initializer_count(), 1);
        assert_eq!(handle.memory_report().initializer_materialized_bytes, 16);

        let outputs = engine
            .run(
                0,
                options,
                HashMap::from([(
                    "x".to_string(),
                    RuntimeTensor::I64 {
                        shape: vec![2],
                        data: vec![2, 9],
                    },
                )]),
            )
            .unwrap();

        assert_eq!(
            outputs.get("z"),
            Some(&RuntimeTensor::I64 {
                shape: vec![2],
                data: vec![12, 5],
            })
        );
    }

    #[test]
    fn missing_initializer_tensor_is_typed_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("missing-initializer.rsmf");
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "fixture.weight".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![1],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
                packed: Vec::new(),
            })
            .with_graph(GraphInput::onnx(tiny_add_external_initializer_onnx_model()))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let err = engine
            .session_handle(
                0,
                SessionOptions {
                    initializers: vec![InitializerBinding::new("bias", "missing.tensor")],
                    ..SessionOptions::default()
                },
            )
            .unwrap_err();

        assert!(matches!(
            err,
            RuntimeError::InitializerTensorNotFound {
                initializer_name,
                tensor_name
            } if initializer_name == "bias" && tensor_name == "missing.tensor"
        ));
    }

    #[test]
    fn initializer_shape_mismatch_is_typed_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shape-mismatch-initializer.rsmf");
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "bias.tensor".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![2],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
                packed: Vec::new(),
            })
            .with_graph(GraphInput::onnx(
                tiny_add_external_initializer_onnx_model_with_dtype_shape(1, &[3]),
            ))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let err = engine
            .session_handle(
                0,
                SessionOptions {
                    initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
                    ..SessionOptions::default()
                },
            )
            .unwrap_err();

        assert!(
            matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("shape"))
        );
    }

    #[test]
    fn initializer_dtype_mismatch_is_typed_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dtype-mismatch-initializer.rsmf");
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "bias.tensor".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![2],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
                packed: Vec::new(),
            })
            .with_graph(GraphInput::onnx(
                tiny_add_external_initializer_onnx_model_with_dtype_shape(7, &[2]),
            ))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let err = engine
            .session_handle(
                0,
                SessionOptions {
                    initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
                    ..SessionOptions::default()
                },
            )
            .unwrap_err();

        assert!(
            matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("dtype"))
        );
    }

    #[test]
    fn executor_runs_same_priority_fifo() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-fifo.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 4,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig::default(),
            },
        );

        let first = executor
            .submit(add_request("first", 1.0, 10.0).with_priority(7))
            .unwrap();
        let second = executor
            .submit(add_request("second", 2.0, 20.0).with_priority(7))
            .unwrap();

        assert!(executor.execute_next().unwrap());
        let first_response = first.receiver.try_recv().unwrap().unwrap();
        assert_eq!(first_response.request_id, "first");
        assert_eq!(f32_output(&first_response, "z"), vec![11.0, 11.0]);
        assert!(matches!(
            second.receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));

        assert!(executor.execute_next().unwrap());
        let second_response = second.wait().unwrap();
        assert_eq!(second_response.request_id, "second");
        assert_eq!(f32_output(&second_response, "z"), vec![22.0, 22.0]);
    }

    #[test]
    fn executor_runs_higher_priority_first() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-priority.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 4,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig::default(),
            },
        );

        let low = executor
            .submit(add_request("low", 1.0, 10.0).with_priority(1))
            .unwrap();
        let high = executor
            .submit(add_request("high", 2.0, 20.0).with_priority(9))
            .unwrap();

        assert!(executor.execute_next().unwrap());
        let high_response = high.receiver.try_recv().unwrap().unwrap();
        assert_eq!(high_response.request_id, "high");
        assert_eq!(f32_output(&high_response, "z"), vec![22.0, 22.0]);
        assert!(matches!(
            low.receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn executor_rejects_expired_deadline_before_runtime_dispatch() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-deadline.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 4,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig::default(),
            },
        );
        let expired = Instant::now() - Duration::from_secs(1);
        let handle = executor
            .submit(RuntimeRequest::new("expired", 99, RuntimeInputs::new()).with_deadline(expired))
            .unwrap();

        assert!(executor.execute_next().unwrap());
        let err = handle.wait().unwrap_err();
        assert!(matches!(
            err,
            RuntimeError::RequestDeadlineExceeded { request_id } if request_id == "expired"
        ));
        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 1);
        assert_eq!(metrics.completed, 0);
        assert_eq!(metrics.failed, 1);
        assert_eq!(metrics.deadline_expired, 1);
        assert_eq!(metrics.cancelled, 0);
    }

    #[test]
    fn executor_rejects_zero_timeout_before_runtime_dispatch() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-timeout.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 4,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig::default(),
            },
        );
        let handle = executor
            .submit(
                RuntimeRequest::new("timeout", 99, RuntimeInputs::new())
                    .with_timeout(Duration::ZERO),
            )
            .unwrap();

        assert!(executor.execute_next().unwrap());
        let err = handle.wait().unwrap_err();
        assert!(matches!(
            err,
            RuntimeError::RequestDeadlineExceeded { request_id } if request_id == "timeout"
        ));
        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 1);
        assert_eq!(metrics.completed, 0);
        assert_eq!(metrics.failed, 1);
        assert_eq!(metrics.deadline_expired, 1);
        assert_eq!(metrics.cancelled, 0);
    }

    #[test]
    fn executor_cancels_queued_request_before_runtime_dispatch() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-cancel.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 4,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig::default(),
            },
        );
        let handle = executor
            .submit(RuntimeRequest::new("cancelled", 99, RuntimeInputs::new()))
            .unwrap();

        assert_eq!(handle.cancel(), RuntimeCancellationResult::Cancelled);
        assert_eq!(handle.cancel(), RuntimeCancellationResult::AlreadyCancelled);
        assert!(executor.execute_next().unwrap());
        let err = handle.wait().unwrap_err();
        assert!(matches!(
            err,
            RuntimeError::RequestCancelled { request_id } if request_id == "cancelled"
        ));
        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 1);
        assert_eq!(metrics.completed, 0);
        assert_eq!(metrics.failed, 1);
        assert_eq!(metrics.deadline_expired, 0);
        assert_eq!(metrics.cancelled, 1);
    }

    #[test]
    fn cancellation_after_completion_reports_completed() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-completed-cancel.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 4,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig::default(),
            },
        );
        let handle = executor.submit(add_request("done", 1.0, 10.0)).unwrap();
        let token = handle.cancellation_token();

        assert!(executor.execute_next().unwrap());
        let response = handle.wait().unwrap();
        assert_eq!(response.request_id, "done");
        assert_eq!(token.cancel(), RuntimeCancellationResult::AlreadyCompleted);

        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 1);
        assert_eq!(metrics.completed, 1);
        assert_eq!(metrics.failed, 0);
        assert_eq!(metrics.cancelled, 0);
    }

    #[test]
    fn running_cancellation_requests_ort_termination() {
        let token = RuntimeCancellationToken::new();
        assert!(token.try_mark_running().is_ok());
        let run_options = Arc::new(RunOptions::new().unwrap());
        token.attach_run_options(Arc::clone(&run_options)).unwrap();

        assert_eq!(
            token.cancel(),
            RuntimeCancellationResult::RunningCancellationRequested
        );
        assert!(token.is_cancellation_requested());
    }

    #[test]
    fn pre_requested_running_cancellation_terminates_ort_run() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-preterminated-run.rsmf"));
        let handle = engine.session_handle(0, SessionOptions::default()).unwrap();
        let token = RuntimeCancellationToken::new();
        assert!(token.try_mark_running().is_ok());
        assert_eq!(token.cancel(), RuntimeCancellationResult::AlreadyRunning);

        let err = handle
            .run_with_cancellation(
                HashMap::from([
                    (
                        "x".to_string(),
                        RuntimeTensor::F32 {
                            shape: vec![2],
                            data: vec![1.0, 2.0],
                        },
                    ),
                    (
                        "y".to_string(),
                        RuntimeTensor::F32 {
                            shape: vec![2],
                            data: vec![10.0, 20.0],
                        },
                    ),
                ]),
                Some(&token),
            )
            .unwrap_err();

        assert!(matches!(err, RuntimeError::Ort { message, .. } if message.contains("terminate")));
        token.mark_completed();
    }

    #[test]
    fn executor_preserves_runtime_errors() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-error.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 4,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig::default(),
            },
        );
        let handle = executor
            .submit(RuntimeRequest::new(
                "missing-graph",
                99,
                RuntimeInputs::new(),
            ))
            .unwrap();

        assert!(executor.execute_next().unwrap());
        let err = handle.wait().unwrap_err();
        assert!(matches!(
            err,
            RuntimeError::GraphNotFound {
                graph_idx: 99,
                graph_count: 1
            }
        ));
        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 1);
        assert_eq!(metrics.completed, 0);
        assert_eq!(metrics.failed, 1);
        assert_eq!(metrics.deadline_expired, 0);
        assert_eq!(metrics.cancelled, 0);
    }

    #[test]
    fn executor_queue_capacity_is_enforced() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-capacity.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 1,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig::default(),
            },
        );

        let _handle = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
        let err = executor
            .submit(add_request("second", 2.0, 20.0))
            .unwrap_err();
        assert!(matches!(
            err,
            RuntimeError::ExecutorQueueFull { capacity: 1 }
        ));
        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 1);
        assert_eq!(metrics.rejected_by_capacity, 1);
        assert_eq!(metrics.rejected_by_memory, 0);
        assert_eq!(metrics.current_queue_depth, 1);
        assert_eq!(metrics.max_observed_queue_depth, 1);
        assert_eq!(metrics.current_queued_tensor_bytes, 16);
        assert_eq!(metrics.max_observed_queued_tensor_bytes, 16);
    }

    #[test]
    fn executor_memory_budget_is_enforced_and_reported() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("executor-memory-budget.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 4,
                dynamic_batching: None,
                admission: RuntimeAdmissionConfig {
                    max_queued_tensor_bytes: Some(16),
                },
            },
        );

        let first = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
        let err = executor
            .submit(add_request("second", 2.0, 20.0))
            .unwrap_err();
        assert!(matches!(
            err,
            RuntimeError::ExecutorQueueBytesExceeded {
                requested_bytes: 16,
                queued_bytes: 16,
                capacity_bytes: 16,
            }
        ));
        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 1);
        assert_eq!(metrics.rejected_by_capacity, 0);
        assert_eq!(metrics.rejected_by_memory, 1);
        assert_eq!(metrics.current_queue_depth, 1);
        assert_eq!(metrics.current_queued_tensor_bytes, 16);
        assert_eq!(metrics.max_observed_queue_depth, 1);
        assert_eq!(metrics.max_observed_queued_tensor_bytes, 16);

        assert!(executor.execute_next().unwrap());
        let response = first.wait().unwrap();
        assert_eq!(f32_output(&response, "z"), vec![11.0, 11.0]);
        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.current_queue_depth, 0);
        assert_eq!(metrics.current_queued_tensor_bytes, 0);
        assert_eq!(metrics.completed, 1);
        assert_eq!(metrics.runtime_invocations, 1);
        assert_eq!(metrics.active_requests, 0);
        assert_eq!(metrics.active_runtime_invocations, 0);
        assert_eq!(metrics.active_batch_size, 0);
        assert_eq!(metrics.max_active_requests, 1);
        assert_eq!(metrics.max_active_runtime_invocations, 1);
        assert_eq!(metrics.max_active_batch_size, 1);
    }

    #[test]
    fn executor_batches_compatible_requests() {
        let dir = tempdir().unwrap();
        let engine = dynamic_add_graph_engine(dir.path().join("executor-batch.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 8,
                dynamic_batching: Some(DynamicBatchingConfig {
                    max_batch_size: 4,
                    max_queue_delay: Duration::ZERO,
                }),
                admission: RuntimeAdmissionConfig::default(),
            },
        );

        let first = executor
            .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
            .unwrap();
        let second = executor
            .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
            .unwrap();

        assert!(executor.execute_next().unwrap());
        let first_response = first.wait().unwrap();
        let second_response = second.wait().unwrap();
        assert_eq!(f32_output_shape(&first_response, "z"), vec![1, 2]);
        assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
        assert_eq!(f32_output_shape(&second_response, "z"), vec![1, 2]);
        assert_eq!(f32_output(&second_response, "z"), vec![33.0, 44.0]);

        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 2);
        assert_eq!(metrics.completed, 2);
        assert_eq!(metrics.failed, 0);
        assert_eq!(metrics.runtime_invocations, 1);
        assert_eq!(metrics.batches_executed, 1);
        assert_eq!(metrics.batched_requests, 2);
        assert_eq!(metrics.batch_fallbacks, 0);
        assert_eq!(metrics.batch_flushes_full, 0);
        assert_eq!(metrics.batch_flushes_delay, 0);
        assert_eq!(metrics.batch_flushes_memory_pressure, 0);
        assert_eq!(metrics.batch_flushes_manual, 1);
        assert_eq!(metrics.batch_flushes_shutdown, 0);
        assert_eq!(metrics.active_requests, 0);
        assert_eq!(metrics.active_runtime_invocations, 0);
        assert_eq!(metrics.active_batch_size, 0);
        assert_eq!(metrics.max_active_requests, 2);
        assert_eq!(metrics.max_active_runtime_invocations, 1);
        assert_eq!(metrics.max_active_batch_size, 2);
    }

    #[test]
    fn executor_skips_incompatible_batch_candidates() {
        let dir = tempdir().unwrap();
        let engine = dynamic_add_graph_engine(dir.path().join("executor-batch-skip.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 8,
                dynamic_batching: Some(DynamicBatchingConfig {
                    max_batch_size: 4,
                    max_queue_delay: Duration::ZERO,
                }),
                admission: RuntimeAdmissionConfig::default(),
            },
        );

        let first = executor
            .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
            .unwrap();
        let incompatible = executor
            .submit(
                dynamic_add_request("incompatible", &[3.0, 4.0], &[30.0, 40.0]).with_priority(-1),
            )
            .unwrap();

        assert!(executor.execute_next().unwrap());
        let first_response = first.wait().unwrap();
        assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
        assert!(matches!(
            incompatible.receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));

        assert!(executor.execute_next().unwrap());
        let incompatible_response = incompatible.wait().unwrap();
        assert_eq!(f32_output(&incompatible_response, "z"), vec![33.0, 44.0]);

        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.submitted, 2);
        assert_eq!(metrics.completed, 2);
        assert_eq!(metrics.runtime_invocations, 2);
        assert_eq!(metrics.batches_executed, 0);
        assert_eq!(metrics.batched_requests, 0);
        assert_eq!(metrics.batch_fallbacks, 0);
    }

    #[test]
    fn executor_reports_full_batch_flush_reason() {
        let dir = tempdir().unwrap();
        let engine = dynamic_add_graph_engine(dir.path().join("executor-full-batch.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 8,
                dynamic_batching: Some(DynamicBatchingConfig {
                    max_batch_size: 2,
                    max_queue_delay: Duration::from_secs(1),
                }),
                admission: RuntimeAdmissionConfig::default(),
            },
        );

        let first = executor
            .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
            .unwrap();
        let second = executor
            .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
            .unwrap();

        assert!(executor.execute_next().unwrap());
        assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 22.0]);
        assert_eq!(f32_output(&second.wait().unwrap(), "z"), vec![33.0, 44.0]);

        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.runtime_invocations, 1);
        assert_eq!(metrics.batches_executed, 1);
        assert_eq!(metrics.batch_flushes_full, 1);
        assert_eq!(metrics.batch_flushes_delay, 0);
        assert_eq!(metrics.batch_flushes_memory_pressure, 0);
        assert_eq!(metrics.batch_flushes_manual, 0);
        assert_eq!(metrics.batch_flushes_shutdown, 0);
    }

    #[test]
    fn background_scheduler_collects_compatible_arrivals_until_delay() {
        let dir = tempdir().unwrap();
        let engine = dynamic_add_graph_engine(dir.path().join("executor-delay-batch.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 1,
                queue_capacity: 8,
                dynamic_batching: Some(DynamicBatchingConfig {
                    max_batch_size: 4,
                    max_queue_delay: Duration::from_millis(100),
                }),
                admission: RuntimeAdmissionConfig::default(),
            },
        );

        let first = executor
            .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
            .unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert!(matches!(
            first.receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        let second = executor
            .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
            .unwrap();

        let first_response = first.wait().unwrap();
        let second_response = second.wait().unwrap();
        assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
        assert_eq!(f32_output(&second_response, "z"), vec![33.0, 44.0]);

        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.completed, 2);
        assert_eq!(metrics.runtime_invocations, 1);
        assert_eq!(metrics.batches_executed, 1);
        assert_eq!(metrics.batched_requests, 2);
        assert_eq!(metrics.batch_flushes_full, 0);
        assert_eq!(metrics.batch_flushes_delay, 1);
        assert_eq!(metrics.batch_flushes_memory_pressure, 0);
        assert_eq!(metrics.batch_flushes_manual, 0);
        assert_eq!(metrics.batch_flushes_shutdown, 0);
    }

    #[test]
    fn background_scheduler_flushes_open_batch_on_shutdown() {
        let dir = tempdir().unwrap();
        let engine = dynamic_add_graph_engine(dir.path().join("executor-shutdown-batch.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 1,
                queue_capacity: 8,
                dynamic_batching: Some(DynamicBatchingConfig {
                    max_batch_size: 4,
                    max_queue_delay: Duration::from_secs(60),
                }),
                admission: RuntimeAdmissionConfig::default(),
            },
        );

        let handle = executor
            .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
            .unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert!(matches!(
            handle.receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));

        executor.close().unwrap();
        let response = handle.wait().unwrap();
        assert_eq!(f32_output(&response, "z"), vec![11.0, 22.0]);

        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.completed, 1);
        assert_eq!(metrics.runtime_invocations, 1);
        assert_eq!(metrics.batch_flushes_full, 0);
        assert_eq!(metrics.batch_flushes_delay, 0);
        assert_eq!(metrics.batch_flushes_memory_pressure, 0);
        assert_eq!(metrics.batch_flushes_manual, 0);
        assert_eq!(metrics.batch_flushes_shutdown, 1);
    }

    #[test]
    fn executor_reports_memory_pressure_batch_flush_reason() {
        let dir = tempdir().unwrap();
        let engine = dynamic_add_graph_engine(dir.path().join("executor-pressure-batch.rsmf"));
        let executor = RuntimeExecutor::new(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 0,
                queue_capacity: 8,
                dynamic_batching: Some(DynamicBatchingConfig {
                    max_batch_size: 4,
                    max_queue_delay: Duration::from_secs(1),
                }),
                admission: RuntimeAdmissionConfig {
                    max_queued_tensor_bytes: Some(32),
                },
            },
        );

        let first = executor
            .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
            .unwrap();
        let second = executor
            .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
            .unwrap();

        assert!(executor.execute_next().unwrap());
        assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 22.0]);
        assert_eq!(f32_output(&second.wait().unwrap(), "z"), vec![33.0, 44.0]);

        let metrics = executor.metrics().unwrap();
        assert_eq!(metrics.runtime_invocations, 1);
        assert_eq!(metrics.batches_executed, 1);
        assert_eq!(metrics.batch_flushes_full, 0);
        assert_eq!(metrics.batch_flushes_delay, 0);
        assert_eq!(metrics.batch_flushes_memory_pressure, 1);
        assert_eq!(metrics.batch_flushes_manual, 0);
        assert_eq!(metrics.batch_flushes_shutdown, 0);
    }

    #[test]
    fn network_server_reports_health_and_metrics() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("network-health.rsmf"));
        let server = start_test_network_server(engine, RuntimeExecutorConfig::default());

        let (status, body) = http_json(server.local_addr(), "GET", "/health", None);
        assert_eq!(status, 200);
        assert_eq!(body["status"], "ok");

        let (status, body) = http_json(server.local_addr(), "GET", "/metrics", None);
        assert_eq!(status, 200);
        assert_eq!(body["submitted"], 0);
        assert_eq!(body["runtime_invocations"], 0);

        server.shutdown().unwrap();
    }

    #[test]
    fn network_server_runs_json_inference_request() {
        let dir = tempdir().unwrap();
        let engine = add_graph_engine(dir.path().join("network-run.rsmf"));
        let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
        let request = serde_json::json!({
            "request_id": "net-run",
            "graph_idx": 0,
            "inputs": {
                "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
                "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
            }
        });

        let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
        assert_eq!(status, 200);
        assert_eq!(body["request_id"], "net-run");
        assert_eq!(body["outputs"]["z"]["dtype"], "f32");
        assert_eq!(body["outputs"]["z"]["shape"], serde_json::json!([2]));
        assert_eq!(
            body["outputs"]["z"]["data"],
            serde_json::json!([11.0, 22.0])
        );

        let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
        assert_eq!(status, 200);
        assert_eq!(metrics["submitted"], 1);
        assert_eq!(metrics["completed"], 1);
        assert_eq!(metrics["runtime_invocations"], 1);

        server.shutdown().unwrap();
    }

    #[test]
    fn network_server_cancels_inflight_request() {
        let dir = tempdir().unwrap();
        let engine = dynamic_add_graph_engine(dir.path().join("network-cancel.rsmf"));
        let server = start_test_network_server(
            engine,
            RuntimeExecutorConfig {
                worker_threads: 1,
                queue_capacity: 8,
                dynamic_batching: Some(DynamicBatchingConfig {
                    max_batch_size: 4,
                    max_queue_delay: Duration::from_millis(100),
                }),
                admission: RuntimeAdmissionConfig::default(),
            },
        );
        let addr = server.local_addr();
        let request = serde_json::json!({
            "request_id": "net-cancel",
            "graph_idx": 0,
            "inputs": {
                "x": { "dtype": "f32", "shape": [1, 2], "data": [1.0, 2.0] },
                "y": { "dtype": "f32", "shape": [1, 2], "data": [10.0, 20.0] }
            }
        });
        let request_thread =
            std::thread::spawn(move || http_json(addr, "POST", "/v1/run", Some(&request)));
        std::thread::sleep(Duration::from_millis(20));

        let (status, body) = http_json(server.local_addr(), "GET", "/v1/requests/net-cancel", None);
        assert_eq!(status, 200);
        assert_eq!(body["status"], "inflight");

        let (status, body) = http_json(
            server.local_addr(),
            "DELETE",
            "/v1/requests/net-cancel",
            None,
        );
        assert_eq!(status, 200);
        assert_eq!(body["cancellation"], "Cancelled");

        let (status, body) = request_thread.join().unwrap();
        assert_eq!(status, 499);
        assert_eq!(body["error"]["code"], "cancelled");

        let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
        assert_eq!(status, 200);
        assert_eq!(metrics["cancelled"], 1);

        server.shutdown().unwrap();
    }

    fn start_test_network_server(
        engine: Engine,
        executor_config: RuntimeExecutorConfig,
    ) -> RuntimeNetworkServerHandle {
        RuntimeNetworkServer::new(
            RuntimeExecutor::new(engine, executor_config),
            RuntimeNetworkServerConfig::default(),
        )
        .start()
        .unwrap()
    }

    fn http_json(
        addr: SocketAddr,
        method: &str,
        path: &str,
        body: Option<&serde_json::Value>,
    ) -> (u16, serde_json::Value) {
        let body = body
            .map(serde_json::to_string)
            .transpose()
            .unwrap()
            .unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nhost: {addr}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").unwrap();
        let status = headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u16>()
            .unwrap();
        (status, serde_json::from_str(body).unwrap())
    }

    fn tiny_add_onnx_model() -> Vec<u8> {
        let mut model = Vec::new();
        push_i64(&mut model, 1, 7);
        push_string(&mut model, 2, "rsmf-runtime-test");
        push_message(&mut model, 7, tiny_add_graph());
        push_message(&mut model, 8, opset_import("", 13));
        model
    }

    fn tiny_add_external_initializer_onnx_model() -> Vec<u8> {
        tiny_add_external_initializer_onnx_model_with_dtype_shape(1, &[2])
    }

    fn tiny_add_external_initializer_onnx_model_with_dtype_shape(
        data_type: i32,
        shape: &[i64],
    ) -> Vec<u8> {
        let mut model = Vec::new();
        push_i64(&mut model, 1, 7);
        push_string(&mut model, 2, "rsmf-runtime-test");
        push_message(
            &mut model,
            7,
            tiny_add_graph_with_external_initializer(data_type, shape),
        );
        push_message(&mut model, 8, opset_import("", 13));
        model
    }

    fn tiny_add_graph() -> Vec<u8> {
        let mut graph = Vec::new();
        push_message(&mut graph, 1, add_node());
        push_string(&mut graph, 2, "rsmf_add_graph");
        push_message(&mut graph, 11, value_info("x", &[2]));
        push_message(&mut graph, 11, value_info("y", &[2]));
        push_message(&mut graph, 12, value_info("z", &[2]));
        graph
    }

    fn tiny_dynamic_add_onnx_model() -> Vec<u8> {
        let mut model = Vec::new();
        push_i64(&mut model, 1, 7);
        push_string(&mut model, 2, "rsmf-runtime-test");
        push_message(&mut model, 7, tiny_dynamic_add_graph());
        push_message(&mut model, 8, opset_import("", 13));
        model
    }

    fn tiny_dynamic_add_graph() -> Vec<u8> {
        let mut graph = Vec::new();
        push_message(&mut graph, 1, add_node());
        push_string(&mut graph, 2, "rsmf_dynamic_add_graph");
        push_message(&mut graph, 11, dynamic_value_info("x"));
        push_message(&mut graph, 11, dynamic_value_info("y"));
        push_message(&mut graph, 12, dynamic_value_info("z"));
        graph
    }

    fn tiny_add_graph_with_external_initializer(data_type: i32, shape: &[i64]) -> Vec<u8> {
        let mut graph = Vec::new();
        push_message(&mut graph, 1, add_initializer_node());
        push_string(&mut graph, 2, "rsmf_add_initializer_graph");
        push_message(&mut graph, 5, external_tensor("bias", data_type, shape));
        push_message(&mut graph, 11, value_info_typed("x", &[2], data_type));
        push_message(&mut graph, 12, value_info_typed("z", &[2], data_type));
        graph
    }

    fn add_node() -> Vec<u8> {
        let mut node = Vec::new();
        push_string(&mut node, 1, "x");
        push_string(&mut node, 1, "y");
        push_string(&mut node, 2, "z");
        push_string(&mut node, 3, "add");
        push_string(&mut node, 4, "Add");
        node
    }

    fn add_initializer_node() -> Vec<u8> {
        let mut node = Vec::new();
        push_string(&mut node, 1, "x");
        push_string(&mut node, 1, "bias");
        push_string(&mut node, 2, "z");
        push_string(&mut node, 3, "add_initializer");
        push_string(&mut node, 4, "Add");
        node
    }

    fn external_tensor(name: &str, data_type: i32, shape: &[i64]) -> Vec<u8> {
        let mut tensor = Vec::new();
        for &dim in shape {
            push_i64(&mut tensor, 1, dim);
        }
        push_i32(&mut tensor, 2, data_type);
        push_string(&mut tensor, 8, name);
        push_message(&mut tensor, 13, string_string_entry("location", "rsmf"));
        push_i32(&mut tensor, 14, 1);
        tensor
    }

    fn raw_variant(storage_dtype: StorageDtype, layout: LayoutTag, bytes: Vec<u8>) -> VariantInput {
        VariantInput {
            target: TargetTag::CpuGeneric,
            encoding: EncodingKind::Raw,
            storage_dtype: Some(storage_dtype),
            layout,
            alignment: 64,
            bytes,
            meta: VariantMeta::default(),
        }
    }

    fn string_string_entry(key: &str, value: &str) -> Vec<u8> {
        let mut entry = Vec::new();
        push_string(&mut entry, 1, key);
        push_string(&mut entry, 2, value);
        entry
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn add_graph_engine(path: std::path::PathBuf) -> Engine {
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "fixture.weight".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![1],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
                packed: Vec::new(),
            })
            .with_graph(GraphInput::onnx(tiny_add_onnx_model()))
            .write_to_path(&path)
            .unwrap();
        Engine::new(RsmfFile::open(path).unwrap()).unwrap()
    }

    fn dynamic_add_graph_engine(path: std::path::PathBuf) -> Engine {
        RsmfWriter::new()
            .with_tensor(TensorInput {
                name: "fixture.weight".to_string(),
                dtype: LogicalDtype::F32,
                shape: vec![1],
                shard_id: 0,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
                packed: Vec::new(),
            })
            .with_graph(GraphInput::onnx(tiny_dynamic_add_onnx_model()))
            .write_to_path(&path)
            .unwrap();
        Engine::new(RsmfFile::open(path).unwrap()).unwrap()
    }

    fn add_request(request_id: &str, x: f32, y: f32) -> RuntimeRequest {
        RuntimeRequest::new(
            request_id,
            0,
            HashMap::from([
                (
                    "x".to_string(),
                    RuntimeTensor::F32 {
                        shape: vec![2],
                        data: vec![x, x],
                    },
                ),
                (
                    "y".to_string(),
                    RuntimeTensor::F32 {
                        shape: vec![2],
                        data: vec![y, y],
                    },
                ),
            ]),
        )
    }

    fn dynamic_add_request(request_id: &str, x: &[f32], y: &[f32]) -> RuntimeRequest {
        RuntimeRequest::new(
            request_id,
            0,
            HashMap::from([
                (
                    "x".to_string(),
                    RuntimeTensor::F32 {
                        shape: vec![1, x.len()],
                        data: x.to_vec(),
                    },
                ),
                (
                    "y".to_string(),
                    RuntimeTensor::F32 {
                        shape: vec![1, y.len()],
                        data: y.to_vec(),
                    },
                ),
            ]),
        )
    }

    fn f32_output(response: &RuntimeResponse, name: &str) -> Vec<f32> {
        match response.outputs.get(name).unwrap() {
            RuntimeTensor::F32 { data, .. } => data.clone(),
            other => panic!("expected F32 output, got {other:?}"),
        }
    }

    fn f32_output_shape(response: &RuntimeResponse, name: &str) -> Vec<usize> {
        match response.outputs.get(name).unwrap() {
            RuntimeTensor::F32 { shape, .. } => shape.clone(),
            other => panic!("expected F32 output, got {other:?}"),
        }
    }

    fn value_info(name: &str, shape: &[i64]) -> Vec<u8> {
        value_info_typed(name, shape, 1)
    }

    fn value_info_typed(name: &str, shape: &[i64], data_type: i32) -> Vec<u8> {
        let mut value = Vec::new();
        push_string(&mut value, 1, name);
        push_message(&mut value, 2, type_proto(data_type, shape));
        value
    }

    fn dynamic_value_info(name: &str) -> Vec<u8> {
        let mut value = Vec::new();
        push_string(&mut value, 1, name);
        push_message(&mut value, 2, dynamic_type_proto());
        value
    }

    fn type_proto(data_type: i32, shape: &[i64]) -> Vec<u8> {
        let mut tensor = Vec::new();
        push_i32(&mut tensor, 1, data_type);
        push_message(&mut tensor, 2, tensor_shape(shape));

        let mut type_proto = Vec::new();
        push_message(&mut type_proto, 1, tensor);
        type_proto
    }

    fn dynamic_type_proto() -> Vec<u8> {
        let mut tensor = Vec::new();
        push_i32(&mut tensor, 1, 1);
        push_message(&mut tensor, 2, dynamic_tensor_shape());

        let mut type_proto = Vec::new();
        push_message(&mut type_proto, 1, tensor);
        type_proto
    }

    fn tensor_shape(shape: &[i64]) -> Vec<u8> {
        let mut tensor_shape = Vec::new();
        for &dim in shape {
            let mut dimension = Vec::new();
            push_i64(&mut dimension, 1, dim);
            push_message(&mut tensor_shape, 1, dimension);
        }
        tensor_shape
    }

    fn dynamic_tensor_shape() -> Vec<u8> {
        let mut tensor_shape = Vec::new();
        let mut batch = Vec::new();
        push_string(&mut batch, 2, "batch");
        push_message(&mut tensor_shape, 1, batch);

        let mut width = Vec::new();
        push_i64(&mut width, 1, 2);
        push_message(&mut tensor_shape, 1, width);
        tensor_shape
    }

    fn opset_import(domain: &str, version: i64) -> Vec<u8> {
        let mut opset = Vec::new();
        if !domain.is_empty() {
            push_string(&mut opset, 1, domain);
        }
        push_i64(&mut opset, 2, version);
        opset
    }

    fn push_i32(out: &mut Vec<u8>, field: u32, value: i32) {
        push_varint_field(out, field, value as u64);
    }

    fn push_i64(out: &mut Vec<u8>, field: u32, value: i64) {
        push_varint_field(out, field, value as u64);
    }

    fn push_string(out: &mut Vec<u8>, field: u32, value: &str) {
        push_bytes(out, field, value.as_bytes());
    }

    fn push_message(out: &mut Vec<u8>, field: u32, message: Vec<u8>) {
        push_bytes(out, field, &message);
    }

    fn push_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
        push_tag(out, field, 2);
        push_varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }

    fn push_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
        push_tag(out, field, 0);
        push_varint(out, value);
    }

    fn push_tag(out: &mut Vec<u8>, field: u32, wire_type: u64) {
        push_varint(out, ((field as u64) << 3) | wire_type);
    }

    fn push_varint(out: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }
}
