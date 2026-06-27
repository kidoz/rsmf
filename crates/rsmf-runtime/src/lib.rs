//! Production-oriented ONNX Runtime integration for RSMF graph payloads.
//!
//! `rsmf-runtime` keeps RSMF's storage/container boundary intact: graph bytes
//! remain opaque ONNX / ORT payloads, while this crate owns session lifecycle,
//! typed request values, runtime options, and session caching.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ndarray::{ArrayD, ArrayViewD};
use ort::ep::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel as OrtGraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::{DynValue, Outlet, Tensor, TensorElementType};
use rsmf_core::manifest::GraphKind;
use rsmf_core::tensor::variant::{EncodingKind, LayoutTag};
use rsmf_core::{LogicalDtype, RsmfError, RsmfFile, StorageDtype, TensorView};

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
}

/// ONNX Runtime graph optimization level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq)]
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
}

impl Default for RuntimeExecutorConfig {
    fn default() -> Self {
        Self {
            worker_threads: 1,
            queue_capacity: 1024,
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
    /// Cumulative queue time for completed and failed dispatched requests.
    pub total_queue_time: Duration,
    /// Cumulative run time for completed and failed dispatched requests.
    pub total_run_time: Duration,
}

/// Completion handle returned by [`RuntimeExecutor::submit`].
#[derive(Debug)]
pub struct RuntimeRequestHandle {
    request_id: String,
    receiver: Receiver<Result<RuntimeResponse>>,
}

impl RuntimeRequestHandle {
    /// Caller-provided request identifier for this handle.
    #[must_use]
    pub fn request_id(&self) -> &str {
        &self.request_id
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
        let ort_inputs = inputs
            .into_iter()
            .map(|(name, tensor)| Ok((name, tensor.into_ort_value()?)))
            .collect::<Result<Vec<_>>>()?;
        let mut session = self.lock_session()?;
        let outputs = session
            .run(ort_inputs)
            .map_err(|e| ort_error("session run", e))?;
        materialize_outputs(outputs)
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
        let mut state = self.inner.lock_state()?;
        if state.closed {
            return Err(RuntimeError::ExecutorClosed);
        }
        if state.queue.len() >= self.inner.queue_capacity {
            return Err(RuntimeError::ExecutorQueueFull {
                capacity: self.inner.queue_capacity,
            });
        }

        let sequence = state.next_sequence;
        state.next_sequence = state
            .next_sequence
            .checked_add(1)
            .ok_or(RuntimeError::ExecutorSequenceOverflow)?;
        let priority = request.priority;
        state.queue.push(QueuedRuntimeRequest {
            priority,
            sequence,
            queued_at: Instant::now(),
            request,
            sender,
        });
        state.metrics.submitted = state.metrics.submitted.saturating_add(1);
        drop(state);
        self.inner.available.notify_one();

        Ok(RuntimeRequestHandle {
            request_id,
            receiver,
        })
    }

    /// Execute the next queued request on the caller's thread.
    ///
    /// Returns `Ok(true)` when a request was executed and `Ok(false)` when no
    /// request is currently queued.
    pub fn execute_next(&self) -> Result<bool> {
        let Some(queued) = self.inner.pop_ready()? else {
            return Ok(false);
        };
        execute_queued_request(&self.inner, queued);
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

struct RuntimeExecutorInner {
    engine: Arc<Engine>,
    state: Mutex<RuntimeExecutorState>,
    available: Condvar,
    queue_capacity: usize,
}

impl RuntimeExecutorInner {
    fn lock_state(&self) -> Result<MutexGuard<'_, RuntimeExecutorState>> {
        self.state
            .lock()
            .map_err(|_| RuntimeError::ExecutorPoisoned)
    }

    fn pop_ready(&self) -> Result<Option<QueuedRuntimeRequest>> {
        self.lock_state().map(|mut state| state.queue.pop())
    }

    fn pop_blocking(&self) -> Option<QueuedRuntimeRequest> {
        let mut state = self.state.lock().ok()?;
        loop {
            if let Some(queued) = state.queue.pop() {
                return Some(queued);
            }
            if state.closed {
                return None;
            }
            state = self.available.wait(state).ok()?;
        }
    }

    fn record_completed(&self, queue_time: Duration, run_time: Duration) {
        if let Ok(mut state) = self.state.lock() {
            state.metrics.completed = state.metrics.completed.saturating_add(1);
            state.metrics.total_queue_time += queue_time;
            state.metrics.total_run_time += run_time;
        }
    }

    fn record_failed(&self, queue_time: Duration, run_time: Duration, deadline_expired: bool) {
        if let Ok(mut state) = self.state.lock() {
            state.metrics.failed = state.metrics.failed.saturating_add(1);
            if deadline_expired {
                state.metrics.deadline_expired = state.metrics.deadline_expired.saturating_add(1);
            }
            state.metrics.total_queue_time += queue_time;
            state.metrics.total_run_time += run_time;
        }
    }
}

struct RuntimeExecutorState {
    queue: BinaryHeap<QueuedRuntimeRequest>,
    closed: bool,
    next_sequence: u64,
    metrics: RuntimeExecutorMetrics,
}

struct QueuedRuntimeRequest {
    priority: i32,
    sequence: u64,
    queued_at: Instant,
    request: RuntimeRequest,
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
    while let Some(queued) = inner.pop_blocking() {
        execute_queued_request(&inner, queued);
    }
}

fn execute_queued_request(inner: &RuntimeExecutorInner, queued: QueuedRuntimeRequest) {
    let started_at = Instant::now();
    let queue_time = started_at.saturating_duration_since(queued.queued_at);
    if queued
        .request
        .deadline
        .is_some_and(|deadline| deadline <= started_at)
    {
        inner.record_failed(queue_time, Duration::ZERO, true);
        let _ = queued
            .sender
            .send(Err(RuntimeError::RequestDeadlineExceeded {
                request_id: queued.request.request_id,
            }));
        return;
    }

    let request_id = queued.request.request_id;
    let result = inner.engine.run(
        queued.request.graph_idx,
        queued.request.options,
        queued.request.inputs,
    );
    let finished_at = Instant::now();
    let run_time = finished_at.saturating_duration_since(started_at);
    match result {
        Ok(outputs) => {
            inner.record_completed(queue_time, run_time);
            let _ = queued.sender.send(Ok(RuntimeResponse {
                request_id,
                outputs,
                timings: RuntimeRequestTimings {
                    queued_at: queued.queued_at,
                    started_at,
                    finished_at,
                    queue_time,
                    run_time,
                },
            }));
        }
        Err(error) => {
            inner.record_failed(queue_time, run_time, false);
            let _ = queued.sender.send(Err(error));
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

    fn f32_output(response: &RuntimeResponse, name: &str) -> Vec<f32> {
        match response.outputs.get(name).unwrap() {
            RuntimeTensor::F32 { data, .. } => data.clone(),
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

    fn type_proto(data_type: i32, shape: &[i64]) -> Vec<u8> {
        let mut tensor = Vec::new();
        push_i32(&mut tensor, 1, data_type);
        push_message(&mut tensor, 2, tensor_shape(shape));

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
