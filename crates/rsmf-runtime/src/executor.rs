use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::{self, Debug};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ort::session::RunOptions;
use serde::{Deserialize, Serialize};

use crate::tensor::{
    merge_runtime_tensors, request_leading_batch_size, runtime_inputs_data_bytes,
    split_runtime_outputs, tensors_are_batch_compatible,
};
use crate::{
    Engine, Result, RuntimeError, RuntimeInputs, RuntimeOutputs, SessionOptions, ort_error,
};

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
    /// Soft and hard queued-memory pressure policy.
    pub memory_pressure: RuntimeMemoryPressureConfig,
    /// Maximum number of queued requests per tenant. `None` disables this
    /// tenant quota.
    pub max_queued_requests_per_tenant: Option<usize>,
    /// Maximum bytes of owned input tensor data allowed to wait per tenant.
    /// `None` disables this tenant quota.
    pub max_queued_tensor_bytes_per_tenant: Option<usize>,
}

/// Queued-memory pressure policy for [`RuntimeExecutor`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RuntimeMemoryPressureConfig {
    /// Queued input bytes at or above this threshold are reported as soft
    /// pressure. `None` disables soft-pressure reporting.
    pub soft_queued_tensor_bytes: Option<usize>,
    /// Queued input bytes above this threshold are rejected with a typed
    /// hard-pressure error. `None` disables hard-pressure rejection.
    pub hard_queued_tensor_bytes: Option<usize>,
    /// Flush dynamic batches early when queued bytes plus the in-flight batch
    /// reach the soft-pressure threshold.
    pub flush_dynamic_batches_on_soft_pressure: bool,
}

/// Current queued-memory pressure level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMemoryPressureLevel {
    /// No configured pressure threshold is active for current queued bytes.
    #[default]
    Normal,
    /// Current queued bytes meet or exceed the configured soft threshold.
    Soft,
    /// Current queued bytes meet or exceed the configured hard threshold.
    Hard,
}

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
    /// Optional tenant identifier used for per-tenant admission quotas.
    pub tenant_id: Option<String>,
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
            tenant_id: None,
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

    /// Set tenant identifier for per-tenant admission quotas.
    #[must_use]
    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    fn effective_tenant_id(&self) -> &str {
        self.tenant_id
            .as_deref()
            .filter(|tenant_id| !tenant_id.is_empty())
            .unwrap_or(DEFAULT_TENANT_ID)
    }
}

const DEFAULT_TENANT_ID: &str = "default";

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
    /// Requests rejected because the hard memory-pressure threshold would be
    /// exceeded.
    pub rejected_by_memory_pressure: u64,
    /// Requests rejected because a tenant queue reached its configured
    /// capacity.
    pub rejected_by_tenant_capacity: u64,
    /// Requests rejected because a tenant queued tensor byte budget would be
    /// exceeded.
    pub rejected_by_tenant_memory: u64,
    /// Requests currently waiting in the queue.
    pub current_queue_depth: usize,
    /// Maximum queue depth observed by this executor.
    pub max_observed_queue_depth: usize,
    /// Owned input tensor bytes currently waiting in the queue.
    pub current_queued_tensor_bytes: usize,
    /// Maximum queued owned input tensor bytes observed by this executor.
    pub max_observed_queued_tensor_bytes: usize,
    /// Current queued-memory pressure level.
    pub memory_pressure_level: RuntimeMemoryPressureLevel,
    /// Accepted requests that left queued bytes at or above the configured soft
    /// pressure threshold.
    pub memory_pressure_soft_events: u64,
    /// Requests rejected by the configured hard memory-pressure threshold.
    pub memory_pressure_hard_rejections: u64,
    /// Dynamic scheduler flushes caused by queued-memory pressure policy.
    pub memory_pressure_flushes: u64,
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
    /// Per-tenant queued request and byte accounting.
    pub tenant_metrics: Vec<RuntimeTenantMetrics>,
}

/// Per-tenant in-process executor admission metrics.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTenantMetrics {
    /// Tenant identifier.
    pub tenant_id: String,
    /// Requests currently waiting in the queue for this tenant.
    pub current_queued_requests: usize,
    /// Maximum queued requests observed for this tenant.
    pub max_observed_queued_requests: usize,
    /// Owned input tensor bytes currently waiting in the queue for this tenant.
    pub current_queued_tensor_bytes: usize,
    /// Maximum queued owned input tensor bytes observed for this tenant.
    pub max_observed_queued_tensor_bytes: usize,
    /// Requests rejected because this tenant queue reached its configured
    /// capacity.
    pub rejected_by_capacity: u64,
    /// Requests rejected because this tenant queued tensor byte budget would be
    /// exceeded.
    pub rejected_by_memory: u64,
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
    pub(crate) fn new() -> Self {
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

    pub(crate) fn try_mark_running(&self) -> std::result::Result<(), RuntimeCancellationResult> {
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

    pub(crate) fn attach_run_options(&self, run_options: Arc<RunOptions>) -> Result<()> {
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

    pub(crate) fn clear_run_options(&self) {
        if let Ok(mut active) = self.active_run.lock() {
            *active = None;
        }
    }

    pub(crate) fn mark_completed(&self) {
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
    pub(crate) receiver: Receiver<Result<RuntimeResponse>>,
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
        let tenant_id = request.effective_tenant_id().to_string();
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
        let queued_bytes = state.metrics.current_queued_tensor_bytes;
        let would_queue = queued_bytes
            .checked_add(input_bytes)
            .ok_or_else(|| RuntimeError::Shape("queued tensor byte count overflow".to_string()))?;
        if let Some(capacity_bytes) = self.inner.admission.max_queued_tensor_bytes {
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
        if let Some(hard_limit_bytes) = self
            .inner
            .admission
            .memory_pressure
            .hard_queued_tensor_bytes
        {
            if would_queue > hard_limit_bytes {
                state.record_memory_pressure_rejection();
                return Err(RuntimeError::ExecutorMemoryPressureExceeded {
                    requested_bytes: input_bytes,
                    queued_bytes,
                    hard_limit_bytes,
                });
            }
        }
        if let Some(capacity) = self.inner.admission.max_queued_requests_per_tenant {
            let queued_requests = state.tenant_queued_requests(&tenant_id);
            if queued_requests >= capacity {
                state.record_tenant_capacity_rejection(&tenant_id);
                return Err(RuntimeError::ExecutorTenantQueueFull {
                    tenant_id,
                    capacity,
                });
            }
        }
        if let Some(capacity_bytes) = self.inner.admission.max_queued_tensor_bytes_per_tenant {
            let queued_bytes = state.tenant_queued_tensor_bytes(&tenant_id);
            let would_queue = queued_bytes.checked_add(input_bytes).ok_or_else(|| {
                RuntimeError::Shape("tenant queued tensor byte count overflow".to_string())
            })?;
            if would_queue > capacity_bytes {
                state.record_tenant_memory_rejection(&tenant_id);
                return Err(RuntimeError::ExecutorTenantQueueBytesExceeded {
                    tenant_id,
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
        if queued_soft_memory_pressure_active(would_queue, &self.inner.admission) {
            state.record_soft_memory_pressure_event();
        }
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
        self.inner
            .lock_state()
            .map(|state| state.metrics_snapshot(&self.inner.admission))
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
        self.record_tenant_queued(queued.tenant_id(), queued.input_bytes);
        self.queue.push(queued);
    }

    fn pop_queued(&mut self) -> Option<QueuedRuntimeRequest> {
        let queued = self.queue.pop()?;
        self.metrics.current_queue_depth = self.metrics.current_queue_depth.saturating_sub(1);
        self.metrics.current_queued_tensor_bytes = self
            .metrics
            .current_queued_tensor_bytes
            .saturating_sub(queued.input_bytes);
        self.record_tenant_dequeued(queued.tenant_id(), queued.input_bytes);
        Some(queued)
    }

    fn metrics_snapshot(&self, admission: &RuntimeAdmissionConfig) -> RuntimeExecutorMetrics {
        let mut metrics = self.metrics.clone();
        metrics.memory_pressure_level =
            queued_memory_pressure_level(metrics.current_queued_tensor_bytes, admission);
        metrics
            .tenant_metrics
            .sort_by(|left, right| left.tenant_id.cmp(&right.tenant_id));
        metrics
    }

    fn tenant_queued_requests(&self, tenant_id: &str) -> usize {
        self.tenant_metrics(tenant_id)
            .map_or(0, |metrics| metrics.current_queued_requests)
    }

    fn tenant_queued_tensor_bytes(&self, tenant_id: &str) -> usize {
        self.tenant_metrics(tenant_id)
            .map_or(0, |metrics| metrics.current_queued_tensor_bytes)
    }

    fn tenant_metrics(&self, tenant_id: &str) -> Option<&RuntimeTenantMetrics> {
        self.metrics
            .tenant_metrics
            .iter()
            .find(|metrics| metrics.tenant_id == tenant_id)
    }

    fn tenant_metrics_mut(&mut self, tenant_id: &str) -> &mut RuntimeTenantMetrics {
        if let Some(index) = self
            .metrics
            .tenant_metrics
            .iter()
            .position(|metrics| metrics.tenant_id == tenant_id)
        {
            &mut self.metrics.tenant_metrics[index]
        } else {
            self.metrics.tenant_metrics.push(RuntimeTenantMetrics {
                tenant_id: tenant_id.to_string(),
                ..RuntimeTenantMetrics::default()
            });
            let index = self.metrics.tenant_metrics.len().saturating_sub(1);
            &mut self.metrics.tenant_metrics[index]
        }
    }

    fn record_tenant_queued(&mut self, tenant_id: &str, input_bytes: usize) {
        let metrics = self.tenant_metrics_mut(tenant_id);
        metrics.current_queued_requests = metrics.current_queued_requests.saturating_add(1);
        metrics.max_observed_queued_requests = metrics
            .max_observed_queued_requests
            .max(metrics.current_queued_requests);
        metrics.current_queued_tensor_bytes = metrics
            .current_queued_tensor_bytes
            .saturating_add(input_bytes);
        metrics.max_observed_queued_tensor_bytes = metrics
            .max_observed_queued_tensor_bytes
            .max(metrics.current_queued_tensor_bytes);
    }

    fn record_tenant_dequeued(&mut self, tenant_id: &str, input_bytes: usize) {
        if let Some(metrics) = self
            .metrics
            .tenant_metrics
            .iter_mut()
            .find(|metrics| metrics.tenant_id == tenant_id)
        {
            metrics.current_queued_requests = metrics.current_queued_requests.saturating_sub(1);
            metrics.current_queued_tensor_bytes = metrics
                .current_queued_tensor_bytes
                .saturating_sub(input_bytes);
        }
    }

    fn record_tenant_capacity_rejection(&mut self, tenant_id: &str) {
        self.metrics.rejected_by_tenant_capacity =
            self.metrics.rejected_by_tenant_capacity.saturating_add(1);
        let metrics = self.tenant_metrics_mut(tenant_id);
        metrics.rejected_by_capacity = metrics.rejected_by_capacity.saturating_add(1);
    }

    fn record_tenant_memory_rejection(&mut self, tenant_id: &str) {
        self.metrics.rejected_by_tenant_memory =
            self.metrics.rejected_by_tenant_memory.saturating_add(1);
        let metrics = self.tenant_metrics_mut(tenant_id);
        metrics.rejected_by_memory = metrics.rejected_by_memory.saturating_add(1);
    }

    fn record_memory_pressure_rejection(&mut self) {
        self.metrics.rejected_by_memory_pressure =
            self.metrics.rejected_by_memory_pressure.saturating_add(1);
        self.metrics.memory_pressure_hard_rejections = self
            .metrics
            .memory_pressure_hard_rejections
            .saturating_add(1);
    }

    fn record_soft_memory_pressure_event(&mut self) {
        self.metrics.memory_pressure_soft_events =
            self.metrics.memory_pressure_soft_events.saturating_add(1);
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
                self.metrics.memory_pressure_flushes =
                    self.metrics.memory_pressure_flushes.saturating_add(1);
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

impl QueuedRuntimeRequest {
    fn tenant_id(&self) -> &str {
        self.request.effective_tenant_id()
    }
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
    let batch_bytes = batch.iter().fold(0usize, |total, queued| {
        total.saturating_add(queued.input_bytes)
    });
    let queued_with_batch = state
        .metrics
        .current_queued_tensor_bytes
        .saturating_add(batch_bytes);
    if admission
        .max_queued_tensor_bytes
        .is_some_and(|capacity_bytes| queued_with_batch >= capacity_bytes)
    {
        return true;
    }
    admission
        .memory_pressure
        .flush_dynamic_batches_on_soft_pressure
        && queued_soft_memory_pressure_active(queued_with_batch, admission)
}

fn queued_soft_memory_pressure_active(
    queued_bytes: usize,
    admission: &RuntimeAdmissionConfig,
) -> bool {
    admission
        .memory_pressure
        .soft_queued_tensor_bytes
        .is_some_and(|soft_limit_bytes| queued_bytes >= soft_limit_bytes)
}

fn queued_memory_pressure_level(
    queued_bytes: usize,
    admission: &RuntimeAdmissionConfig,
) -> RuntimeMemoryPressureLevel {
    if admission
        .memory_pressure
        .hard_queued_tensor_bytes
        .is_some_and(|hard_limit_bytes| queued_bytes >= hard_limit_bytes)
    {
        RuntimeMemoryPressureLevel::Hard
    } else if queued_soft_memory_pressure_active(queued_bytes, admission) {
        RuntimeMemoryPressureLevel::Soft
    } else {
        RuntimeMemoryPressureLevel::Normal
    }
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
