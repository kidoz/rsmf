use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::*;

/// Current JSON network protocol version.
pub const RUNTIME_NETWORK_PROTOCOL_VERSION: u32 = 1;

/// Configuration for the dependency-light HTTP/1.1 network serving API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeNetworkServerConfig {
    /// Address to bind. Use port `0` to request an OS-assigned port.
    pub bind_addr: SocketAddr,
    /// Maximum accepted HTTP request header size in bytes.
    pub max_header_bytes: usize,
    /// Maximum accepted request body size in bytes.
    pub max_body_bytes: usize,
    /// Maximum serialized response body size in bytes.
    pub max_response_body_bytes: usize,
    /// Per-connection read timeout.
    pub read_timeout: Duration,
    /// Per-connection write timeout.
    pub write_timeout: Duration,
    /// Optional bearer-token authentication policy.
    pub auth: RuntimeNetworkAuthConfig,
    /// Optional transport-level overload rejection policy.
    pub load_shedding: RuntimeNetworkLoadSheddingConfig,
}

impl Default for RuntimeNetworkServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            max_header_bytes: 16 * 1024,
            max_body_bytes: 16 * 1024 * 1024,
            max_response_body_bytes: 64 * 1024 * 1024,
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(10),
            auth: RuntimeNetworkAuthConfig::default(),
            load_shedding: RuntimeNetworkLoadSheddingConfig::default(),
        }
    }
}

/// Authentication policy for the dependency-light HTTP server.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct RuntimeNetworkAuthConfig {
    /// Accepted bearer tokens. Empty means authentication is disabled.
    pub bearer_tokens: Vec<String>,
}

impl RuntimeNetworkAuthConfig {
    /// Build bearer-token authentication from accepted token strings.
    #[must_use]
    pub fn bearer_tokens(tokens: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            bearer_tokens: tokens.into_iter().map(Into::into).collect(),
        }
    }

    fn is_enabled(&self) -> bool {
        !self.bearer_tokens.is_empty()
    }

    fn accepts(&self, token: &str) -> bool {
        self.bearer_tokens
            .iter()
            .any(|candidate| constant_time_str_eq(candidate, token))
    }
}

impl fmt::Debug for RuntimeNetworkAuthConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeNetworkAuthConfig")
            .field(
                "bearer_tokens",
                &format_args!("<{} configured>", self.bearer_tokens.len()),
            )
            .finish()
    }
}

/// Local load-shedding policy for the dependency-light HTTP server.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RuntimeNetworkLoadSheddingConfig {
    /// Reject `POST /v1/run` when executor queue depth is at or above this
    /// threshold. `None` disables the queue-depth transport gate.
    pub max_queue_depth: Option<usize>,
    /// Reject `POST /v1/run` when active executor request count is at or above
    /// this threshold. `None` disables the active-request transport gate.
    pub max_active_requests: Option<usize>,
    /// Reject `POST /v1/run` while queued-memory pressure is soft or hard.
    pub reject_on_memory_pressure: bool,
}

/// JSON request accepted by `POST /v1/run`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeNetworkRunRequest {
    /// Optional protocol version. Omitted means the current protocol version.
    pub protocol_version: Option<u32>,
    /// Caller-provided request identifier.
    pub request_id: String,
    /// Graph payload index.
    pub graph_idx: usize,
    /// Optional full session options. Defaults to [`SessionOptions::default`]
    /// when omitted.
    pub options: Option<SessionOptions>,
    /// Owned runtime inputs.
    pub inputs: RuntimeInputs,
    /// Optional tenant identifier used for per-tenant admission quotas.
    pub tenant_id: Option<String>,
    /// Optional request priority. Higher values run first.
    pub priority: Option<i32>,
    /// Optional timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

impl RuntimeNetworkRunRequest {
    fn into_runtime_request(self) -> Result<RuntimeRequest> {
        validate_network_protocol_version(self.protocol_version)?;
        validate_network_request_id(&self.request_id)?;
        let mut request = RuntimeRequest::new(self.request_id, self.graph_idx, self.inputs)
            .with_options(self.options.unwrap_or_default());
        if let Some(tenant_id) = self.tenant_id {
            request = request.with_tenant_id(tenant_id);
        }
        if let Some(priority) = self.priority {
            request = request.with_priority(priority);
        }
        if let Some(timeout_ms) = self.timeout_ms {
            request = request.with_timeout(Duration::from_millis(timeout_ms));
        }
        Ok(request)
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
    /// Network protocol version used for this response.
    pub protocol_version: u32,
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
            protocol_version: RUNTIME_NETWORK_PROTOCOL_VERSION,
            request_id: response.request_id,
            outputs: response.outputs,
            timings: RuntimeNetworkTimings::from_timings(&response.timings),
        }
    }
}

/// JSON metrics snapshot returned by `GET /metrics`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeNetworkMetrics {
    /// Network protocol version used for this response.
    pub protocol_version: u32,
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
    /// Owned input tensor bytes currently executing inside graph runtime calls.
    pub current_active_input_tensor_bytes: usize,
    /// Maximum active owned input tensor bytes observed by this executor.
    pub max_observed_active_input_tensor_bytes: usize,
    /// Owned output tensor bytes currently being materialized from active graph
    /// runtime calls.
    pub current_active_output_tensor_bytes: usize,
    /// Maximum active owned output tensor bytes observed by this executor.
    pub max_observed_active_output_tensor_bytes: usize,
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
    /// Dynamic scheduler flushes triggered by manual dispatch.
    pub batch_flushes_manual: u64,
    /// Dynamic scheduler flushes triggered while closing the executor.
    pub batch_flushes_shutdown: u64,
    /// Cumulative queue time, in milliseconds.
    pub total_queue_time_ms: u128,
    /// Cumulative run time, in milliseconds.
    pub total_run_time_ms: u128,
    /// Per-tenant queued request and byte accounting.
    pub tenant_metrics: Vec<RuntimeTenantMetrics>,
}

impl From<RuntimeExecutorMetrics> for RuntimeNetworkMetrics {
    fn from(metrics: RuntimeExecutorMetrics) -> Self {
        Self {
            protocol_version: RUNTIME_NETWORK_PROTOCOL_VERSION,
            submitted: metrics.submitted,
            completed: metrics.completed,
            failed: metrics.failed,
            deadline_expired: metrics.deadline_expired,
            cancelled: metrics.cancelled,
            rejected_by_capacity: metrics.rejected_by_capacity,
            rejected_by_memory: metrics.rejected_by_memory,
            rejected_by_memory_pressure: metrics.rejected_by_memory_pressure,
            rejected_by_tenant_capacity: metrics.rejected_by_tenant_capacity,
            rejected_by_tenant_memory: metrics.rejected_by_tenant_memory,
            current_queue_depth: metrics.current_queue_depth,
            max_observed_queue_depth: metrics.max_observed_queue_depth,
            current_queued_tensor_bytes: metrics.current_queued_tensor_bytes,
            max_observed_queued_tensor_bytes: metrics.max_observed_queued_tensor_bytes,
            current_active_input_tensor_bytes: metrics.current_active_input_tensor_bytes,
            max_observed_active_input_tensor_bytes: metrics.max_observed_active_input_tensor_bytes,
            current_active_output_tensor_bytes: metrics.current_active_output_tensor_bytes,
            max_observed_active_output_tensor_bytes: metrics
                .max_observed_active_output_tensor_bytes,
            memory_pressure_level: metrics.memory_pressure_level,
            memory_pressure_soft_events: metrics.memory_pressure_soft_events,
            memory_pressure_hard_rejections: metrics.memory_pressure_hard_rejections,
            memory_pressure_flushes: metrics.memory_pressure_flushes,
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
            tenant_metrics: metrics.tenant_metrics,
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
        let limits = RuntimeNetworkLimits::from(&self.config);
        let thread = std::thread::spawn(move || {
            network_accept_loop(listener, executor, state, thread_shutdown, limits);
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

#[derive(Debug, Clone)]
struct RuntimeNetworkLimits {
    max_header_bytes: usize,
    max_body_bytes: usize,
    max_response_body_bytes: usize,
    read_timeout: Duration,
    write_timeout: Duration,
    auth: RuntimeNetworkAuthConfig,
    load_shedding: RuntimeNetworkLoadSheddingConfig,
}

impl From<&RuntimeNetworkServerConfig> for RuntimeNetworkLimits {
    fn from(config: &RuntimeNetworkServerConfig) -> Self {
        Self {
            max_header_bytes: config.max_header_bytes,
            max_body_bytes: config.max_body_bytes,
            max_response_body_bytes: config.max_response_body_bytes,
            read_timeout: config.read_timeout,
            write_timeout: config.write_timeout,
            auth: config.auth.clone(),
            load_shedding: config.load_shedding.clone(),
        }
    }
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
    content_type: Option<String>,
    authorization: Option<String>,
    body: Vec<u8>,
}

fn network_accept_loop(
    listener: TcpListener,
    executor: Arc<RuntimeExecutor>,
    state: Arc<RuntimeNetworkServerState>,
    shutdown: Arc<AtomicBool>,
    limits: RuntimeNetworkLimits,
) {
    while !shutdown.load(AtomicOrdering::Acquire) {
        match listener.accept() {
            Ok((stream, _)) => {
                let executor = Arc::clone(&executor);
                let state = Arc::clone(&state);
                let limits = limits.clone();
                std::thread::spawn(move || {
                    handle_network_connection(stream, executor, state, limits);
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
    limits: RuntimeNetworkLimits,
) {
    let response = read_http_request(&mut stream, limits.clone())
        .and_then(|request| route_network_request(request, &executor, &state, limits.clone()));
    match response {
        Ok(body) => {
            if let Err(error) = write_json_response(&mut stream, 200, "OK", &body, limits.clone()) {
                let _ = write_network_error_response(&mut stream, &error, limits);
            }
        }
        Err(error) => {
            let _ = write_network_error_response(&mut stream, &error, limits);
        }
    }
}

fn route_network_request(
    request: RuntimeHttpRequest,
    executor: &RuntimeExecutor,
    state: &RuntimeNetworkServerState,
    limits: RuntimeNetworkLimits,
) -> Result<serde_json::Value> {
    validate_network_auth(&request, &limits.auth)?;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => Ok(serde_json::json!({
            "status": "ok",
            "protocol_version": RUNTIME_NETWORK_PROTOCOL_VERSION
        })),
        ("GET", "/metrics") => {
            let metrics = RuntimeNetworkMetrics::from(executor.metrics()?);
            serde_json::to_value(metrics).map_err(network_json_error)
        }
        ("POST", "/v1/run") => {
            validate_json_content_type(request.content_type.as_deref())?;
            enforce_network_load_shedding(executor, &limits.load_shedding)?;
            handle_network_run(request.body, executor, state)
        }
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

fn validate_network_auth(
    request: &RuntimeHttpRequest,
    config: &RuntimeNetworkAuthConfig,
) -> Result<()> {
    if !config.is_enabled() || request.path == "/health" {
        return Ok(());
    }
    let Some(header) = request.authorization.as_deref() else {
        return Err(RuntimeError::NetworkUnauthorized);
    };
    let Some(token) = header.strip_prefix("Bearer ") else {
        return Err(RuntimeError::NetworkUnauthorized);
    };
    if config.accepts(token.trim()) {
        Ok(())
    } else {
        Err(RuntimeError::NetworkUnauthorized)
    }
}

fn enforce_network_load_shedding(
    executor: &RuntimeExecutor,
    config: &RuntimeNetworkLoadSheddingConfig,
) -> Result<()> {
    if config.max_queue_depth.is_none()
        && config.max_active_requests.is_none()
        && !config.reject_on_memory_pressure
    {
        return Ok(());
    }
    let metrics = executor.metrics()?;
    if let Some(limit) = config.max_queue_depth
        && metrics.current_queue_depth >= limit
    {
        return Err(RuntimeError::NetworkOverloaded {
            reason: format!(
                "queue depth {} is at or above configured limit {limit}",
                metrics.current_queue_depth
            ),
        });
    }
    if let Some(limit) = config.max_active_requests
        && metrics.active_requests >= limit
    {
        return Err(RuntimeError::NetworkOverloaded {
            reason: format!(
                "active requests {} is at or above configured limit {limit}",
                metrics.active_requests
            ),
        });
    }
    if config.reject_on_memory_pressure
        && metrics.memory_pressure_level != RuntimeMemoryPressureLevel::Normal
    {
        return Err(RuntimeError::NetworkOverloaded {
            reason: format!(
                "queued-memory pressure is {:?}",
                metrics.memory_pressure_level
            ),
        });
    }
    Ok(())
}

fn handle_network_run(
    body: Vec<u8>,
    executor: &RuntimeExecutor,
    state: &RuntimeNetworkServerState,
) -> Result<serde_json::Value> {
    let request: RuntimeNetworkRunRequest =
        serde_json::from_slice(&body).map_err(network_json_error)?;
    let request_id = request.request_id.clone();
    validate_network_request_id(&request_id)?;
    state.reserve(&request_id)?;
    let runtime_request = match request.into_runtime_request() {
        Ok(request) => request,
        Err(error) => {
            state.remove(&request_id);
            return Err(error);
        }
    };
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

fn validate_network_request_id(request_id: &str) -> Result<()> {
    if request_id.is_empty() || request_id.contains('/') {
        return Err(RuntimeError::NetworkProtocol {
            reason: "request_id must be non-empty and must not contain '/'".to_string(),
        });
    }
    Ok(())
}

fn validate_network_protocol_version(protocol_version: Option<u32>) -> Result<()> {
    let Some(version) = protocol_version else {
        return Ok(());
    };
    if version != RUNTIME_NETWORK_PROTOCOL_VERSION {
        return Err(RuntimeError::NetworkProtocolVersion {
            version,
            supported_version: RUNTIME_NETWORK_PROTOCOL_VERSION,
        });
    }
    Ok(())
}

fn validate_json_content_type(content_type: Option<&str>) -> Result<()> {
    let Some(content_type) = content_type else {
        return Ok(());
    };
    let media_type = content_type
        .split(';')
        .next()
        .map(str::trim)
        .unwrap_or_default();
    if media_type.eq_ignore_ascii_case("application/json") {
        Ok(())
    } else {
        Err(RuntimeError::NetworkProtocol {
            reason: "content-type must be application/json".to_string(),
        })
    }
}

fn read_http_request(
    stream: &mut TcpStream,
    limits: RuntimeNetworkLimits,
) -> Result<RuntimeHttpRequest> {
    stream
        .set_read_timeout(Some(limits.read_timeout))
        .map_err(|e| network_io_error("configure connection read timeout", e))?;
    let mut buffer = Vec::new();
    let header_end = loop {
        if let Some(header_end) = find_header_end(&buffer) {
            break header_end;
        }
        if buffer.len() > limits.max_header_bytes {
            return Err(RuntimeError::NetworkRequestHeadersTooLarge {
                limit_bytes: limits.max_header_bytes,
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
    if _version != "HTTP/1.1" {
        return Err(RuntimeError::NetworkProtocol {
            reason: "HTTP version must be HTTP/1.1".to_string(),
        });
    }
    if request_parts.next().is_some() {
        return Err(RuntimeError::NetworkProtocol {
            reason: "malformed HTTP request line".to_string(),
        });
    }

    let mut content_length = 0usize;
    let mut content_type = None;
    let mut authorization = None;
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
        } else if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("content-type")
        {
            content_type = Some(value.trim().to_string());
        } else if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("authorization")
        {
            authorization = Some(value.trim().to_string());
        } else if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("transfer-encoding")
            && !value.trim().eq_ignore_ascii_case("identity")
        {
            return Err(RuntimeError::NetworkProtocol {
                reason: "transfer-encoding is not supported".to_string(),
            });
        }
    }
    if content_length > limits.max_body_bytes {
        return Err(RuntimeError::NetworkRequestBodyTooLarge {
            requested_bytes: content_length,
            limit_bytes: limits.max_body_bytes,
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
        content_type,
        authorization,
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
    limits: RuntimeNetworkLimits,
) -> Result<()> {
    let body = serde_json::to_vec(body).map_err(network_json_error)?;
    if body.len() > limits.max_response_body_bytes {
        return Err(RuntimeError::NetworkResponseBodyTooLarge {
            response_bytes: body.len(),
            limit_bytes: limits.max_response_body_bytes,
        });
    }
    stream
        .set_write_timeout(Some(limits.write_timeout))
        .map_err(|e| network_io_error("configure connection write timeout", e))?;
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|()| stream.write_all(&body))
        .map_err(|e| network_io_error("write network response", e))
}

fn write_network_error_response(
    stream: &mut TcpStream,
    error: &RuntimeError,
    limits: RuntimeNetworkLimits,
) -> Result<()> {
    let (status, reason, code) = network_error_status(error);
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": network_public_error_message(error)
        }
    });
    write_json_response(stream, status, reason, &body, limits)
}

fn network_error_status(error: &RuntimeError) -> (u16, &'static str, &'static str) {
    match error {
        RuntimeError::NetworkRequestNotFound { .. } => (404, "Not Found", "not_found"),
        RuntimeError::NetworkRequestAlreadyActive { .. } => (409, "Conflict", "conflict"),
        RuntimeError::NetworkUnauthorized => (401, "Unauthorized", "unauthorized"),
        RuntimeError::NetworkOverloaded { .. } => (503, "Service Unavailable", "overloaded"),
        RuntimeError::NetworkRequestBodyTooLarge { .. } => {
            (413, "Payload Too Large", "payload_too_large")
        }
        RuntimeError::NetworkRequestHeadersTooLarge { .. } => {
            (431, "Request Header Fields Too Large", "headers_too_large")
        }
        RuntimeError::ExecutorQueueFull { .. }
        | RuntimeError::ExecutorQueueBytesExceeded { .. }
        | RuntimeError::ExecutorMemoryPressureExceeded { .. }
        | RuntimeError::ExecutorTenantQueueFull { .. }
        | RuntimeError::ExecutorTenantQueueBytesExceeded { .. } => {
            (429, "Too Many Requests", "admission_rejected")
        }
        RuntimeError::NetworkProtocolVersion { .. } => {
            (400, "Bad Request", "unsupported_protocol_version")
        }
        RuntimeError::NetworkProtocol { .. } => (400, "Bad Request", "bad_request"),
        RuntimeError::RequestDeadlineExceeded { .. } => {
            (408, "Request Timeout", "deadline_expired")
        }
        RuntimeError::NetworkTimeout { .. } => (408, "Request Timeout", "network_timeout"),
        RuntimeError::RequestCancelled { .. } => (499, "Client Closed Request", "cancelled"),
        RuntimeError::NetworkResponseBodyTooLarge { .. } => {
            (500, "Internal Server Error", "response_too_large")
        }
        _ => (500, "Internal Server Error", "runtime_error"),
    }
}

fn network_io_error(operation: &'static str, error: std::io::Error) -> RuntimeError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    ) {
        return RuntimeError::NetworkTimeout { operation };
    }
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

fn network_public_error_message(error: &RuntimeError) -> String {
    match error {
        RuntimeError::NetworkProtocol { reason } => reason.clone(),
        RuntimeError::NetworkProtocolVersion {
            version,
            supported_version,
        } => format!(
            "unsupported protocol version {version}; supported version is {supported_version}"
        ),
        RuntimeError::NetworkRequestBodyTooLarge {
            requested_bytes,
            limit_bytes,
        } => {
            format!("request body is {requested_bytes} bytes, limit is {limit_bytes}")
        }
        RuntimeError::NetworkRequestHeadersTooLarge { limit_bytes } => {
            format!("request headers exceed {limit_bytes} bytes")
        }
        RuntimeError::NetworkRequestNotFound { .. } => "request not found".to_string(),
        RuntimeError::NetworkRequestAlreadyActive { .. } => {
            "request id is already active".to_string()
        }
        RuntimeError::NetworkUnauthorized => "request is unauthorized".to_string(),
        RuntimeError::NetworkOverloaded { .. } => "request rejected by load shedding".to_string(),
        RuntimeError::RequestDeadlineExceeded { .. } => "request deadline expired".to_string(),
        RuntimeError::RequestCancelled { .. } => "request was cancelled".to_string(),
        RuntimeError::ExecutorQueueFull { .. }
        | RuntimeError::ExecutorQueueBytesExceeded { .. }
        | RuntimeError::ExecutorMemoryPressureExceeded { .. }
        | RuntimeError::ExecutorTenantQueueFull { .. }
        | RuntimeError::ExecutorTenantQueueBytesExceeded { .. } => {
            "request rejected by admission control".to_string()
        }
        RuntimeError::NetworkTimeout { .. } => "network operation timed out".to_string(),
        RuntimeError::NetworkResponseBodyTooLarge { .. } => {
            "response exceeded configured size limit".to_string()
        }
        RuntimeError::NetworkIo { .. } => "network I/O error".to_string(),
        _ => "runtime request failed".to_string(),
    }
}

fn constant_time_str_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut diff = 0u8;
    for (left, right) in left.iter().zip(right) {
        diff |= left ^ right;
    }
    diff == 0
}
