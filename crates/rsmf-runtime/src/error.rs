use rsmf_core::RsmfError;

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
    /// A required native decoder asset is not present.
    #[error("native decoder requires asset {asset_name}")]
    NativeDecoderAssetMissing {
        /// Required asset name.
        asset_name: String,
    },
    /// A native decoder config asset is malformed or unsupported.
    #[error("invalid native decoder config: {reason}")]
    NativeDecoderConfigInvalid {
        /// Human-readable validation failure.
        reason: String,
    },
    /// A native decoder family is not supported by this runtime.
    #[error("unsupported native decoder family {family}")]
    UnsupportedNativeDecoder {
        /// Family or model type from config metadata.
        family: String,
    },
    /// A required native decoder tensor is not present.
    #[error("native decoder requires tensor {tensor_name}")]
    NativeDecoderTensorMissing {
        /// Required tensor name.
        tensor_name: String,
    },
    /// A required native decoder tensor has the wrong shape.
    #[error(
        "native decoder tensor {tensor_name} has shape {actual_shape}, expected {expected_shape}"
    )]
    NativeDecoderTensorShapeMismatch {
        /// Tensor name.
        tensor_name: String,
        /// Expected shape string.
        expected_shape: String,
        /// Actual shape string.
        actual_shape: String,
    },
    /// A required native decoder tensor has an unsupported logical dtype.
    #[error("native decoder tensor {tensor_name} has unsupported dtype {dtype}")]
    NativeDecoderTensorDtypeUnsupported {
        /// Tensor name.
        tensor_name: String,
        /// Human-readable dtype.
        dtype: String,
    },
    /// A required native decoder tensor uses an unsupported storage layout,
    /// encoding, or byte representation for the selected runtime path.
    #[error("native decoder tensor {tensor_name} is unsupported: {reason}")]
    NativeDecoderTensorUnsupported {
        /// Tensor name.
        tensor_name: String,
        /// Human-readable reason.
        reason: String,
    },
    /// A requested native decoder backend is not available in this build.
    #[error("native decoder backend {backend} is unavailable: {reason}")]
    NativeDecoderBackendUnavailable {
        /// Requested backend name.
        backend: String,
        /// Human-readable reason.
        reason: String,
    },
    /// Native decoder generation requires at least one prompt token.
    #[error("native decoder generation requires at least one prompt token")]
    NativeDecoderPromptEmpty,
    /// A native decoder token id is outside the configured vocabulary.
    #[error("native decoder token id {token_id} is outside vocabulary size {vocab_size}")]
    NativeDecoderTokenOutOfRange {
        /// Invalid token id.
        token_id: i64,
        /// Configured vocabulary size.
        vocab_size: usize,
    },
    /// Native decoder sampling options are invalid.
    #[error("invalid native decoder sampling options: {reason}")]
    NativeDecoderSamplingInvalid {
        /// Human-readable validation failure.
        reason: String,
    },
    /// Native decoder reference logits do not match within tolerance.
    #[error(
        "native decoder reference logits mismatch: max abs diff {max_abs_diff} exceeds tolerance {tolerance_abs}"
    )]
    NativeDecoderReferenceLogitsMismatch {
        /// Largest absolute difference observed.
        max_abs_diff: f32,
        /// Accepted absolute tolerance.
        tolerance_abs: f32,
    },
    /// A native decoder tokenizer asset is malformed or unsupported.
    #[error("invalid native decoder tokenizer: {reason}")]
    NativeDecoderTokenizerInvalid {
        /// Human-readable validation failure.
        reason: String,
    },
    /// A native decoder tokenizer cannot encode or decode a token.
    #[error("native decoder tokenizer cannot resolve token {token}")]
    NativeDecoderTokenizerTokenUnknown {
        /// Token text or id string.
        token: String,
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
    /// The runtime executor hard memory-pressure threshold would be exceeded.
    #[error(
        "runtime executor hard memory-pressure threshold exceeded; requested {requested_bytes} bytes with {queued_bytes}/{hard_limit_bytes} bytes queued"
    )]
    ExecutorMemoryPressureExceeded {
        /// Bytes in the rejected request's owned inputs.
        requested_bytes: usize,
        /// Bytes already queued before the rejected request.
        queued_bytes: usize,
        /// Maximum configured hard queued-memory pressure threshold.
        hard_limit_bytes: usize,
    },
    /// The runtime executor tenant queue has reached its configured capacity.
    #[error("runtime executor tenant {tenant_id} queue is full; capacity is {capacity}")]
    ExecutorTenantQueueFull {
        /// Tenant identifier.
        tenant_id: String,
        /// Maximum number of queued requests for this tenant.
        capacity: usize,
    },
    /// The runtime executor tenant queued tensor byte budget would be exceeded.
    #[error(
        "runtime executor tenant {tenant_id} queued tensor byte budget exceeded; requested {requested_bytes} bytes with {queued_bytes}/{capacity_bytes} bytes queued"
    )]
    ExecutorTenantQueueBytesExceeded {
        /// Tenant identifier.
        tenant_id: String,
        /// Bytes in the rejected request's owned inputs.
        requested_bytes: usize,
        /// Bytes already queued for this tenant before the rejected request.
        queued_bytes: usize,
        /// Maximum configured queued input bytes for this tenant.
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
    /// A network request used an unsupported protocol version.
    #[error("unsupported runtime network protocol version {version}; expected {supported_version}")]
    NetworkProtocolVersion {
        /// Requested protocol version.
        version: u32,
        /// Supported protocol version.
        supported_version: u32,
    },
    /// A network request header block exceeded the configured limit.
    #[error("runtime network request headers exceed {limit_bytes} bytes")]
    NetworkRequestHeadersTooLarge {
        /// Maximum configured header bytes.
        limit_bytes: usize,
    },
    /// A network request body exceeded the configured limit.
    #[error("runtime network request body is {requested_bytes} bytes, limit is {limit_bytes}")]
    NetworkRequestBodyTooLarge {
        /// Request body bytes declared by content-length.
        requested_bytes: usize,
        /// Maximum configured body bytes.
        limit_bytes: usize,
    },
    /// A network response body exceeded the configured limit.
    #[error("runtime network response body is {response_bytes} bytes, limit is {limit_bytes}")]
    NetworkResponseBodyTooLarge {
        /// Serialized response body bytes.
        response_bytes: usize,
        /// Maximum configured response bytes.
        limit_bytes: usize,
    },
    /// A network operation timed out.
    #[error("runtime network operation timed out during {operation}")]
    NetworkTimeout {
        /// Operation that timed out.
        operation: &'static str,
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
