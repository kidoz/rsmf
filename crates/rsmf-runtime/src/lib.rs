//! Production-oriented ONNX Runtime integration for RSMF graph payloads.
//!
//! `rsmf-runtime` keeps RSMF's storage/container boundary intact: graph bytes
//! remain opaque ONNX / ORT payloads, while this crate owns session lifecycle,
//! typed request values, runtime options, and session caching.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, MutexGuard};

use ndarray::{ArrayD, ArrayViewD};
use ort::ep::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel as OrtGraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::{DynValue, Outlet, Tensor, TensorElementType};
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
}

impl InitializerBinding {
    /// Build an initializer binding from a graph initializer name and an RSMF
    /// tensor name.
    #[must_use]
    pub fn new(initializer_name: impl Into<String>, tensor_name: impl Into<String>) -> Self {
        Self {
            initializer_name: initializer_name.into(),
            tensor_name: tensor_name.into(),
        }
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

/// Cached ORT session handle.
#[derive(Clone, Debug)]
pub struct SessionHandle {
    key: SessionKey,
    session: Arc<Mutex<Session>>,
    inputs: Vec<ValueInfo>,
    outputs: Vec<ValueInfo>,
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

        let session = self.build_session(key.graph_idx, &key.options)?;
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

    fn build_session(&self, graph_idx: usize, options: &SessionOptions) -> Result<Session> {
        let payloads = self.file.graph_payloads();
        let payload = payloads.get(graph_idx).ok_or(RuntimeError::GraphNotFound {
            graph_idx,
            graph_count: payloads.len(),
        })?;

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

        for binding in &options.initializers {
            let initializer = self.initializer_value(binding)?;
            builder = builder
                .with_external_initializer(&binding.initializer_name, initializer)
                .map_err(|e| ort_error("register external initializer", e))?;
        }

        builder
            .commit_from_memory(payload.bytes)
            .map_err(|e| ort_error("session creation", e))
    }

    fn initializer_value(&self, binding: &InitializerBinding) -> Result<Arc<DynValue>> {
        let view = self
            .file
            .tensor_view(&binding.tensor_name)
            .map_err(|error| match error {
                RsmfError::NotFound { .. } => RuntimeError::InitializerTensorNotFound {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                },
                other => RuntimeError::Core(other),
            })?;
        validate_initializer_view(binding, &view)?;
        let shape = shape_u64_to_usize(view.shape()).map_err(|error| {
            RuntimeError::UnsupportedInitializer {
                initializer_name: binding.initializer_name.clone(),
                tensor_name: binding.tensor_name.clone(),
                reason: error,
            }
        })?;
        let data = view
            .to_vec::<f32>()
            .map_err(|error| RuntimeError::UnsupportedInitializer {
                initializer_name: binding.initializer_name.clone(),
                tensor_name: binding.tensor_name.clone(),
                reason: error.to_string(),
            })?;
        tensor_from_vec(shape, data).map(Arc::new)
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

fn validate_initializer_view(binding: &InitializerBinding, view: &TensorView<'_>) -> Result<()> {
    let reason = if view.dtype() != LogicalDtype::F32 {
        Some(format!(
            "only F32 initializers are supported, got {:?}",
            view.dtype()
        ))
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
    } else if view.storage_dtype != StorageDtype::Logical(LogicalDtype::F32) {
        Some(format!(
            "only logical F32 storage is supported, got {:?}",
            view.storage_dtype
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
    use rsmf_core::GraphInput;
    use rsmf_core::LogicalDtype;
    use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
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

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let handle = engine.session_handle(0, SessionOptions::default()).unwrap();
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
            .with_graph(GraphInput::onnx(tiny_add_external_initializer_onnx_model()))
            .write_to_path(&path)
            .unwrap();

        let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
        let options = SessionOptions {
            initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
            ..SessionOptions::default()
        };
        let handle = engine.session_handle(0, options.clone()).unwrap();
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

    fn tiny_add_onnx_model() -> Vec<u8> {
        let mut model = Vec::new();
        push_i64(&mut model, 1, 7);
        push_string(&mut model, 2, "rsmf-runtime-test");
        push_message(&mut model, 7, tiny_add_graph());
        push_message(&mut model, 8, opset_import("", 13));
        model
    }

    fn tiny_add_external_initializer_onnx_model() -> Vec<u8> {
        let mut model = Vec::new();
        push_i64(&mut model, 1, 7);
        push_string(&mut model, 2, "rsmf-runtime-test");
        push_message(&mut model, 7, tiny_add_graph_with_external_initializer());
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

    fn tiny_add_graph_with_external_initializer() -> Vec<u8> {
        let mut graph = Vec::new();
        push_message(&mut graph, 1, add_initializer_node());
        push_string(&mut graph, 2, "rsmf_add_initializer_graph");
        push_message(&mut graph, 5, external_f32_tensor("bias", &[2]));
        push_message(&mut graph, 11, value_info("x", &[2]));
        push_message(&mut graph, 12, value_info("z", &[2]));
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

    fn external_f32_tensor(name: &str, shape: &[i64]) -> Vec<u8> {
        let mut tensor = Vec::new();
        for &dim in shape {
            push_i64(&mut tensor, 1, dim);
        }
        push_i32(&mut tensor, 2, 1);
        push_string(&mut tensor, 8, name);
        push_message(&mut tensor, 13, string_string_entry("location", "rsmf"));
        push_i32(&mut tensor, 14, 1);
        tensor
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

    fn value_info(name: &str, shape: &[i64]) -> Vec<u8> {
        let mut value = Vec::new();
        push_string(&mut value, 1, name);
        push_message(&mut value, 2, type_proto_f32(shape));
        value
    }

    fn type_proto_f32(shape: &[i64]) -> Vec<u8> {
        let mut tensor = Vec::new();
        push_i32(&mut tensor, 1, 1);
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
