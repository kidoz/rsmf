use ort::session::builder::GraphOptimizationLevel as OrtGraphOptimizationLevel;
use ort::value::Outlet;
use serde::{Deserialize, Serialize};

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
    pub(crate) fn from_outlet(outlet: &Outlet) -> Self {
        Self {
            name: outlet.name().to_string(),
            value_type: outlet.dtype().to_string(),
        }
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
