use ort::session::builder::GraphOptimizationLevel as OrtGraphOptimizationLevel;
use ort::value::Outlet;
use serde::{Deserialize, Serialize};

/// One raw ORT allocator statistic key/value entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAllocatorStat {
    /// ORT statistic key.
    pub key: String,
    /// ORT statistic value as returned by the provider.
    pub value: String,
}

/// ORT allocator statistics with parsed common counters.
///
/// ORT providers may return an empty key/value map for allocators that do not
/// expose internal statistics. Raw entries are preserved so callers can inspect
/// provider-specific counters without the runtime guessing their semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RuntimeAllocatorStats {
    /// ORT returned allocator statistics.
    Available {
        /// Total bytes allocated over the allocator lifetime, when ORT reports
        /// `TotalAllocated`.
        total_allocated_bytes: Option<usize>,
        /// Maximum bytes in use at one time, when ORT reports `MaxInUse`.
        max_in_use_bytes: Option<usize>,
        /// Largest single allocation size, when ORT reports `MaxAllocSize`.
        max_alloc_size_bytes: Option<usize>,
        /// Number of allocation calls, when ORT reports `NumAllocs`.
        allocation_count: Option<usize>,
        /// Number of reserve calls, when ORT reports `NumReserves`.
        reserve_count: Option<usize>,
        /// Raw provider key/value entries copied from ORT.
        raw_entries: Vec<RuntimeAllocatorStat>,
    },
    /// The runtime could not obtain allocator statistics for this session.
    Unavailable {
        /// Human-readable reason.
        reason: String,
    },
}

impl RuntimeAllocatorStats {
    /// Build an explicit unavailable allocator-stat report.
    #[must_use]
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    /// Build allocator stats from raw ORT key/value entries.
    #[must_use]
    pub fn from_entries(raw_entries: Vec<RuntimeAllocatorStat>) -> Self {
        if raw_entries.is_empty() {
            return Self::unavailable("ORT allocator returned no stats entries");
        }
        let parsed = |name: &str| -> Option<usize> {
            raw_entries
                .iter()
                .find(|entry| entry.key == name)
                .and_then(|entry| entry.value.parse::<usize>().ok())
        };
        Self::Available {
            total_allocated_bytes: parsed("TotalAllocated"),
            max_in_use_bytes: parsed("MaxInUse"),
            max_alloc_size_bytes: parsed("MaxAllocSize"),
            allocation_count: parsed("NumAllocs"),
            reserve_count: parsed("NumReserves"),
            raw_entries,
        }
    }

    /// Return ORT's `MaxInUse` byte counter when present.
    #[must_use]
    pub fn max_in_use_bytes(&self) -> Option<usize> {
        match self {
            Self::Available {
                max_in_use_bytes, ..
            } => *max_in_use_bytes,
            Self::Unavailable { .. } => None,
        }
    }
}

/// Byte-valued runtime memory measurement.
///
/// Some memory sources, such as ORT provider allocator statistics, depend on
/// backend-specific runtime counters. Those fields are reported as explicitly
/// unavailable instead of inferred when the active backend returns no counter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RuntimeMemoryMeasurement {
    /// The runtime measured this memory source.
    Available {
        /// Measured byte count.
        bytes: usize,
    },
    /// The runtime cannot measure this memory source with the current backend
    /// and build.
    Unavailable {
        /// Human-readable reason.
        reason: String,
    },
}

impl RuntimeMemoryMeasurement {
    /// Build an available byte measurement.
    #[must_use]
    pub fn available(bytes: usize) -> Self {
        Self::Available { bytes }
    }

    /// Build an explicit unavailable measurement.
    #[must_use]
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    /// Return measured bytes when available.
    #[must_use]
    pub fn bytes(&self) -> Option<usize> {
        match self {
            Self::Available { bytes } => Some(*bytes),
            Self::Unavailable { .. } => None,
        }
    }
}

/// Runtime feature or backend capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RuntimeCapability {
    /// The capability is available.
    Available,
    /// The capability is not available in this runtime build or backend.
    Unavailable {
        /// Human-readable reason.
        reason: String,
    },
}

impl RuntimeCapability {
    /// Build an unavailable capability report.
    #[must_use]
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    /// Return whether the capability is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

/// Runtime capability report for residency-sensitive execution features.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCapabilityReport {
    /// Safe ORT CPU I/O binding support for owned runtime tensors.
    pub ort_cpu_io_binding: RuntimeCapability,
    /// ORT provider allocator memory statistics.
    pub ort_provider_allocator_stats: RuntimeCapability,
    /// True borrowed mmap-backed ONNX initializer binding.
    pub mmap_initializer_zero_copy: RuntimeCapability,
    /// Direct native decoder quantized matrix-vector kernels for RawI8, Q8_0,
    /// and Q4_0 projection weights.
    pub native_decoder_i8_q8_q4_direct_kernels: RuntimeCapability,
    /// Direct native decoder QK-family kernels such as Q3_K, Q4_K, Q5_K, and
    /// Q6_K.
    pub native_decoder_qk_family_direct_kernels: RuntimeCapability,
    /// Native continuous batching can fuse compatible LM-head projection rows.
    pub native_decoder_fused_lm_head_continuous_batching: RuntimeCapability,
    /// Native continuous batching fused Q/K/V, attention, and MLP kernels.
    pub native_decoder_fused_qkv_attention_mlp: RuntimeCapability,
    /// Native decoder Metal/WGPU execution path.
    pub native_decoder_metal_wgpu: RuntimeCapability,
    /// ORT CoreML execution provider path for embedded graph payloads.
    pub ort_coreml_execution_provider: RuntimeCapability,
    /// Direct SentencePiece `.model` protobuf tokenizer loading.
    pub sentencepiece_model_protobuf: RuntimeCapability,
    /// Std HTTP bearer-token authentication.
    pub serving_bearer_auth: RuntimeCapability,
    /// Std HTTP local overload/load-shedding policy.
    pub serving_load_shedding: RuntimeCapability,
    /// Native TLS transport in this crate.
    pub serving_tls: RuntimeCapability,
    /// Response streaming transport in this crate.
    pub serving_streaming: RuntimeCapability,
    /// Distributed tenant quota coordination.
    pub distributed_quotas: RuntimeCapability,
    /// Production multi-device/expert runtime in `rsmf-runtime`.
    pub r6_multi_device_expert_runtime: RuntimeCapability,
}

/// Runtime I/O binding policy for ORT graph execution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IoBindingPolicy {
    /// Use the standard ORT `Session::run` API.
    #[default]
    Disabled,
    /// Bind owned CPU tensors through ORT's safe `IoBinding` API.
    Cpu,
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
    /// Optional ORT I/O binding policy.
    pub io_binding: IoBindingPolicy,
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
            io_binding: IoBindingPolicy::Disabled,
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
    /// Bytes in the selected RSMF tensor variant used as the initializer source.
    pub source_bytes: usize,
    /// Bytes bound through a true zero-copy initializer path.
    pub zero_copy_bytes: usize,
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
    /// Total selected RSMF tensor source bytes for initializer bindings.
    pub initializer_source_bytes: usize,
    /// Total initializer bytes kept in borrowed zero-copy residency.
    pub initializer_zero_copy_bytes: usize,
    /// ORT provider allocator memory accounting, if the active allocator
    /// reports ORT's `MaxInUse` byte counter.
    pub provider_allocator_bytes: RuntimeMemoryMeasurement,
    /// Raw and parsed ORT provider allocator statistics, if the active
    /// allocator reports them.
    pub provider_allocator_stats: RuntimeAllocatorStats,
    /// Process resident set size observed after session construction, if
    /// supported on this target.
    pub process_resident_set_bytes: RuntimeMemoryMeasurement,
    /// I/O binding policy configured for this session.
    pub io_binding: IoBindingPolicy,
}

impl SessionMemoryReport {
    /// Number of initializer bindings materialized for this session.
    #[must_use]
    pub fn initializer_count(&self) -> usize {
        self.initializers.len()
    }
}

pub(crate) fn runtime_capability_report() -> RuntimeCapabilityReport {
    RuntimeCapabilityReport {
        ort_cpu_io_binding: RuntimeCapability::Available,
        ort_provider_allocator_stats: provider_allocator_stats_capability(),
        mmap_initializer_zero_copy: RuntimeCapability::unavailable(
            "ONNX external initializer binding currently requires an ORT-owned value; borrowed RSMF mmap initializer lifetimes are not exposed safely",
        ),
        native_decoder_i8_q8_q4_direct_kernels: RuntimeCapability::Available,
        native_decoder_qk_family_direct_kernels: RuntimeCapability::Available,
        native_decoder_fused_lm_head_continuous_batching: RuntimeCapability::Available,
        native_decoder_fused_qkv_attention_mlp: RuntimeCapability::unavailable(
            "continuous batching interleaves decode steps and fuses LM-head projection only; Q/K/V, attention, and MLP are not fused across requests",
        ),
        native_decoder_metal_wgpu: RuntimeCapability::unavailable(
            "Metal/WGPU native decoder kernels are not implemented in this build",
        ),
        ort_coreml_execution_provider: RuntimeCapability::unavailable(
            "CoreML execution provider registration is not implemented in the default ORT graph runtime",
        ),
        sentencepiece_model_protobuf: RuntimeCapability::Available,
        serving_bearer_auth: RuntimeCapability::Available,
        serving_load_shedding: RuntimeCapability::Available,
        serving_tls: RuntimeCapability::unavailable(
            "the dependency-light std HTTP server does not terminate TLS; deploy behind a TLS proxy or add an optional serving crate",
        ),
        serving_streaming: RuntimeCapability::unavailable(
            "the dependency-light std HTTP server returns complete JSON responses and does not implement streaming",
        ),
        distributed_quotas: RuntimeCapability::unavailable(
            "tenant quotas are local to one RuntimeExecutor; no distributed quota store is implemented",
        ),
        r6_multi_device_expert_runtime: RuntimeCapability::unavailable(
            "production multi-device/expert execution is not implemented in rsmf-runtime; rsmf-moe-runtime remains a separate proof-of-concept crate",
        ),
    }
}

#[cfg(feature = "ort-api-23-allocator-stats")]
fn provider_allocator_stats_capability() -> RuntimeCapability {
    RuntimeCapability::Available
}

#[cfg(not(feature = "ort-api-23-allocator-stats"))]
fn provider_allocator_stats_capability() -> RuntimeCapability {
    RuntimeCapability::unavailable(
        "rsmf-runtime was built without the ort-api-23-allocator-stats feature",
    )
}

pub(crate) fn current_process_resident_set_bytes() -> RuntimeMemoryMeasurement {
    current_process_resident_set_bytes_impl()
}

#[cfg(target_os = "linux")]
fn current_process_resident_set_bytes_impl() -> RuntimeMemoryMeasurement {
    let status = match std::fs::read_to_string("/proc/self/status") {
        Ok(status) => status,
        Err(error) => {
            return RuntimeMemoryMeasurement::unavailable(format!(
                "failed to read /proc/self/status: {error}"
            ));
        }
    };
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let Some(kib) = parts.next().and_then(|value| value.parse::<usize>().ok()) else {
            return RuntimeMemoryMeasurement::unavailable("VmRSS was not a numeric KiB value");
        };
        return match kib.checked_mul(1024) {
            Some(bytes) => RuntimeMemoryMeasurement::available(bytes),
            None => RuntimeMemoryMeasurement::unavailable("VmRSS byte count overflowed usize"),
        };
    }
    RuntimeMemoryMeasurement::unavailable("/proc/self/status did not contain VmRSS")
}

#[cfg(not(target_os = "linux"))]
fn current_process_resident_set_bytes_impl() -> RuntimeMemoryMeasurement {
    RuntimeMemoryMeasurement::unavailable(
        "process RSS measurement is implemented only for Linux /proc in this build",
    )
}
