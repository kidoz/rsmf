use super::*;

/// Native decoder backend requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeDecoderBackend {
    /// Let the runtime choose the best available backend.
    Auto,
    /// Deterministic single-threaded CPU reference backend.
    CpuReference,
    /// CPU backend with threaded output-logit projection.
    CpuThreaded,
    /// macOS Accelerate / vecLib BLAS backend for f32 linear projections.
    AppleCpuAccelerate,
    /// Reserved Metal/WGPU backend for LM-head projection.
    MetalWgpuLmHead,
    /// Reserved Metal/WGPU backend for full native decoder kernels.
    MetalWgpuFullDecoder,
    /// Reserved ONNX Runtime CoreML execution-provider backend for graph
    /// payloads, not the native decoder path.
    OrtCoreMl,
    /// Select the best available accelerated backend in this build.
    Accelerated,
}

/// Options for selecting RSMF tensor variants when loading native decoder
/// weights.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NativeDecoderWeightOptions {
    /// Optional global RSMF variant index per tensor name. Missing names load
    /// the canonical variant.
    pub tensor_variants: HashMap<String, u32>,
    /// Allow native decoder weights stored as lower-precision or quantized
    /// dtypes to be decoded into the CPU f32 execution path.
    ///
    /// `F16`/`BF16` are always accepted by the native decoder contract. Raw
    /// integer weights are accepted only when this flag is set, because they
    /// imply a lossy dequantization policy.
    pub allow_lossy_quantized: bool,
}

/// Token sampling controls for native decoder generation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NativeDecoderSamplingOptions {
    /// Sampling temperature. `None` keeps greedy argmax behavior. `Some(value)`
    /// must be positive and finite.
    pub temperature: Option<f32>,
    /// Optional top-k candidate cap. When present, must be greater than zero.
    pub top_k: Option<usize>,
    /// Optional nucleus probability cap in `(0, 1]`.
    pub top_p: Option<f32>,
    /// Optional deterministic sampler seed. A fixed internal seed is used when
    /// sampling is enabled and this is omitted.
    pub seed: Option<u64>,
    /// Optional repetition penalty applied to prompt and generated tokens before
    /// selecting the next token. Values must be finite and at least `1.0`.
    pub repetition_penalty: Option<f32>,
}

/// Performance controls for native decoder execution.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NativeDecoderPerformanceOptions {
    /// Optional page size for KV-cache allocation. This currently controls
    /// page-sized reserve growth for the CPU cache and records page accounting.
    pub kv_cache_page_size_tokens: Option<usize>,
    /// Optional CPU worker count for threaded CPU paths.
    pub cpu_threads: Option<usize>,
    /// Optional prompt prefill chunk size. The current CPU path still executes
    /// token steps serially inside each chunk, but this bounds the scheduling
    /// unit for longer prompts and future chunked kernels.
    pub prefill_chunk_size: Option<usize>,
    /// Optional prefix-cache entry limit for resident sessions. `None` disables
    /// prefix cache lookup/insertion.
    pub prefix_cache_max_entries: Option<usize>,
    /// Optional prefix-cache resident byte limit. When omitted, entries are
    /// limited only by `prefix_cache_max_entries`.
    pub prefix_cache_max_bytes: Option<usize>,
    /// CPU attention implementation to use for cached decode steps.
    pub attention: NativeDecoderAttentionImplementation,
    /// Optional maximum requests to admit into one native token-level
    /// continuous-batching scheduling pass.
    pub continuous_batch_max_requests: Option<usize>,
}

/// CPU attention implementation for native decoder cached decode steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NativeDecoderAttentionImplementation {
    /// Scalar reference implementation.
    #[default]
    Scalar,
    /// Allocation-reduced CPU implementation that reuses per-head score
    /// scratch space while preserving paged KV-cache semantics.
    CpuTiled,
}

/// Options for native decoder greedy generation.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderRunOptions {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Optional EOS token ids. When empty, ids from [`NativeDecoderConfig`] are
    /// used.
    pub eos_token_ids: Vec<i64>,
    /// Requested backend.
    pub backend: NativeDecoderBackend,
    /// Weight variant selection options.
    pub weight_options: NativeDecoderWeightOptions,
    /// Sampling controls. The default preserves greedy argmax behavior.
    pub sampling: NativeDecoderSamplingOptions,
    /// Performance controls for cache allocation and CPU dispatch.
    pub performance: NativeDecoderPerformanceOptions,
    /// Minimum number of new tokens to emit before stop-token checks apply.
    pub min_new_tokens: usize,
    /// Optional stop-token override. When empty, [`Self::eos_token_ids`] and
    /// then [`NativeDecoderConfig::eos_token_ids`] are used.
    pub stop_token_ids: Vec<i64>,
    /// Whether to retain prompt-step logits in [`NativeDecoderGenerateOutput`].
    pub return_prompt_logits: bool,
}

impl Default for NativeDecoderRunOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1,
            eos_token_ids: Vec::new(),
            backend: NativeDecoderBackend::Auto,
            weight_options: NativeDecoderWeightOptions::default(),
            sampling: NativeDecoderSamplingOptions::default(),
            performance: NativeDecoderPerformanceOptions::default(),
            min_new_tokens: 0,
            stop_token_ids: Vec::new(),
            return_prompt_logits: false,
        }
    }
}

/// Reference logits check request for native decoder verification.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderReferenceLogitCheck {
    /// Token ids to feed one step at a time.
    pub input_token_ids: Vec<i64>,
    /// Expected next-token logits after each input token step.
    pub expected_logits: Vec<Vec<f32>>,
    /// Maximum accepted absolute difference.
    pub tolerance_abs: f32,
    /// Requested backend for the check.
    pub backend: NativeDecoderBackend,
    /// Weight variant selection options.
    pub weight_options: NativeDecoderWeightOptions,
    /// Performance controls for cache allocation and CPU dispatch.
    pub performance: NativeDecoderPerformanceOptions,
}

/// Report from a native decoder reference logits check.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderReferenceLogitReport {
    /// Number of logits rows compared.
    pub compared_logits: usize,
    /// Number of scalar values compared.
    pub compared_values: usize,
    /// Largest absolute difference observed.
    pub max_abs_diff: f32,
    /// Tolerance used for the check.
    pub tolerance_abs: f32,
    /// Backend actually used by this check.
    pub backend: NativeDecoderBackend,
}
