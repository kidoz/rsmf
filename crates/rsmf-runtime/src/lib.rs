//! Production-oriented ONNX Runtime integration for RSMF graph payloads.
//!
//! `rsmf-runtime` keeps RSMF's storage/container boundary intact: graph bytes
//! remain opaque ONNX / ORT payloads, while this crate owns session lifecycle,
//! typed request values, runtime options, and session caching.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::{self, Debug};
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
use rsmf_core::{LogicalDtype, RsmfError, RsmfFile, StorageDtype, TensorDescriptor, TensorView};
mod error;
mod network;
mod onnx;

pub use error::{Result, RuntimeError};
pub use network::{
    RUNTIME_NETWORK_PROTOCOL_VERSION, RuntimeNetworkMetrics, RuntimeNetworkRunRequest,
    RuntimeNetworkRunResponse, RuntimeNetworkServer, RuntimeNetworkServerConfig,
    RuntimeNetworkServerHandle, RuntimeNetworkTimings,
};
use onnx::{OnnxTensorDataType, onnx_initializers};

use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::info_span;

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

/// Canonical RSMF asset name for native decoder configuration.
pub const NATIVE_DECODER_CONFIG_ASSET: &str = "config.json";

/// Canonical RSMF asset name for the tokenizer payload used by native decoders.
pub const NATIVE_DECODER_TOKENIZER_ASSET: &str = "tokenizer.json";

/// Optional RSMF asset name for generation defaults.
pub const NATIVE_DECODER_GENERATION_CONFIG_ASSET: &str = "generation_config.json";

/// First native decoder family supported by the model contract layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeDecoderFamily {
    /// LLaMA-style decoder-only transformer with SwiGLU MLP and RMSNorm.
    Llama,
}

impl NativeDecoderFamily {
    fn from_model_type(model_type: &str) -> Result<Self> {
        match model_type {
            "llama" => Ok(Self::Llama),
            other => Err(RuntimeError::UnsupportedNativeDecoder {
                family: other.to_string(),
            }),
        }
    }
}

/// Parsed native decoder configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeDecoderConfig {
    /// Decoder model family.
    pub family: NativeDecoderFamily,
    /// Hidden state width.
    pub hidden_size: usize,
    /// MLP intermediate width.
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value attention heads.
    pub num_key_value_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum configured position count.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base theta.
    pub rope_theta: f32,
    /// Whether the LM head shares token embedding weights.
    pub tie_word_embeddings: bool,
    /// Optional beginning-of-sequence token id.
    pub bos_token_id: Option<i64>,
    /// Optional end-of-sequence token ids.
    pub eos_token_ids: Vec<i64>,
    /// Optional padding token id.
    pub pad_token_id: Option<i64>,
}

impl NativeDecoderConfig {
    /// Parse a HuggingFace-style `config.json` asset into the RSMF native decoder
    /// contract.
    pub fn from_hf_config_json(bytes: &[u8]) -> Result<Self> {
        let raw: HfNativeDecoderConfig = serde_json::from_slice(bytes).map_err(|error| {
            RuntimeError::NativeDecoderConfigInvalid {
                reason: error.to_string(),
            }
        })?;
        let family = NativeDecoderFamily::from_model_type(&raw.model_type)?;
        let num_key_value_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
        let config = Self {
            family,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads,
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: raw.rope_theta.unwrap_or(10_000.0),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
            bos_token_id: raw.bos_token_id,
            eos_token_ids: raw.eos_token_id.map_or_else(Vec::new, TokenIds::into_vec),
            pad_token_id: raw.pad_token_id,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        validate_positive("hidden_size", self.hidden_size)?;
        validate_positive("intermediate_size", self.intermediate_size)?;
        validate_positive("num_hidden_layers", self.num_hidden_layers)?;
        validate_positive("num_attention_heads", self.num_attention_heads)?;
        validate_positive("num_key_value_heads", self.num_key_value_heads)?;
        validate_positive("vocab_size", self.vocab_size)?;
        validate_positive("max_position_embeddings", self.max_position_embeddings)?;
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "hidden_size must be divisible by num_attention_heads".to_string(),
            });
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            });
        }
        if self
            .num_key_value_heads
            .checked_mul(self.head_dim())
            .is_none()
        {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "key/value projection width overflow".to_string(),
            });
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "rms_norm_eps must be positive".to_string(),
            });
        }
        if self.rope_theta <= 0.0 {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "rope_theta must be positive".to_string(),
            });
        }
        Ok(())
    }

    /// Attention head width.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size
            .checked_div(self.num_attention_heads)
            .unwrap_or(0)
    }

    /// Width of the key/value projection output.
    #[must_use]
    pub fn key_value_width(&self) -> usize {
        self.num_key_value_heads.saturating_mul(self.head_dim())
    }

    /// Number of query heads sharing one key/value head.
    #[must_use]
    pub fn query_groups_per_key_value_head(&self) -> usize {
        self.num_attention_heads
            .checked_div(self.num_key_value_heads)
            .unwrap_or(0)
    }
}

/// Native decoder asset contract discovered from an RSMF file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeDecoderAssets {
    /// Required config asset name.
    pub config_asset: String,
    /// Required tokenizer asset name.
    pub tokenizer_asset: String,
    /// Optional generation config asset name.
    pub generation_config_asset: Option<String>,
}

/// One required tensor in the native decoder contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeDecoderTensorBinding {
    /// Stable logical role used by the native runtime.
    pub role: String,
    /// Required RSMF tensor name.
    pub tensor_name: String,
    /// Required logical tensor shape.
    pub shape: Vec<u64>,
    /// Logical dtype recorded in the RSMF manifest.
    pub dtype: String,
}

/// Native decoder model contract resolved from an RSMF file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeDecoderContract {
    /// Parsed decoder configuration.
    pub config: NativeDecoderConfig,
    /// Required and optional asset names.
    pub assets: NativeDecoderAssets,
    /// Required tensor bindings validated against the RSMF manifest.
    pub tensors: Vec<NativeDecoderTensorBinding>,
}

impl NativeDecoderContract {
    fn from_file(file: &RsmfFile) -> Result<Self> {
        let config_asset = file.asset(NATIVE_DECODER_CONFIG_ASSET).ok_or_else(|| {
            RuntimeError::NativeDecoderAssetMissing {
                asset_name: NATIVE_DECODER_CONFIG_ASSET.to_string(),
            }
        })?;
        let config = NativeDecoderConfig::from_hf_config_json(config_asset.bytes)?;
        if file.asset(NATIVE_DECODER_TOKENIZER_ASSET).is_none() {
            return Err(RuntimeError::NativeDecoderAssetMissing {
                asset_name: NATIVE_DECODER_TOKENIZER_ASSET.to_string(),
            });
        }
        let assets = NativeDecoderAssets {
            config_asset: NATIVE_DECODER_CONFIG_ASSET.to_string(),
            tokenizer_asset: NATIVE_DECODER_TOKENIZER_ASSET.to_string(),
            generation_config_asset: file
                .asset(NATIVE_DECODER_GENERATION_CONFIG_ASSET)
                .map(|asset| asset.name.to_string()),
        };
        let expected = expected_native_decoder_tensors(&config)?;
        let tensors = expected
            .into_iter()
            .map(|expected| validate_native_decoder_tensor(file, expected))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            assets,
            tensors,
        })
    }
}

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

/// Minimal tokenizer contract supported by the native decoder text API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDecoderTokenizer {
    /// Tokenizer model kind from `tokenizer.json`.
    pub model_type: String,
    /// Token string to token id map.
    pub vocab: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    unk_token: Option<String>,
    mode: NativeDecoderTokenizerMode,
    pre_tokenizer: NativeDecoderPreTokenizer,
    bpe_ranks: HashMap<(String, String), usize>,
}

impl NativeDecoderTokenizer {
    fn from_json(bytes: &[u8]) -> Result<Self> {
        let raw: NativeDecoderTokenizerJson = serde_json::from_slice(bytes).map_err(|error| {
            RuntimeError::NativeDecoderTokenizerInvalid {
                reason: error.to_string(),
            }
        })?;
        reject_supported_tokenizer_component("normalizer", raw.normalizer.as_ref())?;
        reject_supported_tokenizer_component("post_processor", raw.post_processor.as_ref())?;
        let pre_tokenizer = NativeDecoderPreTokenizer::from_json(raw.pre_tokenizer.as_ref())?;
        let mode = match raw.model.tokenizer_type.as_str() {
            "WordLevel" => NativeDecoderTokenizerMode::WordLevel,
            "BPE" => NativeDecoderTokenizerMode::Bpe,
            other => {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!(
                        "only WordLevel and BPE tokenizer.json assets are supported, got {other}"
                    ),
                });
            }
        };
        let bpe_ranks = if mode == NativeDecoderTokenizerMode::Bpe {
            bpe_ranks_from_merges(&raw.model.merges)?
        } else {
            HashMap::new()
        };
        let unk_token = raw.model.unk_token;
        let mut vocab = raw.model.vocab;
        for added in raw.added_tokens {
            vocab.entry(added.content).or_insert(added.id);
        }
        if vocab.is_empty() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "tokenizer vocab must not be empty".to_string(),
            });
        }
        let mut id_to_token = HashMap::with_capacity(vocab.len());
        for (token, token_id) in &vocab {
            if id_to_token.insert(*token_id, token.clone()).is_some() {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!("duplicate tokenizer id {token_id}"),
                });
            }
        }
        if let Some(unk_token) = &unk_token {
            if !vocab.contains_key(unk_token) {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!("unk_token {unk_token} is not present in vocab"),
                });
            }
        }
        Ok(Self {
            model_type: raw.model.tokenizer_type,
            vocab,
            id_to_token,
            unk_token,
            mode,
            pre_tokenizer,
            bpe_ranks,
        })
    }

    /// Encode text to token ids.
    ///
    /// WordLevel tokenizers use whitespace token lookup. BPE tokenizers support
    /// simple whitespace or ByteLevel-style pre-tokenization, vocab/merges, and
    /// exact special-token lookup. Unsupported tokenizer components fail at
    /// load time with typed errors.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let mut token_ids = Vec::new();
        for token in self.pre_tokenizer.pieces(text) {
            match self.mode {
                NativeDecoderTokenizerMode::WordLevel => {
                    token_ids.push(self.lookup_token_id(&token)?);
                }
                NativeDecoderTokenizerMode::Bpe => {
                    token_ids.extend(self.encode_bpe_piece(&token)?);
                }
            }
        }
        if token_ids.is_empty() {
            return Err(RuntimeError::NativeDecoderPromptEmpty);
        }
        Ok(token_ids)
    }

    /// Decode token ids back to text.
    pub fn decode(&self, token_ids: &[i64]) -> Result<String> {
        let tokens = token_ids
            .iter()
            .map(|token_id| {
                self.id_to_token.get(token_id).cloned().ok_or_else(|| {
                    RuntimeError::NativeDecoderTokenizerTokenUnknown {
                        token: token_id.to_string(),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        match self.mode {
            NativeDecoderTokenizerMode::WordLevel => Ok(tokens.join(" ")),
            NativeDecoderTokenizerMode::Bpe => Ok(decode_bpe_tokens(&tokens)),
        }
    }

    fn lookup_token_id(&self, token: &str) -> Result<i64> {
        if let Some(token_id) = self.vocab.get(token) {
            Ok(*token_id)
        } else if let Some(unk_token) = &self.unk_token {
            self.vocab.get(unk_token).copied().ok_or_else(|| {
                RuntimeError::NativeDecoderTokenizerTokenUnknown {
                    token: token.to_string(),
                }
            })
        } else {
            Err(RuntimeError::NativeDecoderTokenizerTokenUnknown {
                token: token.to_string(),
            })
        }
    }

    fn encode_bpe_piece(&self, piece: &str) -> Result<Vec<i64>> {
        if self.vocab.contains_key(piece) {
            return self.lookup_token_id(piece).map(|token_id| vec![token_id]);
        }
        let mut symbols = piece.chars().map(|c| c.to_string()).collect::<Vec<_>>();
        while symbols.len() > 1 {
            let Some((best_index, _)) = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    self.bpe_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (index, *rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let merged = format!("{}{}", symbols[best_index], symbols[best_index + 1]);
            symbols.splice(best_index..=best_index + 1, [merged]);
        }
        symbols
            .into_iter()
            .map(|symbol| self.lookup_token_id(&symbol))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeDecoderTokenizerMode {
    WordLevel,
    Bpe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeDecoderPreTokenizer {
    Whitespace,
    ByteLevel { add_prefix_space: bool },
}

impl NativeDecoderPreTokenizer {
    fn from_json(value: Option<&serde_json::Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::Whitespace);
        };
        if value.is_null() {
            return Ok(Self::Whitespace);
        }
        let Some(kind) = value.get("type").and_then(serde_json::Value::as_str) else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "pre_tokenizer.type is required when pre_tokenizer is present".to_string(),
            });
        };
        match kind {
            "Whitespace" | "WhitespaceSplit" => Ok(Self::Whitespace),
            "ByteLevel" => Ok(Self::ByteLevel {
                add_prefix_space: value
                    .get("add_prefix_space")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            }),
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported pre_tokenizer {other}"),
            }),
        }
    }

    fn pieces(self, text: &str) -> Vec<String> {
        match self {
            Self::Whitespace => text.split_whitespace().map(ToString::to_string).collect(),
            Self::ByteLevel { add_prefix_space } => {
                let mut pieces = Vec::new();
                for (index, piece) in text.split_whitespace().enumerate() {
                    if index == 0 && !add_prefix_space {
                        pieces.push(piece.to_string());
                    } else {
                        pieces.push(format!("Ġ{piece}"));
                    }
                }
                pieces
            }
        }
    }
}

/// Output from native decoder text generation.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderTextGenerateOutput {
    /// Original prompt string.
    pub prompt: String,
    /// Prompt token ids followed by generated token ids.
    pub token_ids: Vec<i64>,
    /// Newly generated token ids only.
    pub generated_token_ids: Vec<i64>,
    /// Decoded generated text only.
    pub generated_text: String,
    /// Decoded prompt plus generated text.
    pub text: String,
    /// Per-generation-step logits, one row per generated token.
    pub logits: Vec<Vec<f32>>,
    /// Optional per-prompt-token logits. Empty unless
    /// [`NativeDecoderRunOptions::return_prompt_logits`] is enabled.
    pub prompt_logits: Vec<Vec<f32>>,
    /// Backend actually used by this run.
    pub backend: NativeDecoderBackend,
}

/// Resident native decoder session with decoded weights and tokenizer cached in
/// memory.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderSession {
    /// Decoded native decoder weights.
    pub weights: NativeDecoderWeights,
    /// Decoded native decoder tokenizer.
    pub tokenizer: NativeDecoderTokenizer,
}

impl NativeDecoderSession {
    /// Generate token ids without reloading weights from the RSMF file.
    pub fn generate_token_ids(
        &self,
        input_token_ids: &[i64],
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderGenerateOutput> {
        let backend = resolve_native_decoder_backend(options.backend)?;
        native_decoder_generate_with_backend(&self.weights, input_token_ids, options, backend)
    }

    /// Generate text without reloading weights or tokenizer from the RSMF file.
    pub fn generate_text(
        &self,
        prompt: &str,
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderTextGenerateOutput> {
        let prompt_token_ids = self.tokenizer.encode(prompt)?;
        let output = self.generate_token_ids(&prompt_token_ids, options)?;
        let generated_text = self.tokenizer.decode(&output.generated_token_ids)?;
        let text = self.tokenizer.decode(&output.token_ids)?;
        Ok(NativeDecoderTextGenerateOutput {
            prompt: prompt.to_string(),
            token_ids: output.token_ids,
            generated_token_ids: output.generated_token_ids,
            generated_text,
            text,
            logits: output.logits,
            prompt_logits: output.prompt_logits,
            backend: output.backend,
        })
    }

    /// Compare logits against a supplied reference without reloading weights.
    pub fn check_reference_logits(
        &self,
        check: NativeDecoderReferenceLogitCheck,
    ) -> Result<NativeDecoderReferenceLogitReport> {
        let backend = resolve_native_decoder_backend(check.backend)?;
        native_decoder_check_reference_logits(&self.weights, check, backend)
    }
}

#[derive(Debug, Deserialize)]
struct NativeDecoderTokenizerJson {
    model: NativeDecoderTokenizerModelJson,
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    post_processor: Option<serde_json::Value>,
    #[serde(default)]
    added_tokens: Vec<NativeDecoderAddedTokenJson>,
}

#[derive(Debug, Deserialize)]
struct NativeDecoderTokenizerModelJson {
    #[serde(rename = "type")]
    tokenizer_type: String,
    #[serde(default)]
    vocab: HashMap<String, i64>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    merges: Vec<NativeDecoderBpeMergeJson>,
}

#[derive(Debug, Deserialize)]
struct NativeDecoderAddedTokenJson {
    id: i64,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NativeDecoderBpeMergeJson {
    Text(String),
    Pair([String; 2]),
}

fn reject_supported_tokenizer_component(
    name: &str,
    value: Option<&serde_json::Value>,
) -> Result<()> {
    if value.is_some_and(|value| !value.is_null()) {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("{name} is not supported by the native tokenizer yet"),
        });
    }
    Ok(())
}

fn bpe_ranks_from_merges(
    merges: &[NativeDecoderBpeMergeJson],
) -> Result<HashMap<(String, String), usize>> {
    let mut ranks = HashMap::with_capacity(merges.len());
    for (rank, merge) in merges.iter().enumerate() {
        let (left, right) =
            match merge {
                NativeDecoderBpeMergeJson::Text(value) => {
                    let mut parts = value.split_whitespace();
                    let left = parts.next().ok_or_else(|| {
                        RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("invalid BPE merge {value:?}"),
                        }
                    })?;
                    let right = parts.next().ok_or_else(|| {
                        RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("invalid BPE merge {value:?}"),
                        }
                    })?;
                    if parts.next().is_some() {
                        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("invalid BPE merge {value:?}"),
                        });
                    }
                    (left.to_string(), right.to_string())
                }
                NativeDecoderBpeMergeJson::Pair([left, right]) => (left.clone(), right.clone()),
            };
        ranks.insert((left, right), rank);
    }
    Ok(ranks)
}

fn decode_bpe_tokens(tokens: &[String]) -> String {
    let mut text = String::new();
    for token in tokens {
        if let Some(rest) = token.strip_prefix('Ġ') {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(rest);
        } else {
            text.push_str(token);
        }
    }
    text
}

/// Owned LLaMA-style layer weights decoded for native decoder execution.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderLayerWeights {
    /// Input RMSNorm weight, shape `[hidden_size]`.
    pub input_layernorm: Vec<f32>,
    /// Post-attention RMSNorm weight, shape `[hidden_size]`.
    pub post_attention_layernorm: Vec<f32>,
    /// Query projection weight, row-major shape `[hidden_size, hidden_size]`.
    pub q_proj: Vec<f32>,
    /// Key projection weight, row-major shape `[kv_width, hidden_size]`.
    pub k_proj: Vec<f32>,
    /// Value projection weight, row-major shape `[kv_width, hidden_size]`.
    pub v_proj: Vec<f32>,
    /// Attention output projection weight, row-major shape `[hidden_size, hidden_size]`.
    pub o_proj: Vec<f32>,
    /// SwiGLU gate projection weight, row-major shape `[intermediate_size, hidden_size]`.
    pub gate_proj: Vec<f32>,
    /// SwiGLU up projection weight, row-major shape `[intermediate_size, hidden_size]`.
    pub up_proj: Vec<f32>,
    /// SwiGLU down projection weight, row-major shape `[hidden_size, intermediate_size]`.
    pub down_proj: Vec<f32>,
}

impl NativeDecoderLayerWeights {
    fn as_cpu(&self) -> NativeDecoderCpuLayerWeights<'_> {
        NativeDecoderCpuLayerWeights {
            input_layernorm: &self.input_layernorm,
            post_attention_layernorm: &self.post_attention_layernorm,
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.o_proj,
            gate_proj: &self.gate_proj,
            up_proj: &self.up_proj,
            down_proj: &self.down_proj,
        }
    }
}

/// Owned native decoder weights decoded from an RSMF file.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderWeights {
    /// Parsed decoder configuration used to validate these weights.
    pub config: NativeDecoderConfig,
    /// Token embedding matrix, row-major shape `[vocab_size, hidden_size]`.
    pub token_embedding: Vec<f32>,
    /// Final RMSNorm weight, shape `[hidden_size]`.
    pub final_norm: Vec<f32>,
    /// Optional LM head matrix, row-major shape `[vocab_size, hidden_size]`.
    /// When absent, token embeddings are used as tied output embeddings.
    pub lm_head: Option<Vec<f32>>,
    /// Per-layer decoded weights.
    pub layers: Vec<NativeDecoderLayerWeights>,
}

/// KV cache for native decoder CPU reference generation.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderKvCache {
    layers: Vec<NativeDecoderLayerKvCache>,
    position: usize,
    kv_width: usize,
    page_size_tokens: Option<usize>,
}

impl NativeDecoderKvCache {
    /// Create an empty KV cache sized for the supplied decoder configuration.
    #[must_use]
    pub fn new(config: &NativeDecoderConfig) -> Self {
        Self::with_page_size(config, None)
    }

    /// Create an empty KV cache with page-sized allocation growth.
    ///
    /// Returns an error when `page_size_tokens` is zero.
    pub fn new_paged(config: &NativeDecoderConfig, page_size_tokens: usize) -> Result<Self> {
        if page_size_tokens == 0 {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "kv_cache_page_size_tokens must be positive".to_string(),
            });
        }
        Ok(Self::with_page_size(config, Some(page_size_tokens)))
    }

    fn with_page_size(config: &NativeDecoderConfig, page_size_tokens: Option<usize>) -> Self {
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|_| NativeDecoderLayerKvCache {
                    keys: Vec::new(),
                    values: Vec::new(),
                    key_pages: Vec::new(),
                    value_pages: Vec::new(),
                })
                .collect(),
            position: 0,
            kv_width: config.key_value_width(),
            page_size_tokens,
        }
    }

    /// Number of tokens already appended to the cache.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Optional KV-cache allocation page size in tokens.
    #[must_use]
    pub fn page_size_tokens(&self) -> Option<usize> {
        self.page_size_tokens
    }

    /// Number of pages allocated for the current cache position.
    #[must_use]
    pub fn allocated_pages(&self) -> usize {
        self.page_size_tokens
            .map(|page_size| self.position.div_ceil(page_size))
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct NativeDecoderLayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    key_pages: Vec<Vec<f32>>,
    value_pages: Vec<Vec<f32>>,
}

/// Output from one native decoder step.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderStepOutput {
    /// Logits for the next token, shape `[vocab_size]`.
    pub logits: Vec<f32>,
    /// Greedy argmax token id selected from `logits`.
    pub next_token_id: i64,
}

/// Output from native decoder greedy generation.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderGenerateOutput {
    /// Prompt tokens followed by generated tokens.
    pub token_ids: Vec<i64>,
    /// Newly generated token ids only.
    pub generated_token_ids: Vec<i64>,
    /// Per-generation-step logits, one row per generated token.
    pub logits: Vec<Vec<f32>>,
    /// Optional per-prompt-token logits. Empty unless
    /// [`NativeDecoderRunOptions::return_prompt_logits`] is enabled.
    pub prompt_logits: Vec<Vec<f32>>,
    /// Backend actually used by this run.
    pub backend: NativeDecoderBackend,
}

#[derive(Debug, Deserialize)]
struct HfNativeDecoderConfig {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    bos_token_id: Option<i64>,
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
    #[serde(default)]
    pad_token_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(i64),
    Many(Vec<i64>),
}

impl TokenIds {
    fn into_vec(self) -> Vec<i64> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

struct ExpectedNativeDecoderTensor {
    role: String,
    tensor_name: String,
    shape: Vec<u64>,
}

fn validate_positive(field: &str, value: usize) -> Result<()> {
    if value == 0 {
        Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: format!("{field} must be positive"),
        })
    } else {
        Ok(())
    }
}

fn expected_native_decoder_tensors(
    config: &NativeDecoderConfig,
) -> Result<Vec<ExpectedNativeDecoderTensor>> {
    match config.family {
        NativeDecoderFamily::Llama => expected_llama_tensors(config),
    }
}

fn expected_llama_tensors(
    config: &NativeDecoderConfig,
) -> Result<Vec<ExpectedNativeDecoderTensor>> {
    let hidden = usize_to_u64(config.hidden_size, "hidden_size")?;
    let intermediate = usize_to_u64(config.intermediate_size, "intermediate_size")?;
    let vocab = usize_to_u64(config.vocab_size, "vocab_size")?;
    let kv_width = usize_to_u64(
        config.num_key_value_heads.saturating_mul(config.head_dim()),
        "kv projection width",
    )?;
    let mut tensors = vec![
        expected_tensor(
            "token_embedding",
            "model.embed_tokens.weight",
            vec![vocab, hidden],
        ),
        expected_tensor("final_norm", "model.norm.weight", vec![hidden]),
    ];
    if !config.tie_word_embeddings {
        tensors.push(expected_tensor(
            "lm_head",
            "lm_head.weight",
            vec![vocab, hidden],
        ));
    }
    for layer in 0..config.num_hidden_layers {
        tensors.extend([
            expected_tensor(
                format!("layers.{layer}.input_layernorm"),
                format!("model.layers.{layer}.input_layernorm.weight"),
                vec![hidden],
            ),
            expected_tensor(
                format!("layers.{layer}.post_attention_layernorm"),
                format!("model.layers.{layer}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            expected_tensor(
                format!("layers.{layer}.self_attn.q_proj"),
                format!("model.layers.{layer}.self_attn.q_proj.weight"),
                vec![hidden, hidden],
            ),
            expected_tensor(
                format!("layers.{layer}.self_attn.k_proj"),
                format!("model.layers.{layer}.self_attn.k_proj.weight"),
                vec![kv_width, hidden],
            ),
            expected_tensor(
                format!("layers.{layer}.self_attn.v_proj"),
                format!("model.layers.{layer}.self_attn.v_proj.weight"),
                vec![kv_width, hidden],
            ),
            expected_tensor(
                format!("layers.{layer}.self_attn.o_proj"),
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                vec![hidden, hidden],
            ),
            expected_tensor(
                format!("layers.{layer}.mlp.gate_proj"),
                format!("model.layers.{layer}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            expected_tensor(
                format!("layers.{layer}.mlp.up_proj"),
                format!("model.layers.{layer}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
            ),
            expected_tensor(
                format!("layers.{layer}.mlp.down_proj"),
                format!("model.layers.{layer}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
            ),
        ]);
    }
    Ok(tensors)
}

fn expected_tensor(
    role: impl Into<String>,
    tensor_name: impl Into<String>,
    shape: Vec<u64>,
) -> ExpectedNativeDecoderTensor {
    ExpectedNativeDecoderTensor {
        role: role.into(),
        tensor_name: tensor_name.into(),
        shape,
    }
}

fn validate_native_decoder_tensor(
    file: &RsmfFile,
    expected: ExpectedNativeDecoderTensor,
) -> Result<NativeDecoderTensorBinding> {
    let tensor = file
        .manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == expected.tensor_name)
        .ok_or_else(|| RuntimeError::NativeDecoderTensorMissing {
            tensor_name: expected.tensor_name.clone(),
        })?;
    if tensor.shape != expected.shape {
        return Err(RuntimeError::NativeDecoderTensorShapeMismatch {
            tensor_name: tensor.name.clone(),
            expected_shape: format_shape(&expected.shape),
            actual_shape: format_shape(&tensor.shape),
        });
    }
    validate_native_decoder_weight_dtype(tensor)?;
    Ok(NativeDecoderTensorBinding {
        role: expected.role,
        tensor_name: tensor.name.clone(),
        shape: tensor.shape.clone(),
        dtype: format!("{:?}", tensor.dtype),
    })
}

fn validate_native_decoder_weight_dtype(tensor: &TensorDescriptor) -> Result<()> {
    match tensor.dtype {
        LogicalDtype::F32 | LogicalDtype::F16 | LogicalDtype::BF16 => Ok(()),
        dtype => Err(RuntimeError::NativeDecoderTensorDtypeUnsupported {
            tensor_name: tensor.name.clone(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

fn format_shape(shape: &[u64]) -> String {
    let values = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}

fn usize_to_u64(value: usize, field: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| RuntimeError::NativeDecoderConfigInvalid {
        reason: format!("{field} does not fit in u64"),
    })
}

/// Input buffer for one CPU reference native decoder block.
#[derive(Debug, Clone, Copy)]
pub struct NativeDecoderCpuBlockInput<'a> {
    /// Row-major hidden states with shape `[sequence_len, hidden_size]`.
    pub hidden_states: &'a [f32],
    /// Number of tokens in `hidden_states`.
    pub sequence_len: usize,
    /// Absolute position of the first token in `hidden_states`.
    pub position_start: usize,
}

/// Borrowed LLaMA-style layer weights for one CPU reference decoder block.
#[derive(Debug, Clone, Copy)]
pub struct NativeDecoderCpuLayerWeights<'a> {
    /// Input RMSNorm weight, shape `[hidden_size]`.
    pub input_layernorm: &'a [f32],
    /// Post-attention RMSNorm weight, shape `[hidden_size]`.
    pub post_attention_layernorm: &'a [f32],
    /// Query projection weight, row-major shape `[hidden_size, hidden_size]`.
    pub q_proj: &'a [f32],
    /// Key projection weight, row-major shape `[kv_width, hidden_size]`.
    pub k_proj: &'a [f32],
    /// Value projection weight, row-major shape `[kv_width, hidden_size]`.
    pub v_proj: &'a [f32],
    /// Attention output projection weight, row-major shape `[hidden_size, hidden_size]`.
    pub o_proj: &'a [f32],
    /// SwiGLU gate projection weight, row-major shape `[intermediate_size, hidden_size]`.
    pub gate_proj: &'a [f32],
    /// SwiGLU up projection weight, row-major shape `[intermediate_size, hidden_size]`.
    pub up_proj: &'a [f32],
    /// SwiGLU down projection weight, row-major shape `[hidden_size, intermediate_size]`.
    pub down_proj: &'a [f32],
}

/// Output from one CPU reference native decoder block.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderCpuBlockOutput {
    /// Row-major hidden states with shape `[sequence_len, hidden_size]`.
    pub hidden_states: Vec<f32>,
}

/// CPU reference RMSNorm over row-major `[rows, hidden_size]` f32 data.
pub fn native_decoder_cpu_rms_norm(
    input: &[f32],
    rows: usize,
    hidden_size: usize,
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    validate_cpu_matrix_len("rms_norm", "input", input.len(), rows, hidden_size)?;
    validate_cpu_vector_len("rms_norm", "weight", weight.len(), hidden_size)?;
    if eps <= 0.0 {
        return Err(native_decoder_cpu_shape_error(
            "rms_norm",
            "eps must be positive",
        ));
    }
    let mut output = vec![0.0f32; input.len()];
    for row in 0..rows {
        let start = row * hidden_size;
        let values = &input[start..start + hidden_size];
        let mean_square =
            values.iter().map(|value| value * value).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for col in 0..hidden_size {
            output[start + col] = values[col] * scale * weight[col];
        }
    }
    Ok(output)
}

/// CPU reference row-major linear projection.
///
/// `input` has shape `[rows, in_features]`, `weight` has shape
/// `[out_features, in_features]`, and the returned buffer has shape
/// `[rows, out_features]`.
pub fn native_decoder_cpu_linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    validate_cpu_matrix_len("linear", "input", input.len(), rows, in_features)?;
    validate_cpu_matrix_len("linear", "weight", weight.len(), out_features, in_features)?;
    let output_len = cpu_element_count("linear", "output", rows, out_features)?;
    let mut output = vec![0.0f32; output_len];
    for row in 0..rows {
        for out_col in 0..out_features {
            let mut sum = 0.0f32;
            for in_col in 0..in_features {
                sum += input[row * in_features + in_col] * weight[out_col * in_features + in_col];
            }
            output[row * out_features + out_col] = sum;
        }
    }
    Ok(output)
}

/// CPU reference SiLU activation.
#[must_use]
pub fn native_decoder_cpu_silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Apply LLaMA-style RoPE in-place to row-major
/// `[sequence_len, num_heads, head_dim]` values.
pub fn native_decoder_cpu_apply_llama_rope(
    values: &mut [f32],
    sequence_len: usize,
    num_heads: usize,
    head_dim: usize,
    position_start: usize,
    rope_theta: f32,
) -> Result<()> {
    validate_cpu_matrix_len(
        "llama_rope",
        "values",
        values.len(),
        sequence_len,
        num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| native_decoder_cpu_shape_error("llama_rope", "head width overflow"))?,
    )?;
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(native_decoder_cpu_shape_error(
            "llama_rope",
            "head_dim must be a positive even number",
        ));
    }
    if rope_theta <= 0.0 {
        return Err(native_decoder_cpu_shape_error(
            "llama_rope",
            "rope_theta must be positive",
        ));
    }
    for token in 0..sequence_len {
        let position = (position_start + token) as f32;
        for head in 0..num_heads {
            let base = (token * num_heads + head) * head_dim;
            for dim in (0..head_dim).step_by(2) {
                let inv_freq = 1.0 / rope_theta.powf(dim as f32 / head_dim as f32);
                let angle = position * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let even = values[base + dim];
                let odd = values[base + dim + 1];
                values[base + dim] = even * cos - odd * sin;
                values[base + dim + 1] = even * sin + odd * cos;
            }
        }
    }
    Ok(())
}

/// CPU reference grouped-query causal self-attention.
///
/// `query` has shape `[sequence_len, num_attention_heads, head_dim]`; `key` and
/// `value` have shape `[sequence_len, num_key_value_heads, head_dim]`.
pub fn native_decoder_cpu_causal_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence_len: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    validate_cpu_positive("causal_attention", "sequence_len", sequence_len)?;
    validate_cpu_positive(
        "causal_attention",
        "num_attention_heads",
        num_attention_heads,
    )?;
    validate_cpu_positive(
        "causal_attention",
        "num_key_value_heads",
        num_key_value_heads,
    )?;
    validate_cpu_positive("causal_attention", "head_dim", head_dim)?;
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(native_decoder_cpu_shape_error(
            "causal_attention",
            "num_attention_heads must be divisible by num_key_value_heads",
        ));
    }
    let query_width = cpu_element_count(
        "causal_attention",
        "query width",
        num_attention_heads,
        head_dim,
    )?;
    let kv_width = cpu_element_count(
        "causal_attention",
        "key/value width",
        num_key_value_heads,
        head_dim,
    )?;
    let output_len = cpu_element_count("causal_attention", "output", sequence_len, query_width)?;
    validate_cpu_matrix_len(
        "causal_attention",
        "query",
        query.len(),
        sequence_len,
        query_width,
    )?;
    validate_cpu_matrix_len("causal_attention", "key", key.len(), sequence_len, kv_width)?;
    validate_cpu_matrix_len(
        "causal_attention",
        "value",
        value.len(),
        sequence_len,
        kv_width,
    )?;
    let groups = num_attention_heads / num_key_value_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; output_len];
    for token in 0..sequence_len {
        for head in 0..num_attention_heads {
            let kv_head = head / groups;
            let query_offset = (token * num_attention_heads + head) * head_dim;
            let mut scores = vec![0.0f32; token + 1];
            for (key_token, score) in scores.iter_mut().enumerate() {
                let key_offset = (key_token * num_key_value_heads + kv_head) * head_dim;
                *score = dot_product(
                    &query[query_offset..query_offset + head_dim],
                    &key[key_offset..key_offset + head_dim],
                ) * scale;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut weight_sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                weight_sum += *score;
            }
            let output_offset = (token * num_attention_heads + head) * head_dim;
            for (key_token, score) in scores.iter().enumerate() {
                let attention_weight = *score / weight_sum;
                let value_offset = (key_token * num_key_value_heads + kv_head) * head_dim;
                for dim in 0..head_dim {
                    output[output_offset + dim] += attention_weight * value[value_offset + dim];
                }
            }
        }
    }
    Ok(output)
}

/// CPU reference grouped-query attention for one query token over an existing
/// KV cache.
///
/// `query` has shape `[num_attention_heads, head_dim]`; `key_cache` and
/// `value_cache` have shape `[cache_len, num_key_value_heads, head_dim]`.
pub fn native_decoder_cpu_cached_attention(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    cache_len: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    validate_cpu_positive("cached_attention", "cache_len", cache_len)?;
    validate_cpu_positive(
        "cached_attention",
        "num_attention_heads",
        num_attention_heads,
    )?;
    validate_cpu_positive(
        "cached_attention",
        "num_key_value_heads",
        num_key_value_heads,
    )?;
    validate_cpu_positive("cached_attention", "head_dim", head_dim)?;
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(native_decoder_cpu_shape_error(
            "cached_attention",
            "num_attention_heads must be divisible by num_key_value_heads",
        ));
    }
    let query_width = cpu_element_count(
        "cached_attention",
        "query width",
        num_attention_heads,
        head_dim,
    )?;
    let kv_width = cpu_element_count(
        "cached_attention",
        "key/value width",
        num_key_value_heads,
        head_dim,
    )?;
    validate_cpu_vector_len("cached_attention", "query", query.len(), query_width)?;
    validate_cpu_matrix_len(
        "cached_attention",
        "key_cache",
        key_cache.len(),
        cache_len,
        kv_width,
    )?;
    validate_cpu_matrix_len(
        "cached_attention",
        "value_cache",
        value_cache.len(),
        cache_len,
        kv_width,
    )?;

    let groups = num_attention_heads / num_key_value_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; query_width];
    for head in 0..num_attention_heads {
        let kv_head = head / groups;
        let query_offset = head * head_dim;
        let mut scores = vec![0.0f32; cache_len];
        for (key_token, score) in scores.iter_mut().enumerate() {
            let key_offset = (key_token * num_key_value_heads + kv_head) * head_dim;
            *score = dot_product(
                &query[query_offset..query_offset + head_dim],
                &key_cache[key_offset..key_offset + head_dim],
            ) * scale;
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weight_sum = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            weight_sum += *score;
        }
        let output_offset = head * head_dim;
        for (key_token, score) in scores.iter().enumerate() {
            let attention_weight = *score / weight_sum;
            let value_offset = (key_token * num_key_value_heads + kv_head) * head_dim;
            for dim in 0..head_dim {
                output[output_offset + dim] += attention_weight * value_cache[value_offset + dim];
            }
        }
    }
    Ok(output)
}

/// CPU reference LLaMA-style decoder block over supplied f32 layer weights.
///
/// This is a correctness-oriented reference path for R4.2. It performs RMSNorm,
/// QKV projection, RoPE, causal grouped-query attention, output projection,
/// SwiGLU MLP, and residual additions. It does not allocate or consume KV cache.
pub fn native_decoder_cpu_llama_block(
    config: &NativeDecoderConfig,
    input: NativeDecoderCpuBlockInput<'_>,
    weights: NativeDecoderCpuLayerWeights<'_>,
) -> Result<NativeDecoderCpuBlockOutput> {
    if config.family != NativeDecoderFamily::Llama {
        return Err(RuntimeError::UnsupportedNativeDecoder {
            family: format!("{:?}", config.family),
        });
    }
    config.validate()?;
    let sequence_len = input.sequence_len;
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let head_dim = config.head_dim();
    let kv_width = config.key_value_width();
    validate_cpu_positive("llama_block", "sequence_len", sequence_len)?;
    validate_cpu_matrix_len(
        "llama_block",
        "hidden_states",
        input.hidden_states.len(),
        sequence_len,
        hidden_size,
    )?;

    let normalized = native_decoder_cpu_rms_norm(
        input.hidden_states,
        sequence_len,
        hidden_size,
        weights.input_layernorm,
        config.rms_norm_eps,
    )?;
    let mut query = native_decoder_cpu_linear(
        &normalized,
        sequence_len,
        hidden_size,
        weights.q_proj,
        hidden_size,
    )?;
    let mut key = native_decoder_cpu_linear(
        &normalized,
        sequence_len,
        hidden_size,
        weights.k_proj,
        kv_width,
    )?;
    let value = native_decoder_cpu_linear(
        &normalized,
        sequence_len,
        hidden_size,
        weights.v_proj,
        kv_width,
    )?;
    native_decoder_cpu_apply_llama_rope(
        &mut query,
        sequence_len,
        config.num_attention_heads,
        head_dim,
        input.position_start,
        config.rope_theta,
    )?;
    native_decoder_cpu_apply_llama_rope(
        &mut key,
        sequence_len,
        config.num_key_value_heads,
        head_dim,
        input.position_start,
        config.rope_theta,
    )?;
    let attention = native_decoder_cpu_causal_attention(
        &query,
        &key,
        &value,
        sequence_len,
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
    )?;
    let attention_projected = native_decoder_cpu_linear(
        &attention,
        sequence_len,
        hidden_size,
        weights.o_proj,
        hidden_size,
    )?;
    let attention_residual = add_same_shape(
        "llama_block",
        input.hidden_states,
        &attention_projected,
        "attention residual",
    )?;
    let mlp_normalized = native_decoder_cpu_rms_norm(
        &attention_residual,
        sequence_len,
        hidden_size,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
    )?;
    let gate = native_decoder_cpu_linear(
        &mlp_normalized,
        sequence_len,
        hidden_size,
        weights.gate_proj,
        intermediate_size,
    )?;
    let up = native_decoder_cpu_linear(
        &mlp_normalized,
        sequence_len,
        hidden_size,
        weights.up_proj,
        intermediate_size,
    )?;
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| native_decoder_cpu_silu(*gate) * up)
        .collect::<Vec<_>>();
    let mlp_projected = native_decoder_cpu_linear(
        &activated,
        sequence_len,
        intermediate_size,
        weights.down_proj,
        hidden_size,
    )?;
    let hidden_states = add_same_shape(
        "llama_block",
        &attention_residual,
        &mlp_projected,
        "mlp residual",
    )?;
    Ok(NativeDecoderCpuBlockOutput { hidden_states })
}

fn validate_cpu_matrix_len(
    operation: &'static str,
    name: &str,
    actual_len: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let expected_len = cpu_element_count(operation, name, rows, cols)?;
    if actual_len != expected_len {
        return Err(native_decoder_cpu_shape_error(
            operation,
            format!("{name} has {actual_len} elements, expected {expected_len}"),
        ));
    }
    Ok(())
}

fn cpu_element_count(
    operation: &'static str,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    rows.checked_mul(cols).ok_or_else(|| {
        native_decoder_cpu_shape_error(operation, format!("{name} element count overflow"))
    })
}

fn validate_cpu_vector_len(
    operation: &'static str,
    name: &str,
    actual_len: usize,
    expected_len: usize,
) -> Result<()> {
    if actual_len != expected_len {
        return Err(native_decoder_cpu_shape_error(
            operation,
            format!("{name} has {actual_len} elements, expected {expected_len}"),
        ));
    }
    Ok(())
}

fn validate_cpu_positive(operation: &'static str, name: &str, value: usize) -> Result<()> {
    if value == 0 {
        Err(native_decoder_cpu_shape_error(
            operation,
            format!("{name} must be positive"),
        ))
    } else {
        Ok(())
    }
}

fn native_decoder_cpu_shape_error(
    operation: &'static str,
    reason: impl Into<String>,
) -> RuntimeError {
    RuntimeError::Shape(format!("native decoder CPU {operation}: {}", reason.into()))
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn add_same_shape(
    operation: &'static str,
    left: &[f32],
    right: &[f32],
    name: &str,
) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        return Err(native_decoder_cpu_shape_error(
            operation,
            format!(
                "{name} inputs have different lengths: {} and {}",
                left.len(),
                right.len()
            ),
        ));
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left + right)
        .collect())
}

fn resolve_native_decoder_backend(backend: NativeDecoderBackend) -> Result<NativeDecoderBackend> {
    match backend {
        NativeDecoderBackend::Auto | NativeDecoderBackend::CpuReference => Ok(NativeDecoderBackend::CpuReference),
        NativeDecoderBackend::CpuThreaded => Ok(NativeDecoderBackend::CpuThreaded),
        NativeDecoderBackend::AppleCpuAccelerate | NativeDecoderBackend::Accelerated => {
            if apple_accelerate_available() {
                Ok(NativeDecoderBackend::AppleCpuAccelerate)
            } else {
                Ok(NativeDecoderBackend::CpuReference)
            }
        }
        NativeDecoderBackend::MetalWgpuLmHead => Err(RuntimeError::NativeDecoderBackendUnavailable {
            backend: "metal_wgpu_lm_head".to_string(),
            reason: "Metal/WGPU LM-head projection kernels are not implemented yet".to_string(),
        }),
        NativeDecoderBackend::MetalWgpuFullDecoder => {
            Err(RuntimeError::NativeDecoderBackendUnavailable {
                backend: "metal_wgpu_full_decoder".to_string(),
                reason: "Metal/WGPU full decoder kernels are not implemented yet".to_string(),
            })
        }
        NativeDecoderBackend::OrtCoreMl => Err(RuntimeError::NativeDecoderBackendUnavailable {
            backend: "ort_core_ml".to_string(),
            reason: "ORT CoreML execution provider applies to graph payloads, not the native decoder path yet".to_string(),
        }),
    }
}

fn native_decoder_generate_with_backend(
    weights: &NativeDecoderWeights,
    input_token_ids: &[i64],
    options: NativeDecoderRunOptions,
    backend: NativeDecoderBackend,
) -> Result<NativeDecoderGenerateOutput> {
    match backend {
        NativeDecoderBackend::CpuReference
        | NativeDecoderBackend::CpuThreaded
        | NativeDecoderBackend::AppleCpuAccelerate => {
            native_decoder_cpu_greedy_decode(weights, input_token_ids, options, backend)
        }
        NativeDecoderBackend::Auto
        | NativeDecoderBackend::Accelerated
        | NativeDecoderBackend::MetalWgpuLmHead
        | NativeDecoderBackend::MetalWgpuFullDecoder
        | NativeDecoderBackend::OrtCoreMl => Err(RuntimeError::NativeDecoderBackendUnavailable {
            backend: format!("{backend:?}"),
            reason: "backend selector did not resolve to an executable native decoder backend"
                .to_string(),
        }),
    }
}

fn apple_accelerate_available() -> bool {
    cfg!(all(target_os = "macos", feature = "apple-accelerate"))
}

fn load_native_decoder_weights(
    file: &RsmfFile,
    config: NativeDecoderConfig,
    options: &NativeDecoderWeightOptions,
) -> Result<NativeDecoderWeights> {
    if config.family != NativeDecoderFamily::Llama {
        return Err(RuntimeError::UnsupportedNativeDecoder {
            family: format!("{:?}", config.family),
        });
    }
    let hidden = usize_to_u64(config.hidden_size, "hidden_size")?;
    let intermediate = usize_to_u64(config.intermediate_size, "intermediate_size")?;
    let vocab = usize_to_u64(config.vocab_size, "vocab_size")?;
    let kv_width = usize_to_u64(config.key_value_width(), "kv projection width")?;
    let token_embedding = load_native_decoder_tensor_f32(
        file,
        options,
        "model.embed_tokens.weight",
        &[vocab, hidden],
    )?;
    let final_norm = load_native_decoder_tensor_f32(file, options, "model.norm.weight", &[hidden])?;
    let lm_head = if config.tie_word_embeddings {
        None
    } else {
        Some(load_native_decoder_tensor_f32(
            file,
            options,
            "lm_head.weight",
            &[vocab, hidden],
        )?)
    };
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        layers.push(NativeDecoderLayerWeights {
            input_layernorm: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.input_layernorm.weight"),
                &[hidden],
            )?,
            post_attention_layernorm: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            q_proj: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                &[hidden, hidden],
            )?,
            k_proj: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                &[kv_width, hidden],
            )?,
            v_proj: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.self_attn.v_proj.weight"),
                &[kv_width, hidden],
            )?,
            o_proj: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.self_attn.o_proj.weight"),
                &[hidden, hidden],
            )?,
            gate_proj: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.mlp.gate_proj.weight"),
                &[intermediate, hidden],
            )?,
            up_proj: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.mlp.up_proj.weight"),
                &[intermediate, hidden],
            )?,
            down_proj: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.mlp.down_proj.weight"),
                &[hidden, intermediate],
            )?,
        });
    }
    Ok(NativeDecoderWeights {
        config,
        token_embedding,
        final_norm,
        lm_head,
        layers,
    })
}

fn load_native_decoder_tensor_f32(
    file: &RsmfFile,
    options: &NativeDecoderWeightOptions,
    tensor_name: &str,
    expected_shape: &[u64],
) -> Result<Vec<f32>> {
    let view = native_decoder_weight_view(file, options, tensor_name)?;
    if view.shape() != expected_shape {
        return Err(RuntimeError::NativeDecoderTensorShapeMismatch {
            tensor_name: tensor_name.to_string(),
            expected_shape: format_shape(expected_shape),
            actual_shape: format_shape(view.shape()),
        });
    }
    validate_native_decoder_weight_dtype(view.descriptor)?;
    if view.layout != LayoutTag::RowMajor {
        return Err(RuntimeError::NativeDecoderTensorUnsupported {
            tensor_name: tensor_name.to_string(),
            reason: format!(
                "only row-major weights are supported, got {:?}",
                view.layout
            ),
        });
    }
    let values =
        view.decode_f32()
            .map_err(|error| RuntimeError::NativeDecoderTensorUnsupported {
                tensor_name: tensor_name.to_string(),
                reason: error.to_string(),
            })?;
    let expected_len = shape_element_count(expected_shape)?;
    if values.len() != expected_len {
        return Err(RuntimeError::NativeDecoderTensorUnsupported {
            tensor_name: tensor_name.to_string(),
            reason: format!("decoded {} elements, expected {expected_len}", values.len()),
        });
    }
    Ok(values)
}

fn native_decoder_weight_view<'a>(
    file: &'a RsmfFile,
    options: &NativeDecoderWeightOptions,
    tensor_name: &str,
) -> Result<TensorView<'a>> {
    let result = if let Some(variant_idx) = options.tensor_variants.get(tensor_name) {
        file.tensor_view_variant(tensor_name, *variant_idx)
    } else {
        file.tensor_view(tensor_name)
    };
    result.map_err(|error| match error {
        RsmfError::NotFound { what } if what == format!("tensor {tensor_name}") => {
            RuntimeError::NativeDecoderTensorMissing {
                tensor_name: tensor_name.to_string(),
            }
        }
        RsmfError::NotFound { what } => RuntimeError::NativeDecoderTensorUnsupported {
            tensor_name: tensor_name.to_string(),
            reason: what,
        },
        other => RuntimeError::Core(other),
    })
}

fn shape_element_count(shape: &[u64]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim).map_err(|_| {
            RuntimeError::Shape(format!(
                "native decoder dimension {dim} cannot convert to usize"
            ))
        })?;
        count
            .checked_mul(dim)
            .ok_or_else(|| RuntimeError::Shape("native decoder element count overflow".to_string()))
    })
}

fn native_decoder_cpu_greedy_decode(
    weights: &NativeDecoderWeights,
    input_token_ids: &[i64],
    options: NativeDecoderRunOptions,
    backend: NativeDecoderBackend,
) -> Result<NativeDecoderGenerateOutput> {
    validate_native_decoder_sampling_options(&options.sampling)?;
    validate_native_decoder_performance_options(&options.performance)?;
    if input_token_ids.is_empty() {
        return Err(RuntimeError::NativeDecoderPromptEmpty);
    }
    for &token_id in input_token_ids {
        validate_native_decoder_token_id(token_id, weights.config.vocab_size)?;
    }
    let mut token_ids = input_token_ids.to_vec();
    if options.max_new_tokens == 0 {
        return Ok(NativeDecoderGenerateOutput {
            token_ids,
            generated_token_ids: Vec::new(),
            logits: Vec::new(),
            prompt_logits: Vec::new(),
            backend,
        });
    }

    let mut sampler_rng =
        NativeDecoderSamplerRng::new(options.sampling.seed.unwrap_or(0x9E37_79B9_7F4A_7C15));
    if options.min_new_tokens > options.max_new_tokens {
        return Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: "min_new_tokens must be less than or equal to max_new_tokens".to_string(),
        });
    }
    let stop_token_ids = if !options.stop_token_ids.is_empty() {
        options.stop_token_ids.clone()
    } else if !options.eos_token_ids.is_empty() {
        options.eos_token_ids.clone()
    } else {
        weights.config.eos_token_ids.clone()
    };
    let mut cache =
        native_decoder_kv_cache_with_performance(&weights.config, &options.performance)?;
    let mut last_step = None;
    let mut prompt_logits = Vec::new();
    let prefill_chunk_size = options
        .performance
        .prefill_chunk_size
        .unwrap_or(input_token_ids.len().max(1));
    for chunk in input_token_ids.chunks(prefill_chunk_size) {
        for &token_id in chunk {
            let step = native_decoder_cpu_step(
                weights,
                &mut cache,
                token_id,
                backend,
                &options.performance,
            )?;
            if options.return_prompt_logits {
                prompt_logits.push(step.logits.clone());
            }
            last_step = Some(step);
        }
    }

    let mut generated_token_ids = Vec::with_capacity(options.max_new_tokens);
    let mut logits = Vec::with_capacity(options.max_new_tokens);
    for step_index in 0..options.max_new_tokens {
        let step = last_step
            .take()
            .ok_or(RuntimeError::NativeDecoderPromptEmpty)?;
        let adjusted_logits =
            apply_native_decoder_repetition_penalty(&step.logits, &token_ids, &options.sampling)?;
        let next_token_id =
            select_native_decoder_token(&adjusted_logits, &options.sampling, &mut sampler_rng)?;
        logits.push(step.logits);
        generated_token_ids.push(next_token_id);
        token_ids.push(next_token_id);
        if step_index + 1 >= options.min_new_tokens && stop_token_ids.contains(&next_token_id) {
            break;
        }
        if step_index + 1 < options.max_new_tokens {
            last_step = Some(native_decoder_cpu_step(
                weights,
                &mut cache,
                next_token_id,
                backend,
                &options.performance,
            )?);
        }
    }

    Ok(NativeDecoderGenerateOutput {
        token_ids,
        generated_token_ids,
        logits,
        prompt_logits,
        backend,
    })
}

fn native_decoder_check_reference_logits(
    weights: &NativeDecoderWeights,
    check: NativeDecoderReferenceLogitCheck,
    backend: NativeDecoderBackend,
) -> Result<NativeDecoderReferenceLogitReport> {
    validate_native_decoder_performance_options(&check.performance)?;
    if check.input_token_ids.is_empty() {
        return Err(RuntimeError::NativeDecoderPromptEmpty);
    }
    if check.input_token_ids.len() != check.expected_logits.len() {
        return Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: format!(
                "expected_logits has {} rows, expected {}",
                check.expected_logits.len(),
                check.input_token_ids.len()
            ),
        });
    }
    if !check.tolerance_abs.is_finite() || check.tolerance_abs < 0.0 {
        return Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: "tolerance_abs must be finite and non-negative".to_string(),
        });
    }
    let mut cache = native_decoder_kv_cache_with_performance(&weights.config, &check.performance)?;
    let mut compared_values = 0usize;
    let mut max_abs_diff = 0.0f32;
    for (token_id, expected_logits) in check
        .input_token_ids
        .iter()
        .copied()
        .zip(check.expected_logits.iter())
    {
        if expected_logits.len() != weights.config.vocab_size {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: format!(
                    "expected logits row has {} values, expected vocab size {}",
                    expected_logits.len(),
                    weights.config.vocab_size
                ),
            });
        }
        let output =
            native_decoder_cpu_step(weights, &mut cache, token_id, backend, &check.performance)?;
        for (actual, expected) in output.logits.iter().zip(expected_logits.iter()) {
            let diff = (actual - expected).abs();
            max_abs_diff = max_abs_diff.max(diff);
            compared_values += 1;
        }
    }
    if max_abs_diff > check.tolerance_abs {
        return Err(RuntimeError::NativeDecoderReferenceLogitsMismatch {
            max_abs_diff,
            tolerance_abs: check.tolerance_abs,
        });
    }
    Ok(NativeDecoderReferenceLogitReport {
        compared_logits: check.expected_logits.len(),
        compared_values,
        max_abs_diff,
        tolerance_abs: check.tolerance_abs,
        backend,
    })
}

fn native_decoder_kv_cache_with_performance(
    config: &NativeDecoderConfig,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<NativeDecoderKvCache> {
    if let Some(page_size) = performance.kv_cache_page_size_tokens {
        NativeDecoderKvCache::new_paged(config, page_size)
    } else {
        Ok(NativeDecoderKvCache::new(config))
    }
}

fn append_native_decoder_layer_cache(
    cache: &mut NativeDecoderLayerKvCache,
    key: &[f32],
    value: &[f32],
    position: usize,
    kv_width: usize,
    page_size_tokens: Option<usize>,
) -> Result<()> {
    validate_cpu_vector_len("kv_cache", "key", key.len(), kv_width)?;
    validate_cpu_vector_len("kv_cache", "value", value.len(), kv_width)?;
    if let Some(page_size) = page_size_tokens {
        let page_width = cpu_element_count("kv_cache", "page", page_size, kv_width)?;
        if position % page_size == 0 {
            cache.key_pages.push(Vec::with_capacity(page_width));
            cache.value_pages.push(Vec::with_capacity(page_width));
        }
        let page_index = position / page_size;
        let key_page = cache.key_pages.get_mut(page_index).ok_or_else(|| {
            native_decoder_cpu_shape_error("kv_cache", format!("missing key page {page_index}"))
        })?;
        let value_page = cache.value_pages.get_mut(page_index).ok_or_else(|| {
            native_decoder_cpu_shape_error("kv_cache", format!("missing value page {page_index}"))
        })?;
        key_page.extend_from_slice(key);
        value_page.extend_from_slice(value);
    } else {
        cache.keys.extend_from_slice(key);
        cache.values.extend_from_slice(value);
    }
    Ok(())
}

fn validate_native_decoder_sampling_options(options: &NativeDecoderSamplingOptions) -> Result<()> {
    if let Some(temperature) = options.temperature {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(RuntimeError::NativeDecoderSamplingInvalid {
                reason: "temperature must be positive and finite".to_string(),
            });
        }
    }
    if matches!(options.top_k, Some(0)) {
        return Err(RuntimeError::NativeDecoderSamplingInvalid {
            reason: "top_k must be greater than zero".to_string(),
        });
    }
    if let Some(top_p) = options.top_p {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(RuntimeError::NativeDecoderSamplingInvalid {
                reason: "top_p must be in (0, 1]".to_string(),
            });
        }
    }
    if let Some(repetition_penalty) = options.repetition_penalty {
        if !repetition_penalty.is_finite() || repetition_penalty < 1.0 {
            return Err(RuntimeError::NativeDecoderSamplingInvalid {
                reason: "repetition_penalty must be finite and at least 1.0".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_native_decoder_performance_options(
    options: &NativeDecoderPerformanceOptions,
) -> Result<()> {
    if matches!(options.kv_cache_page_size_tokens, Some(0)) {
        return Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: "kv_cache_page_size_tokens must be positive".to_string(),
        });
    }
    if matches!(options.cpu_threads, Some(0)) {
        return Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: "cpu_threads must be positive".to_string(),
        });
    }
    if matches!(options.prefill_chunk_size, Some(0)) {
        return Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: "prefill_chunk_size must be positive".to_string(),
        });
    }
    Ok(())
}

fn apply_native_decoder_repetition_penalty(
    logits: &[f32],
    token_ids: &[i64],
    sampling: &NativeDecoderSamplingOptions,
) -> Result<Vec<f32>> {
    let Some(penalty) = sampling.repetition_penalty else {
        return Ok(logits.to_vec());
    };
    let mut adjusted = logits.to_vec();
    for &token_id in token_ids {
        let token_index =
            usize::try_from(token_id).map_err(|_| RuntimeError::NativeDecoderTokenOutOfRange {
                token_id,
                vocab_size: logits.len(),
            })?;
        if let Some(logit) = adjusted.get_mut(token_index) {
            if *logit < 0.0 {
                *logit *= penalty;
            } else {
                *logit /= penalty;
            }
        }
    }
    Ok(adjusted)
}

fn select_native_decoder_token(
    logits: &[f32],
    sampling: &NativeDecoderSamplingOptions,
    rng: &mut NativeDecoderSamplerRng,
) -> Result<i64> {
    if sampling.temperature.is_none() {
        return greedy_argmax_token(logits);
    }
    let temperature = sampling
        .temperature
        .ok_or_else(|| native_decoder_cpu_shape_error("sample", "temperature missing"))?;
    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, logit)| (token_id, logit / temperature))
        .collect::<Vec<_>>();
    candidates.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    if let Some(top_k) = sampling.top_k {
        candidates.truncate(top_k.min(candidates.len()));
    }
    if candidates.is_empty() {
        return Err(native_decoder_cpu_shape_error(
            "sample",
            "candidate set must not be empty",
        ));
    }

    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = candidates
        .into_iter()
        .map(|(token_id, logit)| (token_id, (logit - max_logit).exp()))
        .collect::<Vec<_>>();
    let total = probabilities
        .iter()
        .map(|(_, probability)| *probability)
        .sum::<f32>();
    if total <= 0.0 || !total.is_finite() {
        return greedy_argmax_token(logits);
    }
    for (_, probability) in &mut probabilities {
        *probability /= total;
    }
    if let Some(top_p) = sampling.top_p {
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for (_, probability) in &probabilities {
            cumulative += *probability;
            keep += 1;
            if cumulative >= top_p {
                break;
            }
        }
        probabilities.truncate(keep.max(1));
        let renormalized_total = probabilities
            .iter()
            .map(|(_, probability)| *probability)
            .sum::<f32>();
        if renormalized_total > 0.0 {
            for (_, probability) in &mut probabilities {
                *probability /= renormalized_total;
            }
        }
    }

    let draw = rng.next_unit_f32();
    let mut cumulative = 0.0f32;
    for (token_id, probability) in &probabilities {
        cumulative += *probability;
        if draw <= cumulative {
            return i64::try_from(*token_id)
                .map_err(|_| native_decoder_cpu_shape_error("sample", "token id overflow"));
        }
    }
    probabilities
        .last()
        .map(|(token_id, _)| *token_id)
        .ok_or_else(|| native_decoder_cpu_shape_error("sample", "candidate set must not be empty"))
        .and_then(|token_id| {
            i64::try_from(token_id)
                .map_err(|_| native_decoder_cpu_shape_error("sample", "token id overflow"))
        })
}

struct NativeDecoderSamplerRng {
    state: u64,
}

impl NativeDecoderSamplerRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    fn next_unit_f32(&mut self) -> f32 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        let value = value.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((value >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

fn native_decoder_cpu_step(
    weights: &NativeDecoderWeights,
    cache: &mut NativeDecoderKvCache,
    token_id: i64,
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<NativeDecoderStepOutput> {
    let token_index = validate_native_decoder_token_id(token_id, weights.config.vocab_size)?;
    if cache.layers.len() != weights.config.num_hidden_layers {
        return Err(native_decoder_cpu_shape_error(
            "step",
            format!(
                "cache has {} layers, expected {}",
                cache.layers.len(),
                weights.config.num_hidden_layers
            ),
        ));
    }
    if cache.position >= weights.config.max_position_embeddings {
        return Err(native_decoder_cpu_shape_error(
            "step",
            format!(
                "position {} exceeds max_position_embeddings {}",
                cache.position, weights.config.max_position_embeddings
            ),
        ));
    }
    let hidden_size = weights.config.hidden_size;
    let embed_start = token_index
        .checked_mul(hidden_size)
        .ok_or_else(|| native_decoder_cpu_shape_error("step", "embedding offset overflow"))?;
    let embed_end = embed_start
        .checked_add(hidden_size)
        .ok_or_else(|| native_decoder_cpu_shape_error("step", "embedding end overflow"))?;
    let mut hidden_states = weights
        .token_embedding
        .get(embed_start..embed_end)
        .ok_or_else(|| {
            native_decoder_cpu_shape_error(
                "step",
                format!("token embedding missing row {token_index}"),
            )
        })?
        .to_vec();

    let mut layer_updates = Vec::with_capacity(weights.layers.len());
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &cache.layers[layer_idx];
        let step = native_decoder_cpu_llama_cached_step(
            &weights.config,
            &hidden_states,
            cache.position,
            layer_cache,
            cache.page_size_tokens,
            NativeDecoderLinearBackend {
                backend,
                performance,
            },
            layer_weights.as_cpu(),
        )?;
        hidden_states = step.hidden_states;
        layer_updates.push((step.key, step.value));
    }
    for (layer_cache, (key, value)) in cache.layers.iter_mut().zip(layer_updates) {
        append_native_decoder_layer_cache(
            layer_cache,
            &key,
            &value,
            cache.position,
            cache.kv_width,
            cache.page_size_tokens,
        )?;
    }
    cache.position += 1;

    let normalized = native_decoder_cpu_rms_norm(
        &hidden_states,
        1,
        hidden_size,
        &weights.final_norm,
        weights.config.rms_norm_eps,
    )?;
    let lm_head = weights.lm_head.as_ref().unwrap_or(&weights.token_embedding);
    let logits = native_decoder_cpu_logits(
        &normalized,
        hidden_size,
        lm_head,
        weights.config.vocab_size,
        backend,
        performance,
    )?;
    let next_token_id = greedy_argmax_token(&logits)?;
    Ok(NativeDecoderStepOutput {
        logits,
        next_token_id,
    })
}

struct NativeDecoderCpuCachedBlockOutput {
    hidden_states: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

#[derive(Clone, Copy)]
struct NativeDecoderLinearBackend<'a> {
    backend: NativeDecoderBackend,
    performance: &'a NativeDecoderPerformanceOptions,
}

fn native_decoder_cpu_llama_cached_step(
    config: &NativeDecoderConfig,
    hidden_states: &[f32],
    position: usize,
    cache: &NativeDecoderLayerKvCache,
    page_size_tokens: Option<usize>,
    linear_backend: NativeDecoderLinearBackend<'_>,
    weights: NativeDecoderCpuLayerWeights<'_>,
) -> Result<NativeDecoderCpuCachedBlockOutput> {
    if config.family != NativeDecoderFamily::Llama {
        return Err(RuntimeError::UnsupportedNativeDecoder {
            family: format!("{:?}", config.family),
        });
    }
    config.validate()?;
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let head_dim = config.head_dim();
    let kv_width = config.key_value_width();
    validate_cpu_vector_len(
        "llama_cached_step",
        "hidden_states",
        hidden_states.len(),
        hidden_size,
    )?;
    validate_native_decoder_layer_cache(cache, position, kv_width, page_size_tokens)?;

    let normalized = native_decoder_cpu_rms_norm(
        hidden_states,
        1,
        hidden_size,
        weights.input_layernorm,
        config.rms_norm_eps,
    )?;
    let mut query = native_decoder_backend_linear(
        &normalized,
        1,
        hidden_size,
        weights.q_proj,
        hidden_size,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let mut key = native_decoder_backend_linear(
        &normalized,
        1,
        hidden_size,
        weights.k_proj,
        kv_width,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let value = native_decoder_backend_linear(
        &normalized,
        1,
        hidden_size,
        weights.v_proj,
        kv_width,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    native_decoder_cpu_apply_llama_rope(
        &mut query,
        1,
        config.num_attention_heads,
        head_dim,
        position,
        config.rope_theta,
    )?;
    native_decoder_cpu_apply_llama_rope(
        &mut key,
        1,
        config.num_key_value_heads,
        head_dim,
        position,
        config.rope_theta,
    )?;

    let attention = native_decoder_cpu_layer_cached_attention(NativeDecoderLayerAttentionInput {
        query: &query,
        cache,
        current_key: &key,
        current_value: &value,
        page_size_tokens,
        cache_len: position + 1,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim,
    })?;
    let attention_projected = native_decoder_backend_linear(
        &attention,
        1,
        hidden_size,
        weights.o_proj,
        hidden_size,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let attention_residual = add_same_shape(
        "llama_cached_step",
        hidden_states,
        &attention_projected,
        "attention residual",
    )?;
    let mlp_normalized = native_decoder_cpu_rms_norm(
        &attention_residual,
        1,
        hidden_size,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
    )?;
    let gate = native_decoder_backend_linear(
        &mlp_normalized,
        1,
        hidden_size,
        weights.gate_proj,
        intermediate_size,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let up = native_decoder_backend_linear(
        &mlp_normalized,
        1,
        hidden_size,
        weights.up_proj,
        intermediate_size,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| native_decoder_cpu_silu(*gate) * up)
        .collect::<Vec<_>>();
    let mlp_projected = native_decoder_backend_linear(
        &activated,
        1,
        intermediate_size,
        weights.down_proj,
        hidden_size,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let hidden_states = add_same_shape(
        "llama_cached_step",
        &attention_residual,
        &mlp_projected,
        "mlp residual",
    )?;

    Ok(NativeDecoderCpuCachedBlockOutput {
        hidden_states,
        key,
        value,
    })
}

fn validate_native_decoder_layer_cache(
    cache: &NativeDecoderLayerKvCache,
    position: usize,
    kv_width: usize,
    page_size_tokens: Option<usize>,
) -> Result<()> {
    let expected_cache_len = cpu_element_count("llama_cached_step", "cache", position, kv_width)?;
    if let Some(page_size) = page_size_tokens {
        let expected_pages = position.div_ceil(page_size);
        if cache.key_pages.len() != expected_pages || cache.value_pages.len() != expected_pages {
            return Err(native_decoder_cpu_shape_error(
                "llama_cached_step",
                format!(
                    "paged cache has {}/{} key/value pages, expected {expected_pages}",
                    cache.key_pages.len(),
                    cache.value_pages.len()
                ),
            ));
        }
        let mut key_len = 0usize;
        let mut value_len = 0usize;
        for page in &cache.key_pages {
            key_len = key_len.checked_add(page.len()).ok_or_else(|| {
                native_decoder_cpu_shape_error("llama_cached_step", "key page length overflow")
            })?;
        }
        for page in &cache.value_pages {
            value_len = value_len.checked_add(page.len()).ok_or_else(|| {
                native_decoder_cpu_shape_error("llama_cached_step", "value page length overflow")
            })?;
        }
        validate_cpu_vector_len(
            "llama_cached_step",
            "key_pages",
            key_len,
            expected_cache_len,
        )?;
        validate_cpu_vector_len(
            "llama_cached_step",
            "value_pages",
            value_len,
            expected_cache_len,
        )?;
        Ok(())
    } else {
        validate_cpu_vector_len(
            "llama_cached_step",
            "key_cache",
            cache.keys.len(),
            expected_cache_len,
        )?;
        validate_cpu_vector_len(
            "llama_cached_step",
            "value_cache",
            cache.values.len(),
            expected_cache_len,
        )
    }
}

struct NativeDecoderLayerAttentionInput<'a> {
    query: &'a [f32],
    cache: &'a NativeDecoderLayerKvCache,
    current_key: &'a [f32],
    current_value: &'a [f32],
    page_size_tokens: Option<usize>,
    cache_len: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

fn native_decoder_cpu_layer_cached_attention(
    input: NativeDecoderLayerAttentionInput<'_>,
) -> Result<Vec<f32>> {
    validate_cpu_positive("layer_cached_attention", "cache_len", input.cache_len)?;
    validate_cpu_positive(
        "layer_cached_attention",
        "num_attention_heads",
        input.num_attention_heads,
    )?;
    validate_cpu_positive(
        "layer_cached_attention",
        "num_key_value_heads",
        input.num_key_value_heads,
    )?;
    validate_cpu_positive("layer_cached_attention", "head_dim", input.head_dim)?;
    if input.num_attention_heads % input.num_key_value_heads != 0 {
        return Err(native_decoder_cpu_shape_error(
            "layer_cached_attention",
            "num_attention_heads must be divisible by num_key_value_heads",
        ));
    }
    let query_width = cpu_element_count(
        "layer_cached_attention",
        "query width",
        input.num_attention_heads,
        input.head_dim,
    )?;
    let kv_width = cpu_element_count(
        "layer_cached_attention",
        "key/value width",
        input.num_key_value_heads,
        input.head_dim,
    )?;
    validate_cpu_vector_len(
        "layer_cached_attention",
        "query",
        input.query.len(),
        query_width,
    )?;
    validate_cpu_vector_len(
        "layer_cached_attention",
        "current_key",
        input.current_key.len(),
        kv_width,
    )?;
    validate_cpu_vector_len(
        "layer_cached_attention",
        "current_value",
        input.current_value.len(),
        kv_width,
    )?;

    let groups = input.num_attention_heads / input.num_key_value_heads;
    let scale = 1.0 / (input.head_dim as f32).sqrt();
    let mut output = vec![0.0f32; query_width];
    for head in 0..input.num_attention_heads {
        let kv_head = head / groups;
        let query_offset = head * input.head_dim;
        let mut scores = vec![0.0f32; input.cache_len];
        for (key_token, score) in scores.iter_mut().enumerate() {
            let key_values = native_decoder_cache_kv_slice(NativeDecoderCacheSliceRequest {
                cache: input.cache,
                current: input.current_key,
                page_size_tokens: input.page_size_tokens,
                token: key_token,
                current_position: input.cache_len - 1,
                kv_width,
                kv_head,
                head_dim: input.head_dim,
                key: true,
            })?;
            *score = dot_product(
                &input.query[query_offset..query_offset + input.head_dim],
                key_values,
            ) * scale;
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weight_sum = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            weight_sum += *score;
        }
        let output_offset = head * input.head_dim;
        for (key_token, score) in scores.iter().enumerate() {
            let attention_weight = *score / weight_sum;
            let value_values = native_decoder_cache_kv_slice(NativeDecoderCacheSliceRequest {
                cache: input.cache,
                current: input.current_value,
                page_size_tokens: input.page_size_tokens,
                token: key_token,
                current_position: input.cache_len - 1,
                kv_width,
                kv_head,
                head_dim: input.head_dim,
                key: false,
            })?;
            for dim in 0..input.head_dim {
                output[output_offset + dim] += attention_weight * value_values[dim];
            }
        }
    }
    Ok(output)
}

struct NativeDecoderCacheSliceRequest<'a> {
    cache: &'a NativeDecoderLayerKvCache,
    current: &'a [f32],
    page_size_tokens: Option<usize>,
    token: usize,
    current_position: usize,
    kv_width: usize,
    kv_head: usize,
    head_dim: usize,
    key: bool,
}

fn native_decoder_cache_kv_slice<'a>(
    request: NativeDecoderCacheSliceRequest<'a>,
) -> Result<&'a [f32]> {
    let offset_in_token = request
        .kv_head
        .checked_mul(request.head_dim)
        .ok_or_else(|| {
            native_decoder_cpu_shape_error("layer_cached_attention", "head offset overflow")
        })?;
    if request.token == request.current_position {
        return request
            .current
            .get(offset_in_token..offset_in_token + request.head_dim)
            .ok_or_else(|| {
                native_decoder_cpu_shape_error(
                    "layer_cached_attention",
                    "current kv slice out of range",
                )
            });
    }
    if let Some(page_size) = request.page_size_tokens {
        let page_index = request.token / page_size;
        let page_token = request.token % page_size;
        let page = if request.key {
            request.cache.key_pages.get(page_index)
        } else {
            request.cache.value_pages.get(page_index)
        }
        .ok_or_else(|| {
            native_decoder_cpu_shape_error(
                "layer_cached_attention",
                format!("missing cache page {page_index}"),
            )
        })?;
        let base = page_token
            .checked_mul(request.kv_width)
            .and_then(|base| base.checked_add(offset_in_token))
            .ok_or_else(|| {
                native_decoder_cpu_shape_error("layer_cached_attention", "page kv offset overflow")
            })?;
        page.get(base..base + request.head_dim).ok_or_else(|| {
            native_decoder_cpu_shape_error("layer_cached_attention", "page kv slice out of range")
        })
    } else {
        let base = request
            .token
            .checked_mul(request.kv_width)
            .and_then(|base| base.checked_add(offset_in_token))
            .ok_or_else(|| {
                native_decoder_cpu_shape_error("layer_cached_attention", "flat kv offset overflow")
            })?;
        let values = if request.key {
            &request.cache.keys
        } else {
            &request.cache.values
        };
        values.get(base..base + request.head_dim).ok_or_else(|| {
            native_decoder_cpu_shape_error("layer_cached_attention", "flat kv slice out of range")
        })
    }
}

fn native_decoder_cpu_logits(
    normalized: &[f32],
    hidden_size: usize,
    lm_head: &[f32],
    vocab_size: usize,
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<Vec<f32>> {
    match backend {
        NativeDecoderBackend::AppleCpuAccelerate => native_decoder_backend_linear(
            normalized,
            1,
            hidden_size,
            lm_head,
            vocab_size,
            backend,
            performance,
        ),
        NativeDecoderBackend::CpuThreaded => native_decoder_cpu_linear_threaded(
            normalized,
            1,
            hidden_size,
            lm_head,
            vocab_size,
            performance.cpu_threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
            }),
        ),
        NativeDecoderBackend::CpuReference
        | NativeDecoderBackend::Auto
        | NativeDecoderBackend::Accelerated
        | NativeDecoderBackend::MetalWgpuLmHead
        | NativeDecoderBackend::MetalWgpuFullDecoder
        | NativeDecoderBackend::OrtCoreMl => {
            native_decoder_cpu_linear(normalized, 1, hidden_size, lm_head, vocab_size)
        }
    }
}

fn native_decoder_backend_linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<Vec<f32>> {
    match backend {
        NativeDecoderBackend::AppleCpuAccelerate => {
            native_decoder_apple_accelerate_linear(input, rows, in_features, weight, out_features)
        }
        NativeDecoderBackend::CpuThreaded if rows > 1 => native_decoder_cpu_linear_threaded(
            input,
            rows,
            in_features,
            weight,
            out_features,
            performance.cpu_threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
            }),
        ),
        _ => native_decoder_cpu_linear(input, rows, in_features, weight, out_features),
    }
}

#[cfg(all(target_os = "macos", feature = "apple-accelerate"))]
fn native_decoder_apple_accelerate_linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    validate_cpu_matrix_len(
        "linear_apple_accelerate",
        "input",
        input.len(),
        rows,
        in_features,
    )?;
    validate_cpu_matrix_len(
        "linear_apple_accelerate",
        "weight",
        weight.len(),
        out_features,
        in_features,
    )?;
    let output_len = cpu_element_count("linear_apple_accelerate", "output", rows, out_features)?;
    let mut output = vec![0.0f32; output_len];
    let rows_i32 = i32::try_from(rows).map_err(|_| {
        native_decoder_cpu_shape_error("linear_apple_accelerate", "rows exceed i32")
    })?;
    let in_i32 = i32::try_from(in_features).map_err(|_| {
        native_decoder_cpu_shape_error("linear_apple_accelerate", "in_features exceed i32")
    })?;
    let out_i32 = i32::try_from(out_features).map_err(|_| {
        native_decoder_cpu_shape_error("linear_apple_accelerate", "out_features exceed i32")
    })?;
    if rows == 1 {
        // SAFETY: All pointers are derived from validated Rust slices. Matrix
        // dimensions and strides are checked above and converted to the CBLAS
        // `i32` ABI before the call. Output is uniquely borrowed and sized for
        // `out_features` elements.
        unsafe {
            apple_accelerate::cblas_sgemv(
                apple_accelerate::CBLAS_ROW_MAJOR,
                apple_accelerate::CBLAS_NO_TRANS,
                out_i32,
                in_i32,
                1.0,
                weight.as_ptr(),
                in_i32,
                input.as_ptr(),
                1,
                0.0,
                output.as_mut_ptr(),
                1,
            );
        }
    } else {
        // SAFETY: All pointers are derived from validated Rust slices. A is
        // row-major `[rows, in_features]`, B is row-major
        // `[out_features, in_features]` and is passed transposed, and C is
        // row-major `[rows, out_features]` with non-overlapping output storage.
        unsafe {
            apple_accelerate::cblas_sgemm(
                apple_accelerate::CBLAS_ROW_MAJOR,
                apple_accelerate::CBLAS_NO_TRANS,
                apple_accelerate::CBLAS_TRANS,
                rows_i32,
                out_i32,
                in_i32,
                1.0,
                input.as_ptr(),
                in_i32,
                weight.as_ptr(),
                in_i32,
                0.0,
                output.as_mut_ptr(),
                out_i32,
            );
        }
    }
    Ok(output)
}

#[cfg(not(all(target_os = "macos", feature = "apple-accelerate")))]
fn native_decoder_apple_accelerate_linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    native_decoder_cpu_linear(input, rows, in_features, weight, out_features)
}

#[cfg(all(target_os = "macos", feature = "apple-accelerate"))]
mod apple_accelerate {
    pub const CBLAS_ROW_MAJOR: i32 = 101;
    pub const CBLAS_NO_TRANS: i32 = 111;
    pub const CBLAS_TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemv(
            layout: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            x: *const f32,
            inc_x: i32,
            beta: f32,
            y: *mut f32,
            inc_y: i32,
        );

        pub fn cblas_sgemm(
            layout: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

fn native_decoder_cpu_linear_threaded(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
    requested_threads: usize,
) -> Result<Vec<f32>> {
    validate_cpu_matrix_len("linear_threaded", "input", input.len(), rows, in_features)?;
    validate_cpu_matrix_len(
        "linear_threaded",
        "weight",
        weight.len(),
        out_features,
        in_features,
    )?;
    validate_cpu_positive("linear_threaded", "requested_threads", requested_threads)?;
    let output_len = cpu_element_count("linear_threaded", "output", rows, out_features)?;
    if requested_threads == 1 || out_features <= 1 {
        return native_decoder_cpu_linear(input, rows, in_features, weight, out_features);
    }
    let thread_count = requested_threads.min(out_features);
    let chunk_size = out_features.div_ceil(thread_count);
    let mut chunks = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..out_features).step_by(chunk_size) {
            let end = start.saturating_add(chunk_size).min(out_features);
            handles.push(scope.spawn(move || {
                let mut chunk = vec![0.0f32; rows * (end - start)];
                for row in 0..rows {
                    for out_col in start..end {
                        let mut sum = 0.0f32;
                        for in_col in 0..in_features {
                            sum += input[row * in_features + in_col]
                                * weight[out_col * in_features + in_col];
                        }
                        chunk[row * (end - start) + (out_col - start)] = sum;
                    }
                }
                (start, end, chunk)
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join())
            .collect::<std::result::Result<Vec<_>, _>>()
    })
    .map_err(|_| RuntimeError::Shape("native decoder CPU worker panicked".to_string()))?;
    chunks.sort_by_key(|(start, _, _)| *start);
    let mut output = vec![0.0f32; output_len];
    for (start, end, chunk) in chunks {
        let width = end - start;
        for row in 0..rows {
            let output_start = row * out_features + start;
            let chunk_start = row * width;
            output[output_start..output_start + width]
                .copy_from_slice(&chunk[chunk_start..chunk_start + width]);
        }
    }
    Ok(output)
}

fn validate_native_decoder_token_id(token_id: i64, vocab_size: usize) -> Result<usize> {
    let token_index =
        usize::try_from(token_id).map_err(|_| RuntimeError::NativeDecoderTokenOutOfRange {
            token_id,
            vocab_size,
        })?;
    if token_index >= vocab_size {
        Err(RuntimeError::NativeDecoderTokenOutOfRange {
            token_id,
            vocab_size,
        })
    } else {
        Ok(token_index)
    }
}

fn greedy_argmax_token(logits: &[f32]) -> Result<i64> {
    let (index, _) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| native_decoder_cpu_shape_error("greedy", "logits must not be empty"))?;
    i64::try_from(index).map_err(|_| native_decoder_cpu_shape_error("greedy", "token id overflow"))
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

    /// Resolve and validate the native decoder model contract for this RSMF
    /// file.
    ///
    /// R4.1 only validates assets, config, tensor names, tensor shapes, and
    /// weight dtypes. It does not execute a decoder block or parse tokenizer
    /// internals.
    pub fn native_decoder_contract(&self) -> Result<NativeDecoderContract> {
        NativeDecoderContract::from_file(&self.file)
    }

    /// Load native decoder weights from canonical RSMF tensor variants.
    ///
    /// The contract is validated first, then selected tensor variants are
    /// decoded to owned f32 buffers for the native runtime.
    pub fn native_decoder_weights(&self) -> Result<NativeDecoderWeights> {
        self.native_decoder_weights_with_options(&NativeDecoderWeightOptions::default())
    }

    /// Load native decoder weights from selected RSMF tensor variants.
    ///
    /// Missing entries in `options.tensor_variants` load canonical variants.
    /// Selected variants must belong to the requested tensor and decode to the
    /// expected row-major f32 element count.
    pub fn native_decoder_weights_with_options(
        &self,
        options: &NativeDecoderWeightOptions,
    ) -> Result<NativeDecoderWeights> {
        let contract = self.native_decoder_contract()?;
        load_native_decoder_weights(&self.file, contract.config, options)
    }

    /// Load a resident native decoder session with decoded weights and
    /// tokenizer.
    ///
    /// Use this when issuing multiple generation calls so weight decoding is
    /// paid once instead of on every request.
    pub fn native_decoder_session(&self) -> Result<NativeDecoderSession> {
        self.native_decoder_session_with_options(&NativeDecoderWeightOptions::default())
    }

    /// Load a resident native decoder session with selected RSMF tensor
    /// variants.
    pub fn native_decoder_session_with_options(
        &self,
        options: &NativeDecoderWeightOptions,
    ) -> Result<NativeDecoderSession> {
        Ok(NativeDecoderSession {
            weights: self.native_decoder_weights_with_options(options)?,
            tokenizer: self.native_decoder_tokenizer()?,
        })
    }

    /// Load the native decoder tokenizer from `tokenizer.json`.
    ///
    /// R4.7 supports simple `WordLevel` tokenizer JSON assets with a `vocab`
    /// map and optional `unk_token`.
    pub fn native_decoder_tokenizer(&self) -> Result<NativeDecoderTokenizer> {
        let tokenizer_asset = self
            .file
            .asset(NATIVE_DECODER_TOKENIZER_ASSET)
            .ok_or_else(|| RuntimeError::NativeDecoderAssetMissing {
                asset_name: NATIVE_DECODER_TOKENIZER_ASSET.to_string(),
            })?;
        NativeDecoderTokenizer::from_json(tokenizer_asset.bytes)
    }

    /// Run native decoder token-id generation.
    ///
    /// Defaults preserve greedy CPU-reference generation. Sampling options can
    /// enable deterministic top-k/top-p sampling. `Accelerated` resolves to
    /// Apple Accelerate on macOS when the `apple-accelerate` feature is enabled
    /// and otherwise falls back to CPU reference.
    pub fn native_decoder_greedy_decode(
        &self,
        input_token_ids: &[i64],
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderGenerateOutput> {
        let backend = resolve_native_decoder_backend(options.backend)?;
        let weights = self.native_decoder_weights_with_options(&options.weight_options)?;
        native_decoder_generate_with_backend(&weights, input_token_ids, options, backend)
    }

    /// Generate text with the native decoder tokenizer and token-id runtime.
    ///
    /// The prompt is encoded by [`NativeDecoderTokenizer`], passed to
    /// [`Self::native_decoder_greedy_decode`], and generated token ids are
    /// decoded back to text.
    pub fn native_decoder_generate_text(
        &self,
        prompt: &str,
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderTextGenerateOutput> {
        let weight_options = options.weight_options.clone();
        self.native_decoder_session_with_options(&weight_options)?
            .generate_text(prompt, options)
    }

    /// Compare native decoder logits against a caller-supplied reference.
    ///
    /// This is intended for local tiny-model fixtures or exported reference
    /// logits from another runtime. The check feeds `input_token_ids` one step
    /// at a time and compares each next-token logits row against
    /// `expected_logits`.
    pub fn native_decoder_check_reference_logits(
        &self,
        check: NativeDecoderReferenceLogitCheck,
    ) -> Result<NativeDecoderReferenceLogitReport> {
        let backend = resolve_native_decoder_backend(check.backend)?;
        let weights = self.native_decoder_weights_with_options(&check.weight_options)?;
        native_decoder_check_reference_logits(&weights, check, backend)
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
mod tests;
