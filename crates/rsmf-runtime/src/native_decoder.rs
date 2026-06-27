use std::collections::HashMap;

use rsmf_core::tensor::variant::LayoutTag;
use rsmf_core::{LogicalDtype, RsmfError, RsmfFile, TensorDescriptor, TensorView};
use serde::{Deserialize, Serialize};

use crate::{Result, RuntimeError};

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
    pub(crate) fn from_model_type(model_type: &str) -> Result<Self> {
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

    pub(crate) fn validate(&self) -> Result<()> {
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
    pub(crate) fn from_file(file: &RsmfFile) -> Result<Self> {
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
    normalizer: NativeDecoderNormalizer,
    pre_tokenizer: NativeDecoderPreTokenizer,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_fallback: bool,
}

impl NativeDecoderTokenizer {
    pub(crate) fn from_json(bytes: &[u8]) -> Result<Self> {
        let raw: NativeDecoderTokenizerJson = serde_json::from_slice(bytes).map_err(|error| {
            RuntimeError::NativeDecoderTokenizerInvalid {
                reason: error.to_string(),
            }
        })?;
        reject_supported_tokenizer_component("post_processor", raw.post_processor.as_ref())?;
        let normalizer = NativeDecoderNormalizer::from_json(raw.normalizer.as_ref())?;
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
            normalizer,
            pre_tokenizer,
            bpe_ranks,
            byte_fallback: raw.model.byte_fallback,
        })
    }

    /// Encode text to token ids.
    ///
    /// WordLevel tokenizers use whitespace token lookup. BPE tokenizers support
    /// simple whitespace or ByteLevel-style pre-tokenization, vocab/merges, and
    /// exact special-token lookup. Unsupported tokenizer components fail at
    /// load time with typed errors.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let normalized = self.normalizer.normalize(text);
        let mut token_ids = Vec::new();
        for token in self.pre_tokenizer.pieces(&normalized) {
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
            NativeDecoderTokenizerMode::Bpe => Ok(self.decode_bpe_tokens(&tokens)?),
        }
    }

    pub(crate) fn lookup_token_id(&self, token: &str) -> Result<i64> {
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

    pub(crate) fn encode_bpe_piece(&self, piece: &str) -> Result<Vec<i64>> {
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
            .map(|symbol| self.lookup_bpe_symbol_ids(&symbol))
            .collect::<Result<Vec<_>>>()
            .map(|parts| parts.into_iter().flatten().collect())
    }

    pub(crate) fn lookup_bpe_symbol_ids(&self, symbol: &str) -> Result<Vec<i64>> {
        if let Some(token_id) = self.vocab.get(symbol) {
            return Ok(vec![*token_id]);
        }
        if self.byte_fallback {
            return symbol
                .as_bytes()
                .iter()
                .map(|byte| {
                    self.lookup_token_id(&format!("<0x{byte:02X}>"))
                        .map_err(|_| RuntimeError::NativeDecoderTokenizerTokenUnknown {
                            token: symbol.to_string(),
                        })
                })
                .collect();
        }
        self.lookup_token_id(symbol).map(|token_id| vec![token_id])
    }

    pub(crate) fn decode_bpe_tokens(&self, tokens: &[String]) -> Result<String> {
        if !self.byte_fallback {
            return Ok(decode_bpe_tokens(tokens));
        }
        let mut decoded_tokens = Vec::new();
        let mut byte_buffer = Vec::new();
        for token in tokens {
            if let Some(byte) = parse_byte_fallback_token(token) {
                byte_buffer.push(byte);
            } else {
                flush_byte_fallback_buffer(&mut byte_buffer, &mut decoded_tokens)?;
                decoded_tokens.push(token.clone());
            }
        }
        flush_byte_fallback_buffer(&mut byte_buffer, &mut decoded_tokens)?;
        Ok(decode_bpe_tokens(&decoded_tokens))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderTokenizerMode {
    WordLevel,
    Bpe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderNormalizer {
    None,
    Lowercase,
    Sequence(Vec<NativeDecoderNormalizer>),
}

impl NativeDecoderNormalizer {
    pub(crate) fn from_json(value: Option<&serde_json::Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::None);
        };
        if value.is_null() {
            return Ok(Self::None);
        }
        let Some(kind) = value.get("type").and_then(serde_json::Value::as_str) else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "normalizer.type is required when normalizer is present".to_string(),
            });
        };
        match kind {
            "Lowercase" => Ok(Self::Lowercase),
            "Sequence" => {
                let normalizers = value
                    .get("normalizers")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "normalizer Sequence requires a normalizers array".to_string(),
                    })?
                    .iter()
                    .map(|normalizer| Self::from_json(Some(normalizer)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self::Sequence(normalizers))
            }
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported normalizer {other}"),
            }),
        }
    }

    pub(crate) fn normalize(&self, text: &str) -> String {
        match self {
            Self::None => text.to_string(),
            Self::Lowercase => text.to_lowercase(),
            Self::Sequence(normalizers) => normalizers
                .iter()
                .fold(text.to_string(), |text, next| next.normalize(&text)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderPreTokenizer {
    Whitespace,
    ByteLevel { add_prefix_space: bool },
}

impl NativeDecoderPreTokenizer {
    pub(crate) fn from_json(value: Option<&serde_json::Value>) -> Result<Self> {
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

    pub(crate) fn pieces(self, text: &str) -> Vec<String> {
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
pub(crate) struct NativeDecoderTokenizerJson {
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
pub(crate) struct NativeDecoderTokenizerModelJson {
    #[serde(rename = "type")]
    tokenizer_type: String,
    #[serde(default)]
    vocab: HashMap<String, i64>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    merges: Vec<NativeDecoderBpeMergeJson>,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NativeDecoderAddedTokenJson {
    id: i64,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum NativeDecoderBpeMergeJson {
    Text(String),
    Pair([String; 2]),
}

pub(crate) fn reject_supported_tokenizer_component(
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

pub(crate) fn bpe_ranks_from_merges(
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

pub(crate) fn decode_bpe_tokens(tokens: &[String]) -> String {
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

pub(crate) fn parse_byte_fallback_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

pub(crate) fn flush_byte_fallback_buffer(
    byte_buffer: &mut Vec<u8>,
    decoded_tokens: &mut Vec<String>,
) -> Result<()> {
    if byte_buffer.is_empty() {
        return Ok(());
    }
    let text = String::from_utf8(std::mem::take(byte_buffer)).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("byte_fallback emitted invalid UTF-8: {error}"),
        }
    })?;
    decoded_tokens.push(text);
    Ok(())
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
    pub(crate) fn as_cpu(&self) -> NativeDecoderCpuLayerWeights<'_> {
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

    pub(crate) fn with_page_size(
        config: &NativeDecoderConfig,
        page_size_tokens: Option<usize>,
    ) -> Self {
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
pub(crate) struct NativeDecoderLayerKvCache {
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
pub(crate) struct HfNativeDecoderConfig {
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
pub(crate) enum TokenIds {
    One(i64),
    Many(Vec<i64>),
}

impl TokenIds {
    pub(crate) fn into_vec(self) -> Vec<i64> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

pub(crate) struct ExpectedNativeDecoderTensor {
    pub(crate) role: String,
    pub(crate) tensor_name: String,
    pub(crate) shape: Vec<u64>,
}

pub(crate) fn validate_positive(field: &str, value: usize) -> Result<()> {
    if value == 0 {
        Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: format!("{field} must be positive"),
        })
    } else {
        Ok(())
    }
}

pub(crate) fn expected_native_decoder_tensors(
    config: &NativeDecoderConfig,
) -> Result<Vec<ExpectedNativeDecoderTensor>> {
    match config.family {
        NativeDecoderFamily::Llama => expected_llama_tensors(config),
    }
}

pub(crate) fn expected_llama_tensors(
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

pub(crate) fn expected_tensor(
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

pub(crate) fn validate_native_decoder_tensor(
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

pub(crate) fn validate_native_decoder_weight_dtype(tensor: &TensorDescriptor) -> Result<()> {
    match tensor.dtype {
        LogicalDtype::F32 | LogicalDtype::F16 | LogicalDtype::BF16 => Ok(()),
        dtype => Err(RuntimeError::NativeDecoderTensorDtypeUnsupported {
            tensor_name: tensor.name.clone(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

pub(crate) fn format_shape(shape: &[u64]) -> String {
    let values = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}

pub(crate) fn usize_to_u64(value: usize, field: &str) -> Result<u64> {
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

pub(crate) fn validate_cpu_matrix_len(
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

pub(crate) fn cpu_element_count(
    operation: &'static str,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    rows.checked_mul(cols).ok_or_else(|| {
        native_decoder_cpu_shape_error(operation, format!("{name} element count overflow"))
    })
}

pub(crate) fn validate_cpu_vector_len(
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

pub(crate) fn validate_cpu_positive(
    operation: &'static str,
    name: &str,
    value: usize,
) -> Result<()> {
    if value == 0 {
        Err(native_decoder_cpu_shape_error(
            operation,
            format!("{name} must be positive"),
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn native_decoder_cpu_shape_error(
    operation: &'static str,
    reason: impl Into<String>,
) -> RuntimeError {
    RuntimeError::Shape(format!("native decoder CPU {operation}: {}", reason.into()))
}

pub(crate) fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

pub(crate) fn add_same_shape(
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

pub(crate) fn resolve_native_decoder_backend(
    backend: NativeDecoderBackend,
) -> Result<NativeDecoderBackend> {
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

pub(crate) fn native_decoder_generate_with_backend(
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

pub(crate) fn apple_accelerate_available() -> bool {
    cfg!(all(target_os = "macos", feature = "apple-accelerate"))
}

pub(crate) fn load_native_decoder_weights(
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

pub(crate) fn load_native_decoder_tensor_f32(
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

pub(crate) fn native_decoder_weight_view<'a>(
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

pub(crate) fn shape_element_count(shape: &[u64]) -> Result<usize> {
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

pub(crate) fn native_decoder_cpu_greedy_decode(
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

pub(crate) fn native_decoder_check_reference_logits(
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

pub(crate) fn native_decoder_kv_cache_with_performance(
    config: &NativeDecoderConfig,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<NativeDecoderKvCache> {
    if let Some(page_size) = performance.kv_cache_page_size_tokens {
        NativeDecoderKvCache::new_paged(config, page_size)
    } else {
        Ok(NativeDecoderKvCache::new(config))
    }
}

pub(crate) fn append_native_decoder_layer_cache(
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

pub(crate) fn validate_native_decoder_sampling_options(
    options: &NativeDecoderSamplingOptions,
) -> Result<()> {
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

pub(crate) fn validate_native_decoder_performance_options(
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

pub(crate) fn apply_native_decoder_repetition_penalty(
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

pub(crate) fn select_native_decoder_token(
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

pub(crate) struct NativeDecoderSamplerRng {
    state: u64,
}

impl NativeDecoderSamplerRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    pub(crate) fn next_unit_f32(&mut self) -> f32 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        let value = value.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((value >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

pub(crate) fn native_decoder_cpu_step(
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

pub(crate) struct NativeDecoderCpuCachedBlockOutput {
    hidden_states: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

#[derive(Clone, Copy)]
pub(crate) struct NativeDecoderLinearBackend<'a> {
    backend: NativeDecoderBackend,
    performance: &'a NativeDecoderPerformanceOptions,
}

pub(crate) fn native_decoder_cpu_llama_cached_step(
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

pub(crate) fn validate_native_decoder_layer_cache(
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

pub(crate) struct NativeDecoderLayerAttentionInput<'a> {
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

pub(crate) fn native_decoder_cpu_layer_cached_attention(
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

pub(crate) struct NativeDecoderCacheSliceRequest<'a> {
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

pub(crate) fn native_decoder_cache_kv_slice<'a>(
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

pub(crate) fn native_decoder_cpu_logits(
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

pub(crate) fn native_decoder_backend_linear(
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
pub(crate) fn native_decoder_apple_accelerate_linear(
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
pub(crate) fn native_decoder_apple_accelerate_linear(
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

pub(crate) fn native_decoder_cpu_linear_threaded(
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

pub(crate) fn validate_native_decoder_token_id(token_id: i64, vocab_size: usize) -> Result<usize> {
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

pub(crate) fn greedy_argmax_token(logits: &[f32]) -> Result<i64> {
    let (index, _) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| native_decoder_cpu_shape_error("greedy", "logits must not be empty"))?;
    i64::try_from(index).map_err(|_| native_decoder_cpu_shape_error("greedy", "token id overflow"))
}
