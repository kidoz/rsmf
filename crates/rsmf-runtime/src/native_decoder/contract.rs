use super::*;

pub const NATIVE_DECODER_CONFIG_ASSET: &str = "config.json";

/// Canonical RSMF asset name for the tokenizer payload used by native decoders.
pub const NATIVE_DECODER_TOKENIZER_ASSET: &str = "tokenizer.json";

/// Optional RSMF asset name for Hugging Face tokenizer configuration.
pub const NATIVE_DECODER_TOKENIZER_CONFIG_ASSET: &str = "tokenizer_config.json";

/// Optional RSMF asset name for a standalone chat template payload.
pub const NATIVE_DECODER_CHAT_TEMPLATE_ASSET: &str = "chat_template.json";

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
