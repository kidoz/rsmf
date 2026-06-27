use super::*;

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
