use super::*;

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
    /// Optional direct quantized query projection.
    pub q_proj_quantized: Option<&'a NativeDecoderQuantizedMatrix>,
    /// Key projection weight, row-major shape `[kv_width, hidden_size]`.
    pub k_proj: &'a [f32],
    /// Optional direct quantized key projection.
    pub k_proj_quantized: Option<&'a NativeDecoderQuantizedMatrix>,
    /// Value projection weight, row-major shape `[kv_width, hidden_size]`.
    pub v_proj: &'a [f32],
    /// Optional direct quantized value projection.
    pub v_proj_quantized: Option<&'a NativeDecoderQuantizedMatrix>,
    /// Attention output projection weight, row-major shape `[hidden_size, hidden_size]`.
    pub o_proj: &'a [f32],
    /// Optional direct quantized attention output projection.
    pub o_proj_quantized: Option<&'a NativeDecoderQuantizedMatrix>,
    /// SwiGLU gate projection weight, row-major shape `[intermediate_size, hidden_size]`.
    pub gate_proj: &'a [f32],
    /// Optional direct quantized gate projection.
    pub gate_proj_quantized: Option<&'a NativeDecoderQuantizedMatrix>,
    /// SwiGLU up projection weight, row-major shape `[intermediate_size, hidden_size]`.
    pub up_proj: &'a [f32],
    /// Optional direct quantized up projection.
    pub up_proj_quantized: Option<&'a NativeDecoderQuantizedMatrix>,
    /// SwiGLU down projection weight, row-major shape `[hidden_size, intermediate_size]`.
    pub down_proj: &'a [f32],
    /// Optional direct quantized down projection.
    pub down_proj_quantized: Option<&'a NativeDecoderQuantizedMatrix>,
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
