use super::*;

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

/// Resident native decoder memory held by a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDecoderResidencyReport {
    /// Decoded resident weight bytes.
    pub resident_weight_bytes: usize,
    /// Resident KV-cache bytes currently retained by the session.
    ///
    /// The current native decoder session does not retain KV cache across
    /// generation calls, so this is `0`. Individual [`NativeDecoderKvCache`]
    /// values expose their own resident byte accounting while generation is
    /// active.
    pub resident_kv_cache_bytes: usize,
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
    /// Resident decoded memory retained by this session.
    #[must_use]
    pub fn residency_report(&self) -> NativeDecoderResidencyReport {
        NativeDecoderResidencyReport {
            resident_weight_bytes: self.weights.resident_bytes(),
            resident_kv_cache_bytes: 0,
        }
    }

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
