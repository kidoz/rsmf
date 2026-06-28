use super::*;
use std::sync::{Arc, Mutex};

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
    /// Performance counters for this generation call.
    pub performance: NativeDecoderPerformanceReport,
}

/// Resident native decoder memory held by a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDecoderResidencyReport {
    /// Decoded resident weight bytes.
    pub resident_weight_bytes: usize,
    /// Resident KV-cache bytes currently retained by the session prefix cache.
    pub resident_kv_cache_bytes: usize,
}

/// Resident native decoder session with decoded weights and tokenizer cached in
/// memory.
#[derive(Debug, Clone)]
pub struct NativeDecoderSession {
    /// Decoded native decoder weights.
    pub weights: NativeDecoderWeights,
    /// Decoded native decoder tokenizer.
    pub tokenizer: NativeDecoderTokenizer,
    pub(crate) prefix_cache: Arc<Mutex<NativeDecoderPrefixCache>>,
}

impl NativeDecoderSession {
    /// Resident decoded memory retained by this session.
    #[must_use]
    pub fn residency_report(&self) -> NativeDecoderResidencyReport {
        NativeDecoderResidencyReport {
            resident_weight_bytes: self.weights.resident_bytes(),
            resident_kv_cache_bytes: self.prefix_cache_report().resident_bytes,
        }
    }

    /// Current resident prefix-cache counters for this session.
    #[must_use]
    pub fn prefix_cache_report(&self) -> NativeDecoderPrefixCacheReport {
        self.prefix_cache.lock().map_or_else(
            |_| NativeDecoderPrefixCacheReport::default(),
            |cache| cache.report(),
        )
    }

    /// Generate token ids without reloading weights from the RSMF file.
    pub fn generate_token_ids(
        &self,
        input_token_ids: &[i64],
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderGenerateOutput> {
        let backend = resolve_native_decoder_backend(options.backend)?;
        native_decoder_generate_with_backend_and_prefix_cache(
            &self.weights,
            input_token_ids,
            options,
            backend,
            Some(&self.prefix_cache),
        )
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
            performance: output.performance,
        })
    }

    /// Generate text from role/content chat messages without reloading weights
    /// or tokenizer from the RSMF file.
    pub fn generate_chat(
        &self,
        messages: &[NativeDecoderChatMessage],
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderTextGenerateOutput> {
        let prompt = self.tokenizer.apply_chat_template(messages, true)?;
        self.generate_text(&prompt, options)
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

#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderStepOutput {
    /// Logits for the next token, shape `[vocab_size]`.
    pub logits: Vec<f32>,
    /// Greedy argmax token id selected from `logits`.
    pub next_token_id: i64,
}

pub(crate) struct NativeDecoderHiddenStepOutput {
    pub(crate) normalized: Vec<f32>,
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
    /// Performance counters for this generation call.
    pub performance: NativeDecoderPerformanceReport,
}

pub(crate) fn native_decoder_generate_with_backend(
    weights: &NativeDecoderWeights,
    input_token_ids: &[i64],
    options: NativeDecoderRunOptions,
    backend: NativeDecoderBackend,
) -> Result<NativeDecoderGenerateOutput> {
    native_decoder_generate_with_backend_and_prefix_cache(
        weights,
        input_token_ids,
        options,
        backend,
        None,
    )
}

pub(crate) fn native_decoder_generate_with_backend_and_prefix_cache(
    weights: &NativeDecoderWeights,
    input_token_ids: &[i64],
    options: NativeDecoderRunOptions,
    backend: NativeDecoderBackend,
    prefix_cache: Option<&Arc<Mutex<NativeDecoderPrefixCache>>>,
) -> Result<NativeDecoderGenerateOutput> {
    match backend {
        NativeDecoderBackend::CpuReference
        | NativeDecoderBackend::CpuThreaded
        | NativeDecoderBackend::AppleCpuAccelerate => native_decoder_cpu_greedy_decode(
            weights,
            input_token_ids,
            options,
            backend,
            prefix_cache,
        ),
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
    prefix_cache: Option<&Arc<Mutex<NativeDecoderPrefixCache>>>,
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
            performance: NativeDecoderPerformanceReport::default(),
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
    let mut prefix_stats = NativeDecoderPrefixCacheRunStats {
        miss_tokens: input_token_ids.len(),
        ..NativeDecoderPrefixCacheRunStats::default()
    };
    let can_use_prefix_cache =
        options.performance.prefix_cache_max_entries.is_some() && !options.return_prompt_logits;
    if can_use_prefix_cache {
        if let Some(prefix_cache) = prefix_cache {
            let hit = prefix_cache
                .lock()
                .map_err(|_| RuntimeError::CachePoisoned)?
                .lookup(input_token_ids, backend, &options.performance);
            if let Some(hit) = hit {
                prefix_stats.hit_tokens = hit.token_count;
                prefix_stats.miss_tokens = input_token_ids.len().saturating_sub(hit.token_count);
                cache = hit.cache;
                last_step = Some(hit.last_step);
            }
        }
    }
    let prefill_chunk_size = options
        .performance
        .prefill_chunk_size
        .unwrap_or(input_token_ids.len().max(1));
    for chunk in input_token_ids[prefix_stats.hit_tokens..].chunks(prefill_chunk_size) {
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
    if can_use_prefix_cache {
        if let (Some(prefix_cache), Some(last_step)) = (prefix_cache, last_step.as_ref()) {
            prefix_stats.evictions = prefix_cache
                .lock()
                .map_err(|_| RuntimeError::CachePoisoned)?
                .insert(
                    input_token_ids,
                    &cache,
                    last_step,
                    backend,
                    &options.performance,
                )?;
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

    let prefix_report = prefix_cache
        .map(|prefix_cache| {
            prefix_cache
                .lock()
                .map_err(|_| RuntimeError::CachePoisoned)
                .map(|cache| cache.report())
        })
        .transpose()?
        .unwrap_or_default();
    let performance = NativeDecoderPerformanceReport {
        prefix_cache_hit_tokens: prefix_stats.hit_tokens,
        prefix_cache_miss_tokens: prefix_stats.miss_tokens,
        prefix_cache_entries: prefix_report.entries,
        prefix_cache_bytes: prefix_report.resident_bytes,
        prefix_cache_evictions: prefix_stats.evictions,
        kv_cache_bytes: cache.resident_bytes(),
    };

    Ok(NativeDecoderGenerateOutput {
        token_ids,
        generated_token_ids,
        logits,
        prompt_logits,
        backend,
        performance,
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
    let hidden = native_decoder_cpu_step_hidden(weights, cache, token_id, backend, performance)?;
    let lm_head = weights.lm_head.as_ref().unwrap_or(&weights.token_embedding);
    let lm_head_quantized = if weights.lm_head.is_some() {
        weights.lm_head_quantized.as_ref()
    } else {
        None
    };
    native_decoder_cpu_step_from_normalized(
        &hidden.normalized,
        weights.config.hidden_size,
        lm_head,
        lm_head_quantized,
        weights.config.vocab_size,
        backend,
        performance,
    )
}

pub(crate) fn native_decoder_cpu_step_hidden(
    weights: &NativeDecoderWeights,
    cache: &mut NativeDecoderKvCache,
    token_id: i64,
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<NativeDecoderHiddenStepOutput> {
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
    Ok(NativeDecoderHiddenStepOutput { normalized })
}

pub(crate) struct NativeDecoderBatchHiddenStepRequest<'a> {
    pub(crate) cache: &'a mut NativeDecoderKvCache,
    pub(crate) token_id: i64,
}

pub(crate) fn native_decoder_cpu_step_hidden_batch(
    weights: &NativeDecoderWeights,
    requests: &mut [NativeDecoderBatchHiddenStepRequest<'_>],
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<Vec<Vec<f32>>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let rows = requests.len();
    let hidden_size = weights.config.hidden_size;
    let kv_width = weights.config.key_value_width();
    let mut hidden_states = Vec::with_capacity(rows * hidden_size);
    for request in requests.iter() {
        let token_index =
            validate_native_decoder_token_id(request.token_id, weights.config.vocab_size)?;
        validate_native_decoder_step_cache(weights, request.cache)?;
        let embed_start = token_index.checked_mul(hidden_size).ok_or_else(|| {
            native_decoder_cpu_shape_error("step_batch", "embedding offset overflow")
        })?;
        let embed_end = embed_start.checked_add(hidden_size).ok_or_else(|| {
            native_decoder_cpu_shape_error("step_batch", "embedding end overflow")
        })?;
        hidden_states.extend_from_slice(
            weights
                .token_embedding
                .get(embed_start..embed_end)
                .ok_or_else(|| {
                    native_decoder_cpu_shape_error(
                        "step_batch",
                        format!("token embedding missing row {token_index}"),
                    )
                })?,
        );
    }

    let mut layer_updates = (0..rows)
        .map(|_| Vec::with_capacity(weights.layers.len()))
        .collect::<Vec<Vec<(Vec<f32>, Vec<f32>)>>>();
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let cpu_weights = layer_weights.as_cpu();
        hidden_states = native_decoder_cpu_llama_cached_step_batch(
            &weights.config,
            &hidden_states,
            requests,
            layer_idx,
            NativeDecoderLinearBackend {
                backend,
                performance,
            },
            cpu_weights,
            &mut layer_updates,
        )?;
    }

    for (request, updates) in requests.iter_mut().zip(layer_updates) {
        for (layer_cache, (key, value)) in request.cache.layers.iter_mut().zip(updates) {
            append_native_decoder_layer_cache(
                layer_cache,
                &key,
                &value,
                request.cache.position,
                kv_width,
                request.cache.page_size_tokens,
            )?;
        }
        request.cache.position += 1;
    }

    let normalized = native_decoder_cpu_rms_norm(
        &hidden_states,
        rows,
        hidden_size,
        &weights.final_norm,
        weights.config.rms_norm_eps,
    )?;
    Ok(normalized
        .chunks(hidden_size)
        .map(<[f32]>::to_vec)
        .collect())
}

fn validate_native_decoder_step_cache(
    weights: &NativeDecoderWeights,
    cache: &NativeDecoderKvCache,
) -> Result<()> {
    if cache.layers.len() != weights.config.num_hidden_layers {
        return Err(native_decoder_cpu_shape_error(
            "step_batch",
            format!(
                "cache has {} layers, expected {}",
                cache.layers.len(),
                weights.config.num_hidden_layers
            ),
        ));
    }
    if cache.position >= weights.config.max_position_embeddings {
        return Err(native_decoder_cpu_shape_error(
            "step_batch",
            format!(
                "position {} exceeds max_position_embeddings {}",
                cache.position, weights.config.max_position_embeddings
            ),
        ));
    }
    Ok(())
}

pub(crate) fn native_decoder_cpu_step_from_normalized(
    normalized: &[f32],
    hidden_size: usize,
    lm_head: &[f32],
    lm_head_quantized: Option<&NativeDecoderQuantizedMatrix>,
    vocab_size: usize,
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<NativeDecoderStepOutput> {
    let logits = native_decoder_cpu_logits(
        normalized,
        hidden_size,
        lm_head,
        lm_head_quantized,
        vocab_size,
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
    let mut query = native_decoder_projection(
        &normalized,
        hidden_size,
        weights.q_proj,
        weights.q_proj_quantized,
        hidden_size,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let mut key = native_decoder_projection(
        &normalized,
        hidden_size,
        weights.k_proj,
        weights.k_proj_quantized,
        kv_width,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let value = native_decoder_projection(
        &normalized,
        hidden_size,
        weights.v_proj,
        weights.v_proj_quantized,
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
        implementation: linear_backend.performance.attention,
    })?;
    let attention_projected = native_decoder_projection(
        &attention,
        hidden_size,
        weights.o_proj,
        weights.o_proj_quantized,
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
    let gate = native_decoder_projection(
        &mlp_normalized,
        hidden_size,
        weights.gate_proj,
        weights.gate_proj_quantized,
        intermediate_size,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let up = native_decoder_projection(
        &mlp_normalized,
        hidden_size,
        weights.up_proj,
        weights.up_proj_quantized,
        intermediate_size,
        linear_backend.backend,
        linear_backend.performance,
    )?;
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| native_decoder_cpu_silu(*gate) * up)
        .collect::<Vec<_>>();
    let mlp_projected = native_decoder_projection(
        &activated,
        intermediate_size,
        weights.down_proj,
        weights.down_proj_quantized,
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

fn native_decoder_cpu_llama_cached_step_batch(
    config: &NativeDecoderConfig,
    hidden_states: &[f32],
    requests: &[NativeDecoderBatchHiddenStepRequest<'_>],
    layer_idx: usize,
    linear_backend: NativeDecoderLinearBackend<'_>,
    weights: NativeDecoderCpuLayerWeights<'_>,
    layer_updates: &mut [Vec<(Vec<f32>, Vec<f32>)>],
) -> Result<Vec<f32>> {
    if config.family != NativeDecoderFamily::Llama {
        return Err(RuntimeError::UnsupportedNativeDecoder {
            family: format!("{:?}", config.family),
        });
    }
    config.validate()?;
    let rows = requests.len();
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let head_dim = config.head_dim();
    let kv_width = config.key_value_width();
    validate_cpu_matrix_len(
        "llama_cached_step_batch",
        "hidden_states",
        hidden_states.len(),
        rows,
        hidden_size,
    )?;
    if layer_updates.len() != rows {
        return Err(native_decoder_cpu_shape_error(
            "llama_cached_step_batch",
            format!(
                "layer update rows mismatch: got {}, expected {rows}",
                layer_updates.len()
            ),
        ));
    }
    for request in requests {
        let layer_cache = request.cache.layers.get(layer_idx).ok_or_else(|| {
            native_decoder_cpu_shape_error(
                "llama_cached_step_batch",
                format!("cache missing layer {layer_idx}"),
            )
        })?;
        validate_native_decoder_layer_cache(
            layer_cache,
            request.cache.position,
            kv_width,
            request.cache.page_size_tokens,
        )?;
    }

    let normalized = native_decoder_cpu_rms_norm(
        hidden_states,
        rows,
        hidden_size,
        weights.input_layernorm,
        config.rms_norm_eps,
    )?;
    let NativeDecoderQkvBatchOutput {
        mut query,
        mut key,
        value,
    } = native_decoder_qkv_projection_batch(
        &normalized,
        rows,
        hidden_size,
        kv_width,
        linear_backend,
        weights,
    )?;

    for (row, request) in requests.iter().enumerate() {
        let query_start = row * hidden_size;
        let key_start = row * kv_width;
        native_decoder_cpu_apply_llama_rope(
            &mut query[query_start..query_start + hidden_size],
            1,
            config.num_attention_heads,
            head_dim,
            request.cache.position,
            config.rope_theta,
        )?;
        native_decoder_cpu_apply_llama_rope(
            &mut key[key_start..key_start + kv_width],
            1,
            config.num_key_value_heads,
            head_dim,
            request.cache.position,
            config.rope_theta,
        )?;
    }

    let NativeDecoderCachedAttentionBatchOutput {
        attention_rows,
        updates,
    } = native_decoder_cpu_layer_cached_attention_batch(NativeDecoderLayerAttentionBatchInput {
        query: &query,
        key: &key,
        value: &value,
        requests,
        layer_idx,
        rows,
        hidden_size,
        kv_width,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim,
        implementation: linear_backend.performance.attention,
    })?;
    for (row_updates, update) in layer_updates.iter_mut().zip(updates) {
        row_updates.push(update);
    }

    let attention_projected = native_decoder_projection_batch(
        &attention_rows,
        rows,
        hidden_size,
        weights.o_proj,
        weights.o_proj_quantized,
        hidden_size,
        linear_backend,
    )?;
    let attention_residual = add_same_shape(
        "llama_cached_step_batch",
        hidden_states,
        &attention_projected,
        "attention residual",
    )?;
    let mlp_normalized = native_decoder_cpu_rms_norm(
        &attention_residual,
        rows,
        hidden_size,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
    )?;
    let NativeDecoderGateUpBatchOutput { gate, up } = native_decoder_gate_up_projection_batch(
        &mlp_normalized,
        rows,
        hidden_size,
        intermediate_size,
        linear_backend,
        weights,
    )?;
    let activated = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| native_decoder_cpu_silu(*gate) * up)
        .collect::<Vec<_>>();
    let mlp_projected = native_decoder_projection_batch(
        &activated,
        rows,
        intermediate_size,
        weights.down_proj,
        weights.down_proj_quantized,
        hidden_size,
        linear_backend,
    )?;
    add_same_shape(
        "llama_cached_step_batch",
        &attention_residual,
        &mlp_projected,
        "mlp residual",
    )
}

struct NativeDecoderQkvBatchOutput {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

fn native_decoder_qkv_projection_batch(
    input: &[f32],
    rows: usize,
    hidden_size: usize,
    kv_width: usize,
    linear_backend: NativeDecoderLinearBackend<'_>,
    weights: NativeDecoderCpuLayerWeights<'_>,
) -> Result<NativeDecoderQkvBatchOutput> {
    if weights.q_proj_quantized.is_none()
        && weights.k_proj_quantized.is_none()
        && weights.v_proj_quantized.is_none()
    {
        validate_cpu_matrix_len(
            "qkv_projection_batch",
            "input",
            input.len(),
            rows,
            hidden_size,
        )?;
        validate_cpu_matrix_len(
            "qkv_projection_batch",
            "q_proj",
            weights.q_proj.len(),
            hidden_size,
            hidden_size,
        )?;
        validate_cpu_matrix_len(
            "qkv_projection_batch",
            "k_proj",
            weights.k_proj.len(),
            kv_width,
            hidden_size,
        )?;
        validate_cpu_matrix_len(
            "qkv_projection_batch",
            "v_proj",
            weights.v_proj.len(),
            kv_width,
            hidden_size,
        )?;
        let fused_out_features = hidden_size
            .checked_add(kv_width)
            .and_then(|value| value.checked_add(kv_width))
            .ok_or_else(|| {
                native_decoder_cpu_shape_error("qkv_projection_batch", "fused width overflow")
            })?;
        let fused_weight_len = cpu_element_count(
            "qkv_projection_batch",
            "fused weight",
            fused_out_features,
            hidden_size,
        )?;
        let mut fused_weight = Vec::with_capacity(fused_weight_len);
        fused_weight.extend_from_slice(weights.q_proj);
        fused_weight.extend_from_slice(weights.k_proj);
        fused_weight.extend_from_slice(weights.v_proj);
        let fused = native_decoder_backend_linear(
            input,
            rows,
            hidden_size,
            &fused_weight,
            fused_out_features,
            linear_backend.backend,
            linear_backend.performance,
        )?;
        let mut query = Vec::with_capacity(rows * hidden_size);
        let mut key = Vec::with_capacity(rows * kv_width);
        let mut value = Vec::with_capacity(rows * kv_width);
        for row in 0..rows {
            let fused_start = row * fused_out_features;
            let query_end = fused_start + hidden_size;
            let key_end = query_end + kv_width;
            let value_end = key_end + kv_width;
            query.extend_from_slice(&fused[fused_start..query_end]);
            key.extend_from_slice(&fused[query_end..key_end]);
            value.extend_from_slice(&fused[key_end..value_end]);
        }
        return Ok(NativeDecoderQkvBatchOutput { query, key, value });
    }

    Ok(NativeDecoderQkvBatchOutput {
        query: native_decoder_projection_batch(
            input,
            rows,
            hidden_size,
            weights.q_proj,
            weights.q_proj_quantized,
            hidden_size,
            linear_backend,
        )?,
        key: native_decoder_projection_batch(
            input,
            rows,
            hidden_size,
            weights.k_proj,
            weights.k_proj_quantized,
            kv_width,
            linear_backend,
        )?,
        value: native_decoder_projection_batch(
            input,
            rows,
            hidden_size,
            weights.v_proj,
            weights.v_proj_quantized,
            kv_width,
            linear_backend,
        )?,
    })
}

struct NativeDecoderLayerAttentionBatchInput<'a> {
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    requests: &'a [NativeDecoderBatchHiddenStepRequest<'a>],
    layer_idx: usize,
    rows: usize,
    hidden_size: usize,
    kv_width: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    implementation: NativeDecoderAttentionImplementation,
}

struct NativeDecoderCachedAttentionBatchOutput {
    attention_rows: Vec<f32>,
    updates: Vec<(Vec<f32>, Vec<f32>)>,
}

fn native_decoder_cpu_layer_cached_attention_batch(
    input: NativeDecoderLayerAttentionBatchInput<'_>,
) -> Result<NativeDecoderCachedAttentionBatchOutput> {
    if input.requests.len() != input.rows {
        return Err(native_decoder_cpu_shape_error(
            "layer_cached_attention_batch",
            format!(
                "request rows mismatch: got {}, expected {}",
                input.requests.len(),
                input.rows
            ),
        ));
    }
    validate_cpu_matrix_len(
        "layer_cached_attention_batch",
        "query",
        input.query.len(),
        input.rows,
        input.hidden_size,
    )?;
    validate_cpu_matrix_len(
        "layer_cached_attention_batch",
        "key",
        input.key.len(),
        input.rows,
        input.kv_width,
    )?;
    validate_cpu_matrix_len(
        "layer_cached_attention_batch",
        "value",
        input.value.len(),
        input.rows,
        input.kv_width,
    )?;
    let mut attention_rows = Vec::with_capacity(input.rows * input.hidden_size);
    let mut updates = Vec::with_capacity(input.rows);
    for (row, request) in input.requests.iter().enumerate() {
        let query_start = row * input.hidden_size;
        let key_start = row * input.kv_width;
        let current_key = input.key[key_start..key_start + input.kv_width].to_vec();
        let current_value = input.value[key_start..key_start + input.kv_width].to_vec();
        let layer_cache = request.cache.layers.get(input.layer_idx).ok_or_else(|| {
            native_decoder_cpu_shape_error(
                "layer_cached_attention_batch",
                format!("cache missing layer {}", input.layer_idx),
            )
        })?;
        let attention =
            native_decoder_cpu_layer_cached_attention(NativeDecoderLayerAttentionInput {
                query: &input.query[query_start..query_start + input.hidden_size],
                cache: layer_cache,
                current_key: &current_key,
                current_value: &current_value,
                page_size_tokens: request.cache.page_size_tokens,
                cache_len: request.cache.position + 1,
                num_attention_heads: input.num_attention_heads,
                num_key_value_heads: input.num_key_value_heads,
                head_dim: input.head_dim,
                implementation: input.implementation,
            })?;
        attention_rows.extend_from_slice(&attention);
        updates.push((current_key, current_value));
    }
    Ok(NativeDecoderCachedAttentionBatchOutput {
        attention_rows,
        updates,
    })
}

struct NativeDecoderGateUpBatchOutput {
    gate: Vec<f32>,
    up: Vec<f32>,
}

fn native_decoder_gate_up_projection_batch(
    input: &[f32],
    rows: usize,
    hidden_size: usize,
    intermediate_size: usize,
    linear_backend: NativeDecoderLinearBackend<'_>,
    weights: NativeDecoderCpuLayerWeights<'_>,
) -> Result<NativeDecoderGateUpBatchOutput> {
    if weights.gate_proj_quantized.is_none() && weights.up_proj_quantized.is_none() {
        validate_cpu_matrix_len(
            "gate_up_projection_batch",
            "input",
            input.len(),
            rows,
            hidden_size,
        )?;
        validate_cpu_matrix_len(
            "gate_up_projection_batch",
            "gate_proj",
            weights.gate_proj.len(),
            intermediate_size,
            hidden_size,
        )?;
        validate_cpu_matrix_len(
            "gate_up_projection_batch",
            "up_proj",
            weights.up_proj.len(),
            intermediate_size,
            hidden_size,
        )?;
        let fused_out_features = intermediate_size.checked_mul(2).ok_or_else(|| {
            native_decoder_cpu_shape_error("gate_up_projection_batch", "fused width overflow")
        })?;
        let fused_weight_len = cpu_element_count(
            "gate_up_projection_batch",
            "fused weight",
            fused_out_features,
            hidden_size,
        )?;
        let mut fused_weight = Vec::with_capacity(fused_weight_len);
        fused_weight.extend_from_slice(weights.gate_proj);
        fused_weight.extend_from_slice(weights.up_proj);
        let fused = native_decoder_backend_linear(
            input,
            rows,
            hidden_size,
            &fused_weight,
            fused_out_features,
            linear_backend.backend,
            linear_backend.performance,
        )?;
        let mut gate = Vec::with_capacity(rows * intermediate_size);
        let mut up = Vec::with_capacity(rows * intermediate_size);
        for row in 0..rows {
            let fused_start = row * fused_out_features;
            let gate_end = fused_start + intermediate_size;
            let up_end = gate_end + intermediate_size;
            gate.extend_from_slice(&fused[fused_start..gate_end]);
            up.extend_from_slice(&fused[gate_end..up_end]);
        }
        return Ok(NativeDecoderGateUpBatchOutput { gate, up });
    }

    Ok(NativeDecoderGateUpBatchOutput {
        gate: native_decoder_projection_batch(
            input,
            rows,
            hidden_size,
            weights.gate_proj,
            weights.gate_proj_quantized,
            intermediate_size,
            linear_backend,
        )?,
        up: native_decoder_projection_batch(
            input,
            rows,
            hidden_size,
            weights.up_proj,
            weights.up_proj_quantized,
            intermediate_size,
            linear_backend,
        )?,
    })
}

fn native_decoder_projection(
    input: &[f32],
    in_features: usize,
    weight: &[f32],
    weight_quantized: Option<&NativeDecoderQuantizedMatrix>,
    out_features: usize,
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<Vec<f32>> {
    if let Some(quantized) = weight_quantized {
        validate_cpu_vector_len("projection_quantized", "input", input.len(), in_features)?;
        let output = quantized.matvec(input)?;
        validate_cpu_vector_len("projection_quantized", "output", output.len(), out_features)?;
        return Ok(output);
    }
    native_decoder_backend_linear(
        input,
        1,
        in_features,
        weight,
        out_features,
        backend,
        performance,
    )
}

fn native_decoder_projection_batch(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    weight_quantized: Option<&NativeDecoderQuantizedMatrix>,
    out_features: usize,
    linear_backend: NativeDecoderLinearBackend<'_>,
) -> Result<Vec<f32>> {
    validate_cpu_matrix_len("projection_batch", "input", input.len(), rows, in_features)?;
    if let Some(quantized) = weight_quantized {
        let mut output = Vec::with_capacity(rows * out_features);
        for row in 0..rows {
            let start = row * in_features;
            let row_output = quantized.matvec(&input[start..start + in_features])?;
            validate_cpu_vector_len(
                "projection_batch_quantized",
                "output",
                row_output.len(),
                out_features,
            )?;
            output.extend(row_output);
        }
        return Ok(output);
    }
    native_decoder_backend_linear(
        input,
        rows,
        in_features,
        weight,
        out_features,
        linear_backend.backend,
        linear_backend.performance,
    )
}

pub(crate) fn native_decoder_cpu_logits(
    normalized: &[f32],
    hidden_size: usize,
    lm_head: &[f32],
    lm_head_quantized: Option<&NativeDecoderQuantizedMatrix>,
    vocab_size: usize,
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<Vec<f32>> {
    if let Some(quantized) = lm_head_quantized {
        validate_cpu_matrix_len(
            "logits_quantized",
            "normalized",
            normalized.len(),
            1,
            hidden_size,
        )?;
        return quantized.matvec(normalized);
    }
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

pub(crate) fn native_decoder_cpu_logits_batch(
    input: NativeDecoderCpuLogitsBatchInput<'_>,
) -> Result<Vec<Vec<f32>>> {
    validate_cpu_matrix_len(
        "logits_batch",
        "normalized",
        input.normalized.len(),
        input.rows,
        input.hidden_size,
    )?;
    if let Some(quantized) = input.lm_head_quantized {
        let mut outputs = Vec::with_capacity(input.rows);
        for row in 0..input.rows {
            let start = row * input.hidden_size;
            outputs.push(quantized.matvec(&input.normalized[start..start + input.hidden_size])?);
        }
        return Ok(outputs);
    }
    let flat = match input.backend {
        NativeDecoderBackend::AppleCpuAccelerate | NativeDecoderBackend::CpuThreaded => {
            native_decoder_backend_linear(
                input.normalized,
                input.rows,
                input.hidden_size,
                input.lm_head,
                input.vocab_size,
                input.backend,
                input.performance,
            )?
        }
        NativeDecoderBackend::CpuReference
        | NativeDecoderBackend::Auto
        | NativeDecoderBackend::Accelerated
        | NativeDecoderBackend::MetalWgpuLmHead
        | NativeDecoderBackend::MetalWgpuFullDecoder
        | NativeDecoderBackend::OrtCoreMl => native_decoder_cpu_linear(
            input.normalized,
            input.rows,
            input.hidden_size,
            input.lm_head,
            input.vocab_size,
        )?,
    };
    Ok(flat
        .chunks(input.vocab_size)
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>())
}

pub(crate) struct NativeDecoderCpuLogitsBatchInput<'a> {
    pub(crate) normalized: &'a [f32],
    pub(crate) rows: usize,
    pub(crate) hidden_size: usize,
    pub(crate) lm_head: &'a [f32],
    pub(crate) lm_head_quantized: Option<&'a NativeDecoderQuantizedMatrix>,
    pub(crate) vocab_size: usize,
    pub(crate) backend: NativeDecoderBackend,
    pub(crate) performance: &'a NativeDecoderPerformanceOptions,
}
