use super::*;
use std::time::Instant;

/// Request admitted to the native decoder token-level continuous scheduler.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderContinuousBatchRequest {
    /// Stable caller-provided request id used in reports and errors.
    pub request_id: String,
    /// Prompt token ids.
    pub input_token_ids: Vec<i64>,
    /// Per-request generation options.
    pub options: NativeDecoderRunOptions,
    /// Optional deadline checked before prefill and before each decode step.
    pub deadline: Option<Instant>,
    /// Whether the request is already cancelled before admission.
    pub cancelled: bool,
}

/// Output for one request from a continuous native decoder batch.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderContinuousBatchOutput {
    /// Caller-provided request id.
    pub request_id: String,
    /// Generated token output for this request.
    pub output: NativeDecoderGenerateOutput,
}

/// Scheduler-level report for a continuous native decoder batch.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NativeDecoderContinuousBatchReport {
    /// Requests admitted into the scheduling pass.
    pub admitted_requests: usize,
    /// Decode scheduling rounds executed after prefill.
    pub decode_rounds: usize,
    /// Total decode steps scheduled across all requests.
    pub scheduled_decode_steps: usize,
}

impl NativeDecoderSession {
    /// Generate multiple token-id requests by interleaving decode steps across
    /// active requests.
    ///
    /// This is a native token-level scheduling primitive. It does not yet fuse
    /// projection kernels across requests, but it exposes the correct execution
    /// contract for per-request stop, cancellation, deadline, and tenant-aware
    /// serving layers.
    pub fn generate_token_ids_continuous_batch(
        &self,
        requests: Vec<NativeDecoderContinuousBatchRequest>,
    ) -> Result<(
        Vec<NativeDecoderContinuousBatchOutput>,
        NativeDecoderContinuousBatchReport,
    )> {
        native_decoder_generate_continuous_batch(&self.weights, requests)
    }
}

struct ActiveRequest {
    index: usize,
    request_id: String,
    options: NativeDecoderRunOptions,
    backend: NativeDecoderBackend,
    deadline: Option<Instant>,
    cache: NativeDecoderKvCache,
    token_ids: Vec<i64>,
    generated_token_ids: Vec<i64>,
    logits: Vec<Vec<f32>>,
    prompt_logits: Vec<Vec<f32>>,
    last_step: NativeDecoderStepOutput,
    sampler_rng: NativeDecoderSamplerRng,
    stop_token_ids: Vec<i64>,
}

fn native_decoder_generate_continuous_batch(
    weights: &NativeDecoderWeights,
    requests: Vec<NativeDecoderContinuousBatchRequest>,
) -> Result<(
    Vec<NativeDecoderContinuousBatchOutput>,
    NativeDecoderContinuousBatchReport,
)> {
    if requests.is_empty() {
        return Ok((Vec::new(), NativeDecoderContinuousBatchReport::default()));
    }
    let max_requests = requests
        .iter()
        .filter_map(|request| request.options.performance.continuous_batch_max_requests)
        .min()
        .unwrap_or(requests.len());
    if requests.len() > max_requests {
        return Err(RuntimeError::NativeDecoderConfigInvalid {
            reason: format!(
                "continuous batch has {} requests, limit is {max_requests}",
                requests.len()
            ),
        });
    }

    let mut active = Vec::with_capacity(requests.len());
    let mut completed = Vec::with_capacity(requests.len());
    for (index, request) in requests.into_iter().enumerate() {
        if request.cancelled {
            return Err(RuntimeError::RequestCancelled {
                request_id: request.request_id,
            });
        }
        check_deadline(&request.request_id, request.deadline)?;
        let backend = resolve_native_decoder_backend(request.options.backend)?;
        validate_native_decoder_sampling_options(&request.options.sampling)?;
        validate_native_decoder_performance_options(&request.options.performance)?;
        if request.input_token_ids.is_empty() {
            return Err(RuntimeError::NativeDecoderPromptEmpty);
        }
        for &token_id in &request.input_token_ids {
            validate_native_decoder_token_id(token_id, weights.config.vocab_size)?;
        }
        if request.options.min_new_tokens > request.options.max_new_tokens {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "min_new_tokens must be less than or equal to max_new_tokens".to_string(),
            });
        }
        let mut cache = native_decoder_kv_cache_with_performance(
            &weights.config,
            &request.options.performance,
        )?;
        let prefill_chunk_size = request
            .options
            .performance
            .prefill_chunk_size
            .unwrap_or(request.input_token_ids.len().max(1));
        let mut last_step = None;
        let mut prompt_logits = Vec::new();
        for chunk in request.input_token_ids.chunks(prefill_chunk_size) {
            for &token_id in chunk {
                check_deadline(&request.request_id, request.deadline)?;
                let step = native_decoder_cpu_step(
                    weights,
                    &mut cache,
                    token_id,
                    backend,
                    &request.options.performance,
                )?;
                if request.options.return_prompt_logits {
                    prompt_logits.push(step.logits.clone());
                }
                last_step = Some(step);
            }
        }
        let Some(last_step) = last_step else {
            return Err(RuntimeError::NativeDecoderPromptEmpty);
        };
        if request.options.max_new_tokens == 0 {
            completed.push((
                index,
                NativeDecoderContinuousBatchOutput {
                    request_id: request.request_id,
                    output: NativeDecoderGenerateOutput {
                        token_ids: request.input_token_ids,
                        generated_token_ids: Vec::new(),
                        logits: Vec::new(),
                        prompt_logits,
                        backend,
                        performance: NativeDecoderPerformanceReport {
                            prefix_cache_miss_tokens: cache.position(),
                            kv_cache_bytes: cache.resident_bytes(),
                            ..NativeDecoderPerformanceReport::default()
                        },
                    },
                },
            ));
            continue;
        }
        let stop_token_ids = if !request.options.stop_token_ids.is_empty() {
            request.options.stop_token_ids.clone()
        } else if !request.options.eos_token_ids.is_empty() {
            request.options.eos_token_ids.clone()
        } else {
            weights.config.eos_token_ids.clone()
        };
        let sampler_rng = NativeDecoderSamplerRng::new(
            request
                .options
                .sampling
                .seed
                .unwrap_or(0x9E37_79B9_7F4A_7C15),
        );
        active.push(ActiveRequest {
            index,
            request_id: request.request_id,
            options: request.options,
            backend,
            deadline: request.deadline,
            cache,
            token_ids: request.input_token_ids,
            generated_token_ids: Vec::new(),
            logits: Vec::new(),
            prompt_logits,
            last_step,
            sampler_rng,
            stop_token_ids,
        });
    }

    let mut report = NativeDecoderContinuousBatchReport {
        admitted_requests: active.len().saturating_add(completed.len()),
        ..NativeDecoderContinuousBatchReport::default()
    };
    while !active.is_empty() {
        report.decode_rounds = report.decode_rounds.saturating_add(1);
        let mut next_active = Vec::with_capacity(active.len());
        for mut request in active {
            check_deadline(&request.request_id, request.deadline)?;
            let adjusted_logits = apply_native_decoder_repetition_penalty(
                &request.last_step.logits,
                &request.token_ids,
                &request.options.sampling,
            )?;
            let next_token_id = select_native_decoder_token(
                &adjusted_logits,
                &request.options.sampling,
                &mut request.sampler_rng,
            )?;
            request.logits.push(request.last_step.logits.clone());
            request.generated_token_ids.push(next_token_id);
            request.token_ids.push(next_token_id);
            report.scheduled_decode_steps = report.scheduled_decode_steps.saturating_add(1);

            let generated = request.generated_token_ids.len();
            let stop = generated >= request.options.min_new_tokens
                && request.stop_token_ids.contains(&next_token_id);
            if stop || generated >= request.options.max_new_tokens {
                completed.push((request.index, complete_active_request(request)));
            } else {
                request.last_step = native_decoder_cpu_step(
                    weights,
                    &mut request.cache,
                    next_token_id,
                    request.backend,
                    &request.options.performance,
                )?;
                next_active.push(request);
            }
        }
        active = next_active;
    }
    completed.sort_by_key(|(index, _)| *index);
    Ok((
        completed.into_iter().map(|(_, output)| output).collect(),
        report,
    ))
}

fn complete_active_request(request: ActiveRequest) -> NativeDecoderContinuousBatchOutput {
    NativeDecoderContinuousBatchOutput {
        request_id: request.request_id,
        output: NativeDecoderGenerateOutput {
            token_ids: request.token_ids,
            generated_token_ids: request.generated_token_ids,
            logits: request.logits,
            prompt_logits: request.prompt_logits,
            backend: request.backend,
            performance: NativeDecoderPerformanceReport {
                prefix_cache_miss_tokens: request.cache.position(),
                kv_cache_bytes: request.cache.resident_bytes(),
                ..NativeDecoderPerformanceReport::default()
            },
        },
    }
}

fn check_deadline(request_id: &str, deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err(RuntimeError::RequestDeadlineExceeded {
            request_id: request_id.to_string(),
        });
    }
    Ok(())
}
