use super::*;

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
