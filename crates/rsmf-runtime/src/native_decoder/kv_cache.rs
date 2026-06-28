use super::*;

/// KV cache for native decoder CPU reference generation.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderKvCache {
    pub(crate) layers: Vec<NativeDecoderLayerKvCache>,
    pub(crate) position: usize,
    pub(crate) kv_width: usize,
    pub(crate) page_size_tokens: Option<usize>,
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

    /// Resident KV-cache bytes currently allocated by this cache.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.layers.iter().fold(0usize, |total, layer| {
            total.saturating_add(layer.resident_bytes())
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeDecoderLayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    key_pages: Vec<Vec<f32>>,
    value_pages: Vec<Vec<f32>>,
}

impl NativeDecoderLayerKvCache {
    fn resident_bytes(&self) -> usize {
        let flat = f32_capacity_bytes(&self.keys).saturating_add(f32_capacity_bytes(&self.values));
        let paged_keys = self.key_pages.iter().fold(0usize, |total, page| {
            total.saturating_add(f32_capacity_bytes(page))
        });
        let paged_values = self.value_pages.iter().fold(0usize, |total, page| {
            total.saturating_add(f32_capacity_bytes(page))
        });
        flat.saturating_add(paged_keys).saturating_add(paged_values)
    }
}

fn f32_capacity_bytes(values: &Vec<f32>) -> usize {
    values.capacity().saturating_mul(std::mem::size_of::<f32>())
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
    pub(crate) query: &'a [f32],
    pub(crate) cache: &'a NativeDecoderLayerKvCache,
    pub(crate) current_key: &'a [f32],
    pub(crate) current_value: &'a [f32],
    pub(crate) page_size_tokens: Option<usize>,
    pub(crate) cache_len: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) implementation: NativeDecoderAttentionImplementation,
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

    match input.implementation {
        NativeDecoderAttentionImplementation::Scalar => {
            native_decoder_cpu_layer_cached_attention_inner(input)
        }
        NativeDecoderAttentionImplementation::CpuTiled => {
            native_decoder_cpu_layer_cached_attention_tiled(input)
        }
    }
}

fn native_decoder_cpu_layer_cached_attention_inner(
    input: NativeDecoderLayerAttentionInput<'_>,
) -> Result<Vec<f32>> {
    let query_width = input.num_attention_heads * input.head_dim;
    let kv_width = input.num_key_value_heads * input.head_dim;
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

fn native_decoder_cpu_layer_cached_attention_tiled(
    input: NativeDecoderLayerAttentionInput<'_>,
) -> Result<Vec<f32>> {
    let query_width = input.num_attention_heads * input.head_dim;
    let kv_width = input.num_key_value_heads * input.head_dim;
    let groups = input.num_attention_heads / input.num_key_value_heads;
    let scale = 1.0 / (input.head_dim as f32).sqrt();
    let mut output = vec![0.0f32; query_width];
    let mut scores = vec![0.0f32; input.cache_len];
    for head in 0..input.num_attention_heads {
        scores.fill(0.0);
        let kv_head = head / groups;
        let query_offset = head * input.head_dim;
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
