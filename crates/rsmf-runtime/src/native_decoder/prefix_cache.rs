use super::*;

/// Observable native decoder performance counters for one generation call.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NativeDecoderPerformanceReport {
    /// Prompt tokens served from a resident prefix-cache entry.
    pub prefix_cache_hit_tokens: usize,
    /// Prompt tokens evaluated normally because no prefix-cache entry matched.
    pub prefix_cache_miss_tokens: usize,
    /// Number of entries retained in the session prefix cache after the run.
    pub prefix_cache_entries: usize,
    /// Resident bytes retained by the session prefix cache after the run.
    pub prefix_cache_bytes: usize,
    /// Prefix-cache entries evicted during this run.
    pub prefix_cache_evictions: usize,
    /// Resident bytes allocated by the per-run KV cache at generation end.
    pub kv_cache_bytes: usize,
}

/// Resident prefix-cache state retained by a native decoder session.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NativeDecoderPrefixCacheReport {
    /// Number of retained prefix entries.
    pub entries: usize,
    /// Resident bytes retained by those entries.
    pub resident_bytes: usize,
    /// Total prefix-cache lookups performed by this session.
    pub lookups: u64,
    /// Total prefix-cache hits performed by this session.
    pub hits: u64,
    /// Total prefix-cache insertions performed by this session.
    pub inserts: u64,
    /// Total entries evicted by this session.
    pub evictions: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct NativeDecoderPrefixCache {
    entries: Vec<NativeDecoderPrefixCacheEntry>,
    lookups: u64,
    hits: u64,
    inserts: u64,
    evictions: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeDecoderPrefixCacheEntry {
    token_ids: Vec<i64>,
    cache: NativeDecoderKvCache,
    last_step: NativeDecoderStepOutput,
    backend: NativeDecoderBackend,
    performance_key: NativeDecoderPrefixCachePerformanceKey,
    bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeDecoderPrefixCachePerformanceKey {
    kv_cache_page_size_tokens: Option<usize>,
    attention: NativeDecoderAttentionImplementation,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeDecoderPrefixCacheHit {
    pub(crate) token_count: usize,
    pub(crate) cache: NativeDecoderKvCache,
    pub(crate) last_step: NativeDecoderStepOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct NativeDecoderPrefixCacheRunStats {
    pub(crate) hit_tokens: usize,
    pub(crate) miss_tokens: usize,
    pub(crate) evictions: usize,
}

impl NativeDecoderPrefixCache {
    pub(crate) fn report(&self) -> NativeDecoderPrefixCacheReport {
        NativeDecoderPrefixCacheReport {
            entries: self.entries.len(),
            resident_bytes: self.resident_bytes(),
            lookups: self.lookups,
            hits: self.hits,
            inserts: self.inserts,
            evictions: self.evictions,
        }
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.entries
            .iter()
            .fold(0usize, |total, entry| total.saturating_add(entry.bytes))
    }

    pub(crate) fn lookup(
        &mut self,
        token_ids: &[i64],
        backend: NativeDecoderBackend,
        performance: &NativeDecoderPerformanceOptions,
    ) -> Option<NativeDecoderPrefixCacheHit> {
        self.lookups = self.lookups.saturating_add(1);
        let performance_key = NativeDecoderPrefixCachePerformanceKey::from(performance);
        let entry = self
            .entries
            .iter()
            .filter(|entry| {
                entry.backend == backend
                    && entry.performance_key == performance_key
                    && token_ids.starts_with(&entry.token_ids)
            })
            .max_by_key(|entry| entry.token_ids.len())?;
        self.hits = self.hits.saturating_add(1);
        Some(NativeDecoderPrefixCacheHit {
            token_count: entry.token_ids.len(),
            cache: entry.cache.clone(),
            last_step: entry.last_step.clone(),
        })
    }

    pub(crate) fn insert(
        &mut self,
        token_ids: &[i64],
        cache: &NativeDecoderKvCache,
        last_step: &NativeDecoderStepOutput,
        backend: NativeDecoderBackend,
        performance: &NativeDecoderPerformanceOptions,
    ) -> Result<usize> {
        if token_ids.is_empty() {
            return Ok(0);
        }
        self.inserts = self.inserts.saturating_add(1);
        let performance_key = NativeDecoderPrefixCachePerformanceKey::from(performance);
        self.entries.retain(|entry| {
            !(entry.backend == backend
                && entry.performance_key == performance_key
                && entry.token_ids == token_ids)
        });
        let cache = cache.clone();
        let bytes = prefix_cache_entry_bytes(token_ids, &cache, last_step);
        self.entries.push(NativeDecoderPrefixCacheEntry {
            token_ids: token_ids.to_vec(),
            cache,
            last_step: last_step.clone(),
            backend,
            performance_key,
            bytes,
        });
        self.enforce_limits(performance)
    }

    fn enforce_limits(&mut self, performance: &NativeDecoderPerformanceOptions) -> Result<usize> {
        let max_entries = performance.prefix_cache_max_entries.unwrap_or(usize::MAX);
        let max_bytes = performance.prefix_cache_max_bytes.unwrap_or(usize::MAX);
        if max_entries == 0 {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "prefix_cache_max_entries must be positive when set".to_string(),
            });
        }
        if max_bytes == 0 {
            return Err(RuntimeError::NativeDecoderConfigInvalid {
                reason: "prefix_cache_max_bytes must be positive when set".to_string(),
            });
        }
        let mut evicted = 0usize;
        while self.entries.len() > max_entries || self.resident_bytes() > max_bytes {
            if self.entries.is_empty() {
                break;
            }
            self.entries.remove(0);
            self.evictions = self.evictions.saturating_add(1);
            evicted = evicted.saturating_add(1);
        }
        Ok(evicted)
    }
}

impl From<&NativeDecoderPerformanceOptions> for NativeDecoderPrefixCachePerformanceKey {
    fn from(value: &NativeDecoderPerformanceOptions) -> Self {
        Self {
            kv_cache_page_size_tokens: value.kv_cache_page_size_tokens,
            attention: value.attention,
        }
    }
}

fn prefix_cache_entry_bytes(
    token_ids: &[i64],
    cache: &NativeDecoderKvCache,
    last_step: &NativeDecoderStepOutput,
) -> usize {
    token_ids
        .len()
        .saturating_mul(std::mem::size_of::<i64>())
        .saturating_add(cache.resident_bytes())
        .saturating_add(
            last_step
                .logits
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
        )
        .saturating_add(std::mem::size_of::<i64>())
}
