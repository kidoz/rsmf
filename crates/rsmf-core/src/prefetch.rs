//! Reader-side index over prefetch / locality metadata.
//!
//! RSMF stores prefetch hints as regular per-variant metadata under the
//! `prefetch.*` convention defined in `docs/CONVENTIONS.md`. The hints are
//! metadata-only: no tensor bytes, variant selection, or runtime scheduling
//! behavior changes unless a caller explicitly consumes this index.

use crate::error::{Result, RsmfError};
use crate::manifest::Manifest;

/// Metadata key for the opaque prefetch group id.
pub const PREFETCH_GROUP_KEY: &str = "prefetch.group";

/// Metadata key for comma-separated affinity labels.
pub const PREFETCH_AFFINITY_KEY: &str = "prefetch.affinity";

/// A tensor variant annotated with `prefetch.*` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchEntry {
    /// Name of the tensor owning the variant.
    pub tensor_name: String,
    /// Global index into `Manifest::variants`.
    pub variant_index: u32,
    /// Opaque group id; variants in the same group are candidates for
    /// speculative co-loading / co-eviction.
    pub group: String,
    /// Parsed `prefetch.affinity` tokens in declaration order.
    pub affinity: Vec<String>,
}

/// Reference to one variant inside a prefetch group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchVariantRef {
    /// Name of the tensor owning the variant.
    pub tensor_name: String,
    /// Global index into `Manifest::variants`.
    pub variant_index: u32,
}

/// Grouped view of variants sharing the same `prefetch.group`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchGroup {
    /// Opaque group id.
    pub group: String,
    /// Variants in manifest order.
    pub variants: Vec<PrefetchVariantRef>,
    /// Unique affinity tokens observed across group members, in first-seen
    /// order.
    pub affinity: Vec<String>,
}

/// Index over all prefetch hints in a manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefetchIndex {
    /// Annotated variants in manifest order.
    pub entries: Vec<PrefetchEntry>,
    /// Grouped view by `prefetch.group`.
    pub groups: Vec<PrefetchGroup>,
}

impl PrefetchIndex {
    /// Number of annotated variants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no variants carry prefetch hints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the first group with the given id.
    #[must_use]
    pub fn group(&self, group: &str) -> Option<&PrefetchGroup> {
        self.groups.iter().find(|g| g.group == group)
    }
}

/// Insert or replace `prefetch.group` in a metadata map.
pub fn set_prefetch_group(meta: &mut Vec<(String, String)>, group: impl Into<String>) {
    upsert_owned(meta, PREFETCH_GROUP_KEY, group.into());
}

/// Insert or replace `prefetch.affinity` in a metadata map.
///
/// The value is stored as a comma-separated string and parsed by
/// [`prefetch_index_from_manifest`].
pub fn set_prefetch_affinity(meta: &mut Vec<(String, String)>, affinity: impl Into<String>) {
    upsert_owned(meta, PREFETCH_AFFINITY_KEY, affinity.into());
}

/// Build a [`PrefetchIndex`] from a parsed manifest.
///
/// Variants without `prefetch.*` metadata are ignored. A variant carrying any
/// `prefetch.*` key must include a non-empty `prefetch.group`. The optional
/// `prefetch.affinity` value is parsed as comma-separated, non-empty tokens.
pub fn prefetch_index_from_manifest(manifest: &Manifest) -> Result<PrefetchIndex> {
    let mut out = PrefetchIndex::default();

    for tensor in &manifest.tensors {
        let mut variants = Vec::with_capacity(1 + tensor.packed_variants.len());
        variants.push(tensor.canonical_variant);
        variants.extend(tensor.packed_variants.iter().copied());

        for variant_index in variants {
            let variant = manifest
                .variants
                .get(variant_index as usize)
                .ok_or_else(|| {
                    RsmfError::structural(format!(
                        "tensor {} references missing variant {}",
                        tensor.name, variant_index
                    ))
                })?;
            let scope = format!("tensor {} variant {variant_index}", tensor.name);
            let Some(entry) =
                prefetch_entry_from_meta(&variant.meta.extra, &tensor.name, variant_index, &scope)?
            else {
                continue;
            };

            if let Some(group) = out.groups.iter_mut().find(|g| g.group == entry.group) {
                group.variants.push(PrefetchVariantRef {
                    tensor_name: entry.tensor_name.clone(),
                    variant_index: entry.variant_index,
                });
                for affinity in &entry.affinity {
                    if !group.affinity.contains(affinity) {
                        group.affinity.push(affinity.clone());
                    }
                }
            } else {
                out.groups.push(PrefetchGroup {
                    group: entry.group.clone(),
                    variants: vec![PrefetchVariantRef {
                        tensor_name: entry.tensor_name.clone(),
                        variant_index: entry.variant_index,
                    }],
                    affinity: entry.affinity.clone(),
                });
            }
            out.entries.push(entry);
        }
    }

    Ok(out)
}

fn prefetch_entry_from_meta(
    meta: &[(String, String)],
    tensor_name: &str,
    variant_index: u32,
    scope: &str,
) -> Result<Option<PrefetchEntry>> {
    if !has_prefetch_metadata(meta) {
        return Ok(None);
    }

    let group = lookup_unique(meta, PREFETCH_GROUP_KEY, scope)?.ok_or_else(|| {
        RsmfError::structural(format!(
            "{scope} has prefetch.* metadata but no prefetch.group"
        ))
    })?;
    let affinity = lookup_unique(meta, PREFETCH_AFFINITY_KEY, scope)?
        .map(|s| parse_affinity(s, scope))
        .transpose()?
        .unwrap_or_default();

    Ok(Some(PrefetchEntry {
        tensor_name: tensor_name.to_string(),
        variant_index,
        group: group.to_string(),
        affinity,
    }))
}

fn has_prefetch_metadata(meta: &[(String, String)]) -> bool {
    meta.iter().any(|(k, _)| k.starts_with("prefetch."))
}

fn lookup_unique<'a>(
    meta: &'a [(String, String)],
    key: &str,
    scope: &str,
) -> Result<Option<&'a str>> {
    let mut out = None;
    for (k, v) in meta {
        if k != key {
            continue;
        }
        if out.is_some() {
            return Err(RsmfError::structural(format!(
                "{scope} has duplicate {key}"
            )));
        }
        if v.is_empty() {
            return Err(RsmfError::structural(format!("{scope} has empty {key}")));
        }
        out = Some(v.as_str());
    }
    Ok(out)
}

fn parse_affinity(value: &str, scope: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for raw in value.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            return Err(RsmfError::structural(format!(
                "{scope} has empty prefetch.affinity token"
            )));
        }
        out.push(token.to_string());
    }
    Ok(out)
}

fn upsert_owned(meta: &mut Vec<(String, String)>, key: &str, value: String) {
    if let Some((_, existing)) = meta.iter_mut().find(|(k, _)| k == key) {
        *existing = value;
    } else {
        meta.push((key.to_string(), value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_affinity_tokens() {
        assert_eq!(
            parse_affinity("shard:1, expert:0:3", "variant").unwrap(),
            vec!["shard:1", "expert:0:3"]
        );
    }

    #[test]
    fn rejects_empty_affinity_token() {
        let err = parse_affinity("shard:1,,expert:0:3", "variant").unwrap_err();
        assert!(
            err.to_string().contains("empty prefetch.affinity token"),
            "unexpected error: {err}"
        );
    }
}
