//! Reader-side index over LoRA / DoRA / IA³-style adapter tensors.
//!
//! RSMF stores adapters as **regular tensors** annotated with the `adapter.*`
//! metadata convention defined in `docs/CONVENTIONS.md`. No dedicated section
//! or custom codec is required: any writer that follows the convention
//! produces a file that streams, compresses, verifies, and round-trips through
//! `rsmf rewrite` unchanged.
//!
//! This module walks a parsed [`Manifest`] and groups the annotated tensors
//! into [`Adapter`] structs so consumers can resolve a delta like
//! `W + (α/r) · B · A` without parsing metadata strings themselves.
//!
//! Unknown `adapter.kind` and `adapter.role` values are preserved as
//! [`AdapterKind::Other`] / [`AdapterRole::Other`] rather than rejected, per
//! the convention-level unknown-value rule.
//!
//! See `docs/CONVENTIONS.md` → "Adapter-level" for the full key list.

use crate::error::{Result, RsmfError};
use crate::manifest::Manifest;

/// Adapter family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterKind {
    /// Classic LoRA: delta = (α/r) · B · A.
    Lora,
    /// LoRA+ (different learning rate for A vs B, but same delta formula).
    LoraPlus,
    /// DoRA: weight-decomposed LoRA, adds a magnitude vector.
    Dora,
    /// IA³: per-channel multiplicative scales.
    Ia3,
    /// Unknown or writer-specific family — preserved verbatim.
    Other(String),
}

impl AdapterKind {
    /// Parse the `adapter.kind` metadata value.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "lora" => Self::Lora,
            "lora_plus" => Self::LoraPlus,
            "dora" => Self::Dora,
            "ia3" => Self::Ia3,
            other => Self::Other(other.to_string()),
        }
    }

    /// Canonical string form.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Lora => "lora",
            Self::LoraPlus => "lora_plus",
            Self::Dora => "dora",
            Self::Ia3 => "ia3",
            Self::Other(s) => s,
        }
    }
}

/// Role of a tensor inside an adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterRole {
    /// LoRA `A` matrix: shape `[rank, in_features]`.
    LoraA,
    /// LoRA `B` matrix: shape `[out_features, rank]`.
    LoraB,
    /// DoRA magnitude vector: shape `[out_features]`.
    Magnitude,
    /// IA³ scale vector (multiplicative).
    Scale,
    /// Base weight carried alongside the adapter (e.g. merged model).
    BaseWeight,
    /// Unknown or writer-specific role.
    Other(String),
}

impl AdapterRole {
    /// Parse the `adapter.role` metadata value.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "lora_a" => Self::LoraA,
            "lora_b" => Self::LoraB,
            "magnitude" => Self::Magnitude,
            "scale" => Self::Scale,
            "base_weight" => Self::BaseWeight,
            other => Self::Other(other.to_string()),
        }
    }

    /// Canonical string form.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::LoraA => "lora_a",
            Self::LoraB => "lora_b",
            Self::Magnitude => "magnitude",
            Self::Scale => "scale",
            Self::BaseWeight => "base_weight",
            Self::Other(s) => s,
        }
    }
}

/// A single tensor that participates in an adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterEntry {
    /// Name of the tensor in the manifest.
    pub tensor_name: String,
    /// Role within the adapter.
    pub role: AdapterRole,
    /// Target tensor this adapter modifies, if specified
    /// (`adapter.target` — e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub target: Option<String>,
}

/// A named adapter grouped from manifest tensors sharing `adapter.name`.
#[derive(Debug, Clone, PartialEq)]
pub struct Adapter {
    /// Adapter name (from `adapter.name`).
    pub name: String,
    /// Adapter family.
    pub kind: AdapterKind,
    /// Rank `r` if specified on any entry (consistent across entries).
    pub rank: Option<u32>,
    /// Scaling factor α if specified on any entry.
    pub alpha: Option<f32>,
    /// Tensors making up this adapter, in manifest order.
    pub entries: Vec<AdapterEntry>,
}

impl Adapter {
    /// Return the first entry whose role matches.
    #[must_use]
    pub fn entry_for_role(&self, role: &AdapterRole) -> Option<&AdapterEntry> {
        self.entries.iter().find(|e| &e.role == role)
    }

    /// Effective scale `α / r`, if both are set. Returns `None` otherwise.
    #[must_use]
    pub fn effective_scale(&self) -> Option<f32> {
        match (self.alpha, self.rank) {
            (Some(a), Some(r)) if r > 0 => Some(a / r as f32),
            _ => None,
        }
    }
}

/// Index over all adapters in a manifest.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdapterIndex {
    /// Adapters in the order of first occurrence in the manifest.
    pub adapters: Vec<Adapter>,
    /// Optional base-model name from `adapter.base_model_name`.
    pub base_model_name: Option<String>,
    /// Optional base-model hash from `adapter.base_model_sha256`.
    pub base_model_sha256: Option<String>,
}

impl AdapterIndex {
    /// Look up an adapter by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Adapter> {
        self.adapters.iter().find(|a| a.name == name)
    }

    /// Number of indexed adapters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// True when no adapters are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

fn lookup<'a>(meta: &'a [(String, String)], key: &str) -> Option<&'a str> {
    meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// Build an [`AdapterIndex`] from a parsed manifest.
///
/// Tensors without an `adapter.name` key are ignored. Tensors that share an
/// `adapter.name` are grouped; their `adapter.kind`, `adapter.rank`, and
/// `adapter.alpha` values must agree, otherwise this returns
/// [`RsmfError::Structural`]. `adapter.kind` defaults to `lora` when absent.
pub fn adapter_index_from_manifest(manifest: &Manifest) -> Result<AdapterIndex> {
    let mut out = AdapterIndex {
        base_model_name: lookup(&manifest.metadata, "adapter.base_model_name").map(str::to_string),
        base_model_sha256: lookup(&manifest.metadata, "adapter.base_model_sha256")
            .map(str::to_string),
        ..AdapterIndex::default()
    };

    for tensor in &manifest.tensors {
        let Some(name) = lookup(&tensor.metadata, "adapter.name") else {
            continue;
        };

        let role = lookup(&tensor.metadata, "adapter.role")
            .map(AdapterRole::parse)
            .unwrap_or_else(|| AdapterRole::Other(String::new()));
        let target = lookup(&tensor.metadata, "adapter.target").map(str::to_string);

        let kind = lookup(&tensor.metadata, "adapter.kind")
            .map(AdapterKind::parse)
            .unwrap_or(AdapterKind::Lora);

        let rank = lookup(&tensor.metadata, "adapter.rank")
            .map(|s| {
                s.parse::<u32>().map_err(|e| {
                    RsmfError::structural(format!(
                        "tensor {} has invalid adapter.rank {s:?}: {e}",
                        tensor.name
                    ))
                })
            })
            .transpose()?;

        let alpha = lookup(&tensor.metadata, "adapter.alpha")
            .map(|s| {
                s.parse::<f32>().map_err(|e| {
                    RsmfError::structural(format!(
                        "tensor {} has invalid adapter.alpha {s:?}: {e}",
                        tensor.name
                    ))
                })
            })
            .transpose()?;

        let entry = AdapterEntry {
            tensor_name: tensor.name.clone(),
            role,
            target,
        };

        if let Some(existing) = out.adapters.iter_mut().find(|a| a.name == name) {
            if existing.kind != kind {
                return Err(RsmfError::structural(format!(
                    "adapter {name}: tensor {} declares kind={} but earlier entries declared kind={}",
                    tensor.name,
                    kind.as_str(),
                    existing.kind.as_str()
                )));
            }
            if let Some(r) = rank {
                match existing.rank {
                    Some(existing_r) if existing_r != r => {
                        return Err(RsmfError::structural(format!(
                            "adapter {name}: tensor {} declares rank={r} but earlier entries declared rank={existing_r}",
                            tensor.name,
                        )));
                    }
                    _ => existing.rank = Some(r),
                }
            }
            if let Some(a) = alpha {
                match existing.alpha {
                    Some(existing_a) if (existing_a - a).abs() > f32::EPSILON => {
                        return Err(RsmfError::structural(format!(
                            "adapter {name}: tensor {} declares alpha={a} but earlier entries declared alpha={existing_a}",
                            tensor.name,
                        )));
                    }
                    _ => existing.alpha = Some(a),
                }
            }
            existing.entries.push(entry);
        } else {
            out.adapters.push(Adapter {
                name: name.to_string(),
                kind,
                rank,
                alpha,
                entries: vec![entry],
            });
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrips_known_values() {
        for s in ["lora", "lora_plus", "dora", "ia3"] {
            assert_eq!(AdapterKind::parse(s).as_str(), s);
        }
        for s in ["lora_a", "lora_b", "magnitude", "scale", "base_weight"] {
            assert_eq!(AdapterRole::parse(s).as_str(), s);
        }
    }

    #[test]
    fn unknown_values_round_trip_as_other() {
        assert!(matches!(AdapterKind::parse("mystery"), AdapterKind::Other(s) if s == "mystery"));
        assert!(
            matches!(AdapterRole::parse("head_bias"), AdapterRole::Other(s) if s == "head_bias")
        );
    }

    #[test]
    fn effective_scale_requires_both() {
        let a = Adapter {
            name: "x".into(),
            kind: AdapterKind::Lora,
            rank: Some(8),
            alpha: Some(16.0),
            entries: vec![],
        };
        assert_eq!(a.effective_scale(), Some(2.0));

        let b = Adapter {
            rank: None,
            ..a.clone()
        };
        assert_eq!(b.effective_scale(), None);
    }
}
