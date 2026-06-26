//! Reader-side index over Mixture-of-Experts tensor metadata.
//!
//! RSMF stores MoE annotations as **regular tensor metadata** under the
//! `moe.*` convention defined in `docs/CONVENTIONS.md`. No binary layout
//! change is required: readers that never call [`crate::RsmfFile::moe_experts`]
//! continue to read the file exactly as before.

use crate::error::{Result, RsmfError};
use crate::manifest::Manifest;

/// Role of a tensor inside an MoE block.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MoeRole {
    /// Router / gating tensor.
    Router,
    /// Gate projection tensor.
    Gate,
    /// Up projection tensor.
    Up,
    /// Down projection tensor.
    Down,
    /// Unknown or writer-specific role, preserved verbatim.
    Other(String),
}

impl MoeRole {
    /// Parse the `moe.role` metadata value.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "router" => Self::Router,
            "gate" => Self::Gate,
            "up" => Self::Up,
            "down" => Self::Down,
            other => Self::Other(other.to_string()),
        }
    }

    /// Canonical string form.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Router => "router",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
            Self::Other(s) => s,
        }
    }
}

/// A tensor annotated with `moe.*` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeEntry {
    /// Name of the tensor in the manifest.
    pub tensor_name: String,
    /// MoE layer index.
    pub layer: u32,
    /// Expert id within the layer, if this tensor belongs to a routed expert.
    pub expert_id: Option<u32>,
    /// True when `moe.shared = "1"` marks an always-active/shared expert.
    pub shared: bool,
    /// Role within the MoE layer.
    pub role: MoeRole,
}

/// Group of MoE tensors sharing `(layer, expert_id, shared, role)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeGroup {
    /// MoE layer index.
    pub layer: u32,
    /// Expert id within the layer, if any.
    pub expert_id: Option<u32>,
    /// True when this group represents shared expert tensors.
    pub shared: bool,
    /// Role within the MoE layer.
    pub role: MoeRole,
    /// Tensor names in manifest order.
    pub tensor_names: Vec<String>,
}

/// Index over MoE metadata in a manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoeIndex {
    /// Experts per MoE layer, from `moe.n_experts`.
    pub n_experts: Option<u32>,
    /// Active experts per token, from `moe.top_k`.
    pub top_k: Option<u32>,
    /// Shared experts per layer, from `moe.n_shared`.
    pub n_shared: Option<u32>,
    /// Architecture id, from `model.arch`.
    pub model_arch: Option<String>,
    /// Annotated tensors in manifest order.
    pub entries: Vec<MoeEntry>,
    /// Grouped view of annotated tensors.
    pub groups: Vec<MoeGroup>,
}

impl MoeIndex {
    /// Number of indexed tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no tensors carry MoE annotations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the first group matching `(layer, expert_id, role)`.
    #[must_use]
    pub fn group(&self, layer: u32, expert_id: Option<u32>, role: &MoeRole) -> Option<&MoeGroup> {
        self.groups
            .iter()
            .find(|g| g.layer == layer && g.expert_id == expert_id && &g.role == role)
    }
}

fn lookup<'a>(meta: &'a [(String, String)], key: &str) -> Option<&'a str> {
    meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn has_moe_metadata(meta: &[(String, String)]) -> bool {
    meta.iter().any(|(k, _)| k.starts_with("moe."))
}

fn parse_u32(value: &str, key: &str, scope: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .map_err(|e| RsmfError::structural(format!("{scope} has invalid {key} {value:?}: {e}")))
}

/// Build a [`MoeIndex`] from a parsed manifest.
///
/// Tensors without any `moe.*` metadata are ignored. Tensors with MoE
/// annotations must include `moe.layer`; `moe.role` defaults to an empty
/// [`MoeRole::Other`] value if absent so readers can still surface partially
/// annotated legacy files without inventing a role.
pub fn moe_index_from_manifest(manifest: &Manifest) -> Result<MoeIndex> {
    let mut out = MoeIndex {
        n_experts: lookup(&manifest.metadata, "moe.n_experts")
            .map(|s| parse_u32(s, "moe.n_experts", "manifest"))
            .transpose()?,
        top_k: lookup(&manifest.metadata, "moe.top_k")
            .map(|s| parse_u32(s, "moe.top_k", "manifest"))
            .transpose()?,
        n_shared: lookup(&manifest.metadata, "moe.n_shared")
            .map(|s| parse_u32(s, "moe.n_shared", "manifest"))
            .transpose()?,
        model_arch: lookup(&manifest.metadata, "model.arch").map(str::to_string),
        ..MoeIndex::default()
    };

    for tensor in &manifest.tensors {
        if !has_moe_metadata(&tensor.metadata) {
            continue;
        }

        let layer = lookup(&tensor.metadata, "moe.layer")
            .ok_or_else(|| {
                RsmfError::structural(format!(
                    "tensor {} has moe.* metadata but no moe.layer",
                    tensor.name
                ))
            })
            .and_then(|s| parse_u32(s, "moe.layer", &format!("tensor {}", tensor.name)))?;

        let expert_id = lookup(&tensor.metadata, "moe.expert")
            .map(|s| parse_u32(s, "moe.expert", &format!("tensor {}", tensor.name)))
            .transpose()?;

        let shared = match lookup(&tensor.metadata, "moe.shared") {
            Some("1") => true,
            Some(other) => {
                return Err(RsmfError::structural(format!(
                    "tensor {} has invalid moe.shared {other:?}; expected \"1\" when present",
                    tensor.name
                )));
            }
            None => false,
        };

        let role = lookup(&tensor.metadata, "moe.role")
            .map(MoeRole::parse)
            .unwrap_or_else(|| MoeRole::Other(String::new()));

        let entry = MoeEntry {
            tensor_name: tensor.name.clone(),
            layer,
            expert_id,
            shared,
            role,
        };

        if let Some(group) = out.groups.iter_mut().find(|g| {
            g.layer == entry.layer
                && g.expert_id == entry.expert_id
                && g.shared == entry.shared
                && g.role == entry.role
        }) {
            group.tensor_names.push(entry.tensor_name.clone());
        } else {
            out.groups.push(MoeGroup {
                layer: entry.layer,
                expert_id: entry.expert_id,
                shared: entry.shared,
                role: entry.role.clone(),
                tensor_names: vec![entry.tensor_name.clone()],
            });
        }

        out.entries.push(entry);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_roundtrips_known_values() {
        for s in ["router", "gate", "up", "down"] {
            assert_eq!(MoeRole::parse(s).as_str(), s);
        }
    }

    #[test]
    fn unknown_role_is_preserved() {
        assert!(matches!(MoeRole::parse("custom"), MoeRole::Other(s) if s == "custom"));
    }
}
