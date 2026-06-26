//! Tier metadata convention helpers.
//!
//! Variant tier intent is stored in [`crate::tensor::variant::VariantMeta::extra`]
//! under `tier.intent`. It is metadata-only: the binary variant layout does not
//! change.

use crate::error::{Result, RsmfError};

/// Metadata key for the intended residency tier of a variant.
pub const TIER_INTENT_KEY: &str = "tier.intent";

/// Metadata key for an optional hot/warm/cold class label.
pub const TIER_CLASS_KEY: &str = "tier.class";

/// Intended residency tier for a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Device VRAM.
    Vram,
    /// Host RAM.
    Ram,
    /// NVMe / SSD-backed tier.
    Nvme,
}

impl Tier {
    /// Parse a stable metadata / CLI tier name.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "vram" => Self::Vram,
            "ram" => Self::Ram,
            "nvme" => Self::Nvme,
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown tier intent {other:?}; expected vram, ram, or nvme"
                )));
            }
        })
    }

    /// Stable metadata / CLI tier name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Vram => "vram",
            Self::Ram => "ram",
            Self::Nvme => "nvme",
        }
    }
}

/// Return the `tier.intent` value from a metadata map.
pub fn tier_intent_from_meta(meta: &[(String, String)]) -> Result<Option<Tier>> {
    let mut out = None;
    for (k, v) in meta {
        if k != TIER_INTENT_KEY {
            continue;
        }
        if out.is_some() {
            return Err(RsmfError::structural(format!(
                "duplicate {TIER_INTENT_KEY} metadata"
            )));
        }
        out = Some(Tier::parse(v)?);
    }
    Ok(out)
}

/// Return the optional `tier.class` value from a metadata map.
pub fn tier_class_from_meta(meta: &[(String, String)]) -> Result<Option<&str>> {
    let mut out = None;
    for (k, v) in meta {
        if k != TIER_CLASS_KEY {
            continue;
        }
        if out.is_some() {
            return Err(RsmfError::structural(format!(
                "duplicate {TIER_CLASS_KEY} metadata"
            )));
        }
        if v.is_empty() {
            return Err(RsmfError::structural(format!(
                "{TIER_CLASS_KEY} metadata must not be empty"
            )));
        }
        out = Some(v.as_str());
    }
    Ok(out)
}

/// Insert or replace `tier.intent` in a metadata map.
pub fn set_tier_intent(meta: &mut Vec<(String, String)>, tier: Tier) {
    upsert(meta, TIER_INTENT_KEY, tier.name());
}

/// Insert or replace `tier.class` in a metadata map.
pub fn set_tier_class(meta: &mut Vec<(String, String)>, class: impl Into<String>) {
    upsert_owned(meta, TIER_CLASS_KEY, class.into());
}

/// Validate tier metadata on one variant.
pub fn validate_tier_metadata(meta: &[(String, String)]) -> Result<()> {
    tier_intent_from_meta(meta)?;
    tier_class_from_meta(meta)?;
    Ok(())
}

fn upsert(meta: &mut Vec<(String, String)>, key: &str, value: &str) {
    upsert_owned(meta, key, value.to_string());
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
    fn parses_known_tier_names() {
        assert_eq!(Tier::parse("vram").unwrap(), Tier::Vram);
        assert_eq!(Tier::parse("ram").unwrap(), Tier::Ram);
        assert_eq!(Tier::parse("nvme").unwrap(), Tier::Nvme);
    }

    #[test]
    fn rejects_unknown_tier_names() {
        let err = Tier::parse("hbm").unwrap_err();
        assert!(
            err.to_string().contains("unknown tier intent"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn upserts_tier_intent() {
        let mut meta = vec![("other".into(), "1".into())];
        set_tier_intent(&mut meta, Tier::Nvme);
        assert_eq!(tier_intent_from_meta(&meta).unwrap(), Some(Tier::Nvme));
        set_tier_intent(&mut meta, Tier::Vram);
        assert_eq!(tier_intent_from_meta(&meta).unwrap(), Some(Tier::Vram));
    }
}
