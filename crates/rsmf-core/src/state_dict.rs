//! PyTorch-style state-dict inspection and schema validation.
//!
//! This module does not depend on PyTorch. It provides the same useful
//! high-level contract shape: named tensors with dtype/shape metadata, plus
//! strict validation against an expected set of keys.

use std::collections::{BTreeMap, BTreeSet};

use crate::manifest::Manifest;
use crate::tensor::dtype::LogicalDtype;

/// Owned metadata for one tensor in a state dict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDictEntry {
    /// Tensor name, unique within the RSMF manifest.
    pub name: String,
    /// Logical element dtype.
    pub dtype: LogicalDtype,
    /// Logical row-major tensor shape.
    pub shape: Vec<u64>,
    /// Canonical variant index in the manifest's variant array.
    pub canonical_variant: u32,
    /// Packed variant indexes owned by this tensor.
    pub packed_variants: Vec<u32>,
    /// Physical shard id. `0` means the master RSMF file.
    pub shard_id: u64,
    /// Tensor metadata copied from the manifest.
    pub metadata: Vec<(String, String)>,
}

/// Owned PyTorch-style map of tensor names to tensor metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDict {
    entries: BTreeMap<String, StateDictEntry>,
}

impl StateDict {
    /// Build a state dict from an RSMF manifest.
    #[must_use]
    pub fn from_manifest(manifest: &Manifest) -> Self {
        let entries = manifest
            .tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    StateDictEntry {
                        name: tensor.name.clone(),
                        dtype: tensor.dtype,
                        shape: tensor.shape.clone(),
                        canonical_variant: tensor.canonical_variant,
                        packed_variants: tensor.packed_variants.clone(),
                        shard_id: tensor.shard_id,
                        metadata: tensor.metadata.clone(),
                    },
                )
            })
            .collect();
        Self { entries }
    }

    /// Number of tensors in the state dict.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the state dict contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return a tensor entry by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&StateDictEntry> {
        self.entries.get(name)
    }

    /// Return true if a tensor name exists.
    #[must_use]
    pub fn contains_key(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Iterate tensor entries in deterministic name order.
    pub fn entries(&self) -> impl Iterator<Item = &StateDictEntry> {
        self.entries.values()
    }

    /// Iterate tensor names in deterministic order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Validate this state dict against a schema.
    #[must_use]
    pub fn validate(&self, schema: &StateDictSchema) -> StateDictValidationReport {
        let mut issues = Vec::new();
        let mut expected_names = BTreeSet::new();

        for spec in &schema.tensors {
            expected_names.insert(spec.name.clone());
            match self.entries.get(&spec.name) {
                Some(actual) => {
                    if actual.dtype != spec.dtype {
                        issues.push(StateDictValidationIssue::DtypeMismatch {
                            name: spec.name.clone(),
                            expected: spec.dtype,
                            actual: actual.dtype,
                        });
                    }
                    if actual.shape != spec.shape {
                        issues.push(StateDictValidationIssue::ShapeMismatch {
                            name: spec.name.clone(),
                            expected: spec.shape.clone(),
                            actual: actual.shape.clone(),
                        });
                    }
                }
                None if spec.required => {
                    issues.push(StateDictValidationIssue::MissingKey {
                        name: spec.name.clone(),
                    });
                }
                None => {}
            }
        }

        if schema.strict {
            for name in self.entries.keys() {
                if !expected_names.contains(name) {
                    issues.push(StateDictValidationIssue::UnexpectedKey { name: name.clone() });
                }
            }
        }

        StateDictValidationReport { issues }
    }
}

/// Expected tensor set for validating a state dict.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StateDictSchema {
    /// Expected tensor specs.
    pub tensors: Vec<TensorSpec>,
    /// Whether extra tensors should be reported as unexpected keys.
    pub strict: bool,
}

impl StateDictSchema {
    /// Create a strict schema from required tensor specs.
    #[must_use]
    pub fn strict(tensors: Vec<TensorSpec>) -> Self {
        Self {
            tensors,
            strict: true,
        }
    }

    /// Create a non-strict schema from required tensor specs.
    #[must_use]
    pub fn non_strict(tensors: Vec<TensorSpec>) -> Self {
        Self {
            tensors,
            strict: false,
        }
    }
}

/// Expected dtype and shape for one state-dict key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Expected tensor name.
    pub name: String,
    /// Expected logical dtype.
    pub dtype: LogicalDtype,
    /// Expected logical shape.
    pub shape: Vec<u64>,
    /// Whether absence of this tensor is a validation issue.
    pub required: bool,
}

impl TensorSpec {
    /// Create a required tensor spec.
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: LogicalDtype, shape: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
            required: true,
        }
    }

    /// Create an optional tensor spec.
    #[must_use]
    pub fn optional(
        name: impl Into<String>,
        dtype: LogicalDtype,
        shape: impl Into<Vec<u64>>,
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
            required: false,
        }
    }
}

/// Result of validating a state dict against a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDictValidationReport {
    /// Typed validation issues in deterministic order.
    pub issues: Vec<StateDictValidationIssue>,
}

impl StateDictValidationReport {
    /// Whether validation found no issues.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }

    /// Missing required keys.
    pub fn missing_keys(&self) -> impl Iterator<Item = &str> {
        self.issues.iter().filter_map(|issue| match issue {
            StateDictValidationIssue::MissingKey { name } => Some(name.as_str()),
            _ => None,
        })
    }

    /// Unexpected keys found when strict validation is enabled.
    pub fn unexpected_keys(&self) -> impl Iterator<Item = &str> {
        self.issues.iter().filter_map(|issue| match issue {
            StateDictValidationIssue::UnexpectedKey { name } => Some(name.as_str()),
            _ => None,
        })
    }
}

/// Typed state-dict validation issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateDictValidationIssue {
    /// A required schema key is missing from the state dict.
    MissingKey {
        /// Missing tensor name.
        name: String,
    },
    /// The state dict contains a key that is not present in a strict schema.
    UnexpectedKey {
        /// Unexpected tensor name.
        name: String,
    },
    /// A tensor exists but has a different logical dtype than expected.
    DtypeMismatch {
        /// Tensor name.
        name: String,
        /// Expected logical dtype.
        expected: LogicalDtype,
        /// Actual logical dtype.
        actual: LogicalDtype,
    },
    /// A tensor exists but has a different logical shape than expected.
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Expected logical shape.
        expected: Vec<u64>,
        /// Actual logical shape.
        actual: Vec<u64>,
    },
}
