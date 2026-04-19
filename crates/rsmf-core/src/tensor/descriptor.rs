//! Tensor descriptor.

use crate::tensor::dtype::LogicalDtype;

/// Parsed tensor descriptor (owning).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Tensor name. Unique within the file.
    pub name: String,
    /// Logical element type.
    pub dtype: LogicalDtype,
    /// Logical shape. Row-major; first dimension is the outermost.
    pub shape: Vec<u64>,
    /// Index of the canonical variant in the manifest's variants array.
    pub canonical_variant: u32,
    /// Indices of the packed variants, if any.
    pub packed_variants: Vec<u32>,
    /// Physical file index for this tensor. `0` for the current master file.
    pub shard_id: u64,
    /// Freeform metadata.
    pub metadata: Vec<(String, String)>,
}

impl TensorDescriptor {
    /// Total number of elements (product of shape dimensions).
    ///
    /// Returns `None` if the product would overflow `u64`.
    #[must_use]
    pub fn element_count(&self) -> Option<u64> {
        let mut acc: u64 = 1;
        for &d in &self.shape {
            acc = acc.checked_mul(d)?;
        }
        Some(acc)
    }

    /// Total number of bytes for the canonical raw layout.
    #[must_use]
    pub fn canonical_bytes(&self) -> Option<u64> {
        let elems = self.element_count()?;
        elems.checked_mul(self.dtype.size_bytes() as u64)
    }
}
