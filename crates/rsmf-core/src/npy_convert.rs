//! NumPy (.npy) → RSMF conversion.
//!
//! Available only when the `npy` feature is enabled.

use ndarray::ArrayD;
use ndarray_npy::read_npy;
use std::path::Path;

use crate::error::{Result, RsmfError};
use crate::tensor::dtype::LogicalDtype;
use crate::writer::{RsmfWriter, TensorInput, VariantInput};

/// Convert a single .npy file into an [`RsmfWriter`] containing one tensor.
///
/// The tensor name defaults to the file stem.
pub fn writer_from_npy_file(path: &Path) -> Result<RsmfWriter> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tensor")
        .to_string();

    // Support F32 for now as the primary target for large embeddings.
    let array: ArrayD<f32> = read_npy(path)
        .map_err(|e| RsmfError::Unsupported(format!("Failed to read .npy file: {e}")))?;

    let shape: Vec<u64> = array.shape().iter().map(|&d| d as u64).collect();

    // RSMF canonical requires row-major.
    let standard = array.as_standard_layout().to_owned();

    // cast_slice requires &[T]
    let bytes = bytemuck::cast_slice::<f32, u8>(
        standard
            .as_slice()
            .ok_or_else(|| RsmfError::unsupported("NumPy array is not contiguous"))?,
    )
    .to_vec();

    let writer = RsmfWriter::new()
        .with_metadata("source", "npy")
        .with_tensor(TensorInput {
            shard_id: 0,
            name,
            dtype: LogicalDtype::F32,
            shape,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(bytes),
            packed: Vec::new(),
        });

    Ok(writer)
}
