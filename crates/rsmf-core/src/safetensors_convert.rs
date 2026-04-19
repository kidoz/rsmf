//! Safetensors → RSMF conversion.
//!
//! Available only when the `safetensors` feature is enabled (on by default).
//! The converter preserves:
//!
//! * tensor names,
//! * shapes,
//! * logical dtypes (through the exhaustive [`map_dtype`] mapper),
//! * and raw bytes.
//!
//! Global safetensors metadata is imported under the `safetensors.` namespace.

use safetensors::tensor::Dtype as StDtype;

use crate::error::{Result, RsmfError};
use crate::tensor::dtype::LogicalDtype;
use crate::writer::{RsmfWriter, TensorInput, VariantInput};

/// Build an [`RsmfWriter`] from the bytes of a `.safetensors` file.
///
/// This does a single pass over the safetensors archive and populates the
/// writer with every tensor it finds. Every tensor is added as a canonical
/// variant with no packed variants.
pub fn writer_from_safetensors_bytes(bytes: &[u8]) -> Result<RsmfWriter> {
    let st = safetensors::tensor::SafeTensors::deserialize(bytes)
        .map_err(|e| RsmfError::SafetensorsConversion(format!("{e}")))?;

    let mut writer = RsmfWriter::new().with_metadata("source", "safetensors");

    // Import safetensors metadata.
    if let Ok((_, metadata)) = safetensors::tensor::SafeTensors::read_metadata(bytes) {
        if let Some(map) = metadata.metadata() {
            for (k, v) in map {
                writer = writer.with_metadata(format!("safetensors.{k}"), v.clone());
            }
        }
    }

    for name in st.names() {
        let tv = st
            .tensor(name)
            .map_err(|e| RsmfError::SafetensorsConversion(format!("{e}")))?;
        let dtype = map_dtype(tv.dtype())?;
        let shape: Vec<u64> = tv.shape().iter().map(|&d| d as u64).collect();
        let tensor_bytes = tv.data().to_vec();

        writer = writer.with_tensor(TensorInput {
            shard_id: 0,
            name: (*name).clone(),
            dtype,
            shape,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(tensor_bytes),
            packed: Vec::new(),
        });
    }

    Ok(writer)
}

/// Map a safetensors [`Dtype`](safetensors::tensor::Dtype) into our
/// [`LogicalDtype`].
pub fn map_dtype(dtype: StDtype) -> Result<LogicalDtype> {
    Ok(match dtype {
        StDtype::BOOL => LogicalDtype::Bool,
        StDtype::U8 => LogicalDtype::U8,
        StDtype::I8 => LogicalDtype::I8,
        StDtype::U16 => LogicalDtype::U16,
        StDtype::I16 => LogicalDtype::I16,
        StDtype::U32 => LogicalDtype::U32,
        StDtype::I32 => LogicalDtype::I32,
        StDtype::F16 => LogicalDtype::F16,
        StDtype::BF16 => LogicalDtype::BF16,
        StDtype::F32 => LogicalDtype::F32,
        StDtype::F64 => LogicalDtype::F64,
        StDtype::I64 => LogicalDtype::I64,
        StDtype::U64 => LogicalDtype::U64,
        other => {
            return Err(RsmfError::SafetensorsConversion(format!(
                "unsupported safetensors dtype {other:?}"
            )));
        }
    })
}
