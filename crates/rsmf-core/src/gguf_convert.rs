//! GGUF → RSMF conversion.
//!
//! Available only when the `gguf` feature is enabled.

use gguf_rs_lib::prelude::*;
use gguf_rs_lib::reader::file_reader::open_gguf_file;
use std::path::Path;

use crate::error::{Result, RsmfError};
use crate::tensor::dtype::LogicalDtype;
use crate::writer::{RsmfWriter, TensorInput, VariantInput};

/// Convert a GGUF file to an [`RsmfWriter`] with every supported tensor attached.
///
/// GGUF metadata is imported as `gguf.<key>` entries in the RSMF manifest.
pub fn writer_from_gguf_file(path: &Path) -> Result<RsmfWriter> {
    // Open the GGUF file.
    let mut reader = open_gguf_file(path)
        .map_err(|e| RsmfError::GgufConversion(format!("open_gguf_file: {e}")))?;

    // Load all tensor data into memory for conversion.
    reader
        .load_all_tensor_data()
        .map_err(|e| RsmfError::GgufConversion(format!("load_all_tensor_data: {e}")))?;

    let mut writer = RsmfWriter::new().with_metadata("source", "gguf");

    // Import GGUF metadata under the `gguf.*` namespace.
    for (k, v) in reader.metadata().iter() {
        let val_str = match v {
            MetadataValue::U8(v) => v.to_string(),
            MetadataValue::I8(v) => v.to_string(),
            MetadataValue::U16(v) => v.to_string(),
            MetadataValue::I16(v) => v.to_string(),
            MetadataValue::U32(v) => v.to_string(),
            MetadataValue::I32(v) => v.to_string(),
            MetadataValue::F32(v) => v.to_string(),
            MetadataValue::U64(v) => v.to_string(),
            MetadataValue::I64(v) => v.to_string(),
            MetadataValue::F64(v) => v.to_string(),
            MetadataValue::Bool(v) => v.to_string(),
            MetadataValue::String(v) => v.clone(),
            MetadataValue::Array(_) => "[array]".to_string(),
        };
        writer = writer.with_metadata(format!("gguf.{k}"), val_str);
    }

    for tensor_info in reader.tensor_infos() {
        let dtype = map_ggml_type(tensor_info.tensor_type())?;

        // GGUF dimensions are fastest-varying first. RSMF is row-major.
        let mut shape = tensor_info.shape().dims().to_vec();
        shape.reverse();

        // Get the data bytes.
        let data = tensor_info.data().ok_or_else(|| {
            RsmfError::GgufConversion(format!("tensor data not loaded for {}", tensor_info.name()))
        })?;
        let bytes = data.as_slice().to_vec();

        writer = writer.with_tensor(TensorInput {
            shard_id: 0,
            name: tensor_info.name().to_string(),
            dtype,
            shape,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(bytes),
            packed: Vec::new(),
        });
    }

    Ok(writer)
}

fn map_ggml_type(ggml_type: GGUFTensorType) -> Result<LogicalDtype> {
    Ok(match ggml_type {
        GGUFTensorType::F32 => LogicalDtype::F32,
        GGUFTensorType::F16 => LogicalDtype::F16,
        GGUFTensorType::I8 => LogicalDtype::I8,
        GGUFTensorType::I16 => LogicalDtype::I16,
        GGUFTensorType::I32 => LogicalDtype::I32,
        other => {
            return Err(RsmfError::GgufConversion(format!(
                "unsupported GGML type {other:?}"
            )));
        }
    })
}
