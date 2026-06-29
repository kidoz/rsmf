//! GGUF → RSMF conversion.
//!
//! Available only when the `gguf` feature is enabled.
//!
//! ## Coverage
//!
//! Every GGUF tensor whose storage format has a matching RSMF
//! [`StorageDtype`] — including all standard llama.cpp quantization types
//! (`Q4_0`, `Q5_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`) — is
//! imported as a canonical tensor whose bytes are the **unchanged GGUF
//! payload**. The logical dtype is set to `F32` for quantized storage
//! (matching `TensorView::decode_f32`); raw floating-point and integer
//! storage is imported with its native logical dtype.
//!
//! GGUF tensor formats without an RSMF-native dequantizer (`IQ*`, `Q4_1`,
//! `Q8_1`, `Q4_2`, `Q4_3`, `Q5_1`, `Q8_K`) are imported as
//! [`StorageDtype::GgufOpaque`]. RSMF preserves their bytes and records the
//! exact source type in `gguf.storage`; consumers with their own GGUF kernels
//! can read the bytes directly through `TensorView::bytes()`.
//!
//! GGUF file-level metadata is imported under the `gguf.*` namespace in
//! the RSMF manifest so downstream tooling can recover the original
//! architecture / vocab / hyperparameter keys.
//!
//! `--from-gguf` therefore round-trips real llama.cpp `.gguf` checkpoints
//! byte-for-byte through `rsmf pack`, at the cost of storing the
//! original quantised bytes: no dequantisation happens on the write
//! path; dequantisation is deferred to
//! [`crate::tensor::view::TensorView::decode_f32`] on the read path.
use gguf_rs_lib::prelude::*;
use gguf_rs_lib::reader::file_reader::open_gguf_file;
use std::path::Path;

use crate::error::{Result, RsmfError};
use crate::tensor::dtype::{LogicalDtype, StorageDtype};
use crate::tensor::variant::{EncodingKind, LayoutTag, TargetTag, VariantMeta};
use crate::writer::{DEFAULT_CANONICAL_ALIGNMENT, RsmfWriter, TensorInput, VariantInput};

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
        let mapping = map_ggml_type(tensor_info.tensor_type())?;

        // GGUF dimensions are fastest-varying first. RSMF is row-major.
        let mut shape = tensor_info.shape().dims().to_vec();
        shape.reverse();

        // Get the data bytes.
        let data = tensor_info.data().ok_or_else(|| {
            RsmfError::GgufConversion(format!("tensor data not loaded for {}", tensor_info.name()))
        })?;
        let bytes = data.as_slice().to_vec();

        let canonical = build_canonical(&mapping, bytes);

        writer = writer.with_tensor(TensorInput {
            shard_id: 0,
            name: tensor_info.name().to_string(),
            dtype: mapping.logical_dtype,
            shape,
            metadata: vec![("gguf.storage".to_string(), mapping.storage_name.to_string())],
            canonical,
            packed: Vec::new(),
        });
    }

    Ok(writer)
}

/// Result of mapping a GGUF tensor type to an RSMF storage descriptor.
#[derive(Debug)]
struct Mapping {
    /// Logical dtype exposed to rsmf readers.
    logical_dtype: LogicalDtype,
    /// Storage dtype written to disk.
    storage_dtype: StorageDtype,
    /// Encoding kind of the canonical variant.
    encoding: EncodingKind,
    /// Layout tag of the canonical variant.
    layout: LayoutTag,
    /// Block-quantisation block shape, if any.
    block_shape: Vec<u64>,
    /// Human-readable storage name, stored in per-tensor metadata so
    /// downstream tooling can recover the original GGUF type name
    /// without re-deriving it from the storage-dtype discriminant.
    storage_name: &'static str,
    /// Optional scale dtype recorded for RSMF-native block quantizers.
    scale_dtype: Option<StorageDtype>,
}

fn build_canonical(m: &Mapping, bytes: Vec<u8>) -> VariantInput {
    VariantInput {
        target: TargetTag::Canonical,
        encoding: m.encoding,
        storage_dtype: Some(m.storage_dtype),
        layout: m.layout,
        alignment: DEFAULT_CANONICAL_ALIGNMENT,
        bytes,
        meta: VariantMeta {
            block_shape: m.block_shape.clone(),
            group_size: 0,
            scale_dtype: m.scale_dtype,
            zero_point_dtype: None,
            extra: Vec::new(),
        },
    }
}

fn map_ggml_type(ggml_type: GGUFTensorType) -> Result<Mapping> {
    // Raw / non-quantised formats: canonical stays Raw, logical dtype
    // matches storage, layout stays RowMajor.
    let raw = |logical: LogicalDtype, name: &'static str| Mapping {
        logical_dtype: logical,
        storage_dtype: StorageDtype::Logical(logical),
        encoding: EncodingKind::Raw,
        layout: LayoutTag::RowMajor,
        block_shape: Vec::new(),
        storage_name: name,
        scale_dtype: None,
    };

    // Block-quantised formats: canonical is BlockQuantized, logical is
    // F32 (matching the dequantiser target), layout is Blocked, and
    // the block size is captured in `VariantMeta::block_shape`.
    let block = |storage: StorageDtype, block: u64, name: &'static str| Mapping {
        logical_dtype: LogicalDtype::F32,
        storage_dtype: storage,
        encoding: EncodingKind::BlockQuantized,
        layout: LayoutTag::Blocked,
        block_shape: vec![block],
        storage_name: name,
        scale_dtype: Some(StorageDtype::Logical(LogicalDtype::F16)),
    };

    // Opaque GGUF formats are container-only: RSMF stores the exact bytes and
    // source tag, while external engines remain responsible for interpretation.
    let opaque = |name: &'static str| Mapping {
        logical_dtype: LogicalDtype::F32,
        storage_dtype: StorageDtype::GgufOpaque,
        encoding: EncodingKind::BlockQuantized,
        layout: LayoutTag::Blocked,
        block_shape: Vec::new(),
        storage_name: name,
        scale_dtype: None,
    };

    Ok(match ggml_type {
        GGUFTensorType::F32 => raw(LogicalDtype::F32, "f32"),
        GGUFTensorType::F16 => raw(LogicalDtype::F16, "f16"),
        GGUFTensorType::F64 => raw(LogicalDtype::F64, "f64"),
        GGUFTensorType::BF16 => raw(LogicalDtype::BF16, "bf16"),
        GGUFTensorType::I8 => raw(LogicalDtype::I8, "i8"),
        GGUFTensorType::I16 => raw(LogicalDtype::I16, "i16"),
        GGUFTensorType::I32 => raw(LogicalDtype::I32, "i32"),
        GGUFTensorType::I64 => raw(LogicalDtype::I64, "i64"),

        GGUFTensorType::Q4_0 => block(StorageDtype::Q4_0, 32, "q4_0"),
        GGUFTensorType::Q5_0 => block(StorageDtype::Q5_0, 32, "q5_0"),
        GGUFTensorType::Q8_0 => block(StorageDtype::Q8_0, 32, "q8_0"),
        GGUFTensorType::Q2_K => block(StorageDtype::Q2K, 256, "q2_k"),
        GGUFTensorType::Q3_K => block(StorageDtype::Q3K, 256, "q3_k"),
        GGUFTensorType::Q4_K => block(StorageDtype::Q4K, 256, "q4_k"),
        GGUFTensorType::Q5_K => block(StorageDtype::Q5K, 256, "q5_k"),
        GGUFTensorType::Q6_K => block(StorageDtype::Q6K, 256, "q6_k"),

        GGUFTensorType::Q4_1 => opaque("q4_1"),
        GGUFTensorType::Q4_2 => opaque("q4_2"),
        GGUFTensorType::Q4_3 => opaque("q4_3"),
        GGUFTensorType::Q5_1 => opaque("q5_1"),
        GGUFTensorType::Q8_1 => opaque("q8_1"),
        GGUFTensorType::Q8_K => opaque("q8_k"),
        GGUFTensorType::IQ2_XXS => opaque("iq2_xxs"),
        GGUFTensorType::IQ2_XS => opaque("iq2_xs"),
        GGUFTensorType::IQ3_XXS => opaque("iq3_xxs"),
        GGUFTensorType::IQ1_S => opaque("iq1_s"),
        GGUFTensorType::IQ4_NL => opaque("iq4_nl"),
        GGUFTensorType::IQ3_S => opaque("iq3_s"),
        GGUFTensorType::IQ2_S => opaque("iq2_s"),
        GGUFTensorType::IQ4_XS => opaque("iq4_xs"),
        GGUFTensorType::IQ1_M => opaque("iq1_m"),
        GGUFTensorType::IQ4_UNI => opaque("iq4_uni"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_types_keep_their_logical_dtype_and_raw_encoding() {
        for (src, expected) in [
            (GGUFTensorType::F32, LogicalDtype::F32),
            (GGUFTensorType::F16, LogicalDtype::F16),
            (GGUFTensorType::BF16, LogicalDtype::BF16),
            (GGUFTensorType::F64, LogicalDtype::F64),
            (GGUFTensorType::I8, LogicalDtype::I8),
            (GGUFTensorType::I16, LogicalDtype::I16),
            (GGUFTensorType::I32, LogicalDtype::I32),
            (GGUFTensorType::I64, LogicalDtype::I64),
        ] {
            let m = map_ggml_type(src).expect("raw mapping");
            assert_eq!(m.logical_dtype, expected, "logical dtype for {src:?}");
            assert_eq!(m.storage_dtype, StorageDtype::Logical(expected));
            assert_eq!(m.encoding, EncodingKind::Raw);
            assert_eq!(m.layout, LayoutTag::RowMajor);
            assert!(m.block_shape.is_empty());
        }
    }

    #[test]
    fn quantised_types_map_to_block_quantised_canonical() {
        let cases: &[(GGUFTensorType, StorageDtype, u64)] = &[
            (GGUFTensorType::Q4_0, StorageDtype::Q4_0, 32),
            (GGUFTensorType::Q5_0, StorageDtype::Q5_0, 32),
            (GGUFTensorType::Q8_0, StorageDtype::Q8_0, 32),
            (GGUFTensorType::Q2_K, StorageDtype::Q2K, 256),
            (GGUFTensorType::Q3_K, StorageDtype::Q3K, 256),
            (GGUFTensorType::Q4_K, StorageDtype::Q4K, 256),
            (GGUFTensorType::Q5_K, StorageDtype::Q5K, 256),
            (GGUFTensorType::Q6_K, StorageDtype::Q6K, 256),
        ];
        for &(src, storage, block) in cases {
            let m = map_ggml_type(src).expect("quantised mapping");
            assert_eq!(m.logical_dtype, LogicalDtype::F32);
            assert_eq!(m.storage_dtype, storage);
            assert_eq!(m.encoding, EncodingKind::BlockQuantized);
            assert_eq!(m.layout, LayoutTag::Blocked);
            assert_eq!(m.block_shape, vec![block]);
        }
    }

    #[test]
    fn gguf_only_quantized_types_map_to_opaque_passthrough() {
        let cases = [
            (GGUFTensorType::Q4_1, "q4_1"),
            (GGUFTensorType::Q4_2, "q4_2"),
            (GGUFTensorType::Q4_3, "q4_3"),
            (GGUFTensorType::Q5_1, "q5_1"),
            (GGUFTensorType::Q8_1, "q8_1"),
            (GGUFTensorType::Q8_K, "q8_k"),
            (GGUFTensorType::IQ2_XXS, "iq2_xxs"),
            (GGUFTensorType::IQ2_XS, "iq2_xs"),
            (GGUFTensorType::IQ3_XXS, "iq3_xxs"),
            (GGUFTensorType::IQ1_S, "iq1_s"),
            (GGUFTensorType::IQ4_NL, "iq4_nl"),
            (GGUFTensorType::IQ3_S, "iq3_s"),
            (GGUFTensorType::IQ2_S, "iq2_s"),
            (GGUFTensorType::IQ4_XS, "iq4_xs"),
            (GGUFTensorType::IQ1_M, "iq1_m"),
            (GGUFTensorType::IQ4_UNI, "iq4_uni"),
        ];

        for (src, name) in cases {
            let mapping = map_ggml_type(src).expect("opaque mapping");
            assert_eq!(mapping.logical_dtype, LogicalDtype::F32);
            assert_eq!(mapping.storage_dtype, StorageDtype::GgufOpaque);
            assert_eq!(mapping.encoding, EncodingKind::BlockQuantized);
            assert_eq!(mapping.layout, LayoutTag::Blocked);
            assert_eq!(mapping.storage_name, name);
            assert_eq!(mapping.scale_dtype, None);
            assert!(mapping.block_shape.is_empty());
        }
    }
}
