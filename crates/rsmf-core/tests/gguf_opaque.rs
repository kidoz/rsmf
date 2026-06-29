use std::io::Write;

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{
    Capabilities, EncodingKind, GpuBackend, LayoutTag, LogicalDtype, RsmfError, RsmfFile,
    StorageDtype, TargetTag, VariantMeta,
};
use tempfile::NamedTempFile;

#[test]
fn opaque_gguf_tensor_preserves_bytes_and_storage_tag() {
    let source_bytes = vec![
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xF0,
        0x12,
    ];
    let file = open_opaque_file(source_bytes.clone(), 0);

    let view = file.tensor_view("weight").unwrap();
    assert_eq!(view.storage_dtype, StorageDtype::GgufOpaque);
    assert_eq!(view.encoding, EncodingKind::BlockQuantized);
    assert_eq!(view.layout, LayoutTag::Blocked);
    assert_eq!(view.bytes(), source_bytes.as_slice());
    assert!(matches!(view.decode_f32(), Err(RsmfError::Unsupported(_))));

    let entries = file.tensor_entries().unwrap();
    assert_eq!(entries.len(), 1);
    let entry = &entries[0];
    assert_eq!(entry.name, "weight");
    assert_eq!(entry.shape, &[4]);
    assert_eq!(entry.dtype, LogicalDtype::F32);
    assert_eq!(entry.storage_dtype, StorageDtype::GgufOpaque);
    assert_eq!(entry.variant_index, 0);
    assert_eq!(entry.target, TargetTag::Canonical);
    assert_eq!(entry.gguf_storage, Some("iq4_xs"));
    assert_eq!(entry.bytes, source_bytes.as_slice());
}

#[test]
fn tensor_entries_for_plan_resolves_selected_variant_bytes() {
    let canonical = 1.0f32.to_le_bytes().to_vec();
    let packed = 2.0f32.to_le_bytes().to_vec();
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "weight".into(),
        dtype: LogicalDtype::F32,
        shape: vec![1],
        metadata: vec![("gguf.storage".into(), "f32".into())],
        canonical: VariantInput::canonical_raw(canonical),
        packed: vec![VariantInput {
            target: TargetTag::Wgpu,
            encoding: EncodingKind::Raw,
            storage_dtype: Some(StorageDtype::Logical(LogicalDtype::F32)),
            layout: LayoutTag::RowMajor,
            alignment: 64,
            bytes: packed.clone(),
            meta: VariantMeta::default(),
        }],
    });

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&writer.write_to_bytes().unwrap()).unwrap();
    tmp.flush().unwrap();
    let file = RsmfFile::open(tmp.path()).unwrap();
    let plan = file
        .select_variants(
            rsmf_core::ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(Some(GpuBackend::Wgpu)),
        )
        .unwrap();

    let entries = file.tensor_entries_for_plan(&plan).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].target, TargetTag::Wgpu);
    assert_eq!(entries[0].variant_index, 1);
    assert_eq!(entries[0].bytes, packed.as_slice());
    assert_eq!(entries[0].gguf_storage, Some("f32"));
}

#[test]
fn tensor_entries_propagates_missing_shard_error() {
    let source_bytes = vec![
        0xAA, 0xBB, 0xCC, 0xDD, 0x01, 0x02, 0x03, 0x04, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70,
        0x80,
    ];
    let file = open_opaque_file(source_bytes, 7);

    let err = file.tensor_entries().unwrap_err();
    assert!(matches!(err, RsmfError::Unsupported(message) if message.contains("shard 7")));
}

fn open_opaque_file(source_bytes: Vec<u8>, shard_id: u64) -> RsmfFile {
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id,
        name: "weight".into(),
        dtype: LogicalDtype::F32,
        shape: vec![(source_bytes.len() / LogicalDtype::F32.size_bytes()) as u64],
        metadata: vec![("gguf.storage".into(), "iq4_xs".into())],
        canonical: VariantInput {
            target: TargetTag::Canonical,
            encoding: EncodingKind::BlockQuantized,
            storage_dtype: Some(StorageDtype::GgufOpaque),
            layout: LayoutTag::Blocked,
            alignment: 64,
            bytes: source_bytes,
            meta: VariantMeta::default(),
        },
        packed: Vec::new(),
    });

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&writer.write_to_bytes().unwrap()).unwrap();
    tmp.flush().unwrap();
    RsmfFile::open(tmp.path()).unwrap()
}
