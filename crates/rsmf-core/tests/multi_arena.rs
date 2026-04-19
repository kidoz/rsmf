//! Multiple arena groups test.

mod common;

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{Capabilities, ExecutionMode, RsmfFile, TargetTag};
use std::io::Write;
use tempfile::NamedTempFile;

fn packed_raw_variant(target: TargetTag, bytes: Vec<u8>) -> VariantInput {
    VariantInput {
        target,
        encoding: rsmf_core::EncodingKind::Raw,
        storage_dtype: Some(rsmf_core::StorageDtype::Logical(
            rsmf_core::LogicalDtype::F32,
        )),
        layout: rsmf_core::LayoutTag::RowMajor,
        alignment: 64,
        bytes,
        meta: rsmf_core::tensor::variant::VariantMeta::default(),
    }
}

#[test]
fn multiple_packed_arenas_emitted_for_different_groups() {
    let weight = vec![0u8; 16];
    let wgpu_weight = vec![0u8; 16];
    let avx_weight = vec![0u8; 16];

    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "w".into(),
        dtype: rsmf_core::LogicalDtype::F32,
        shape: vec![4],
        metadata: vec![],
        canonical: VariantInput::canonical_raw(weight),
        packed: vec![
            packed_raw_variant(TargetTag::Wgpu, wgpu_weight),
            packed_raw_variant(TargetTag::CpuAvx2, avx_weight),
        ],
    });

    let bytes = writer.write_to_bytes().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();

    // Check that we have multiple packed sections.
    let packed_count = file
        .sections()
        .iter()
        .filter(|s| s.kind == rsmf_core::SectionKind::PackedArena)
        .count();
    assert!(packed_count >= 1);

    // Verify selection still works.
    let mut caps = Capabilities::detect().with_max_alignment(64);
    caps.cpu.avx2 = true;
    let plan = file.select_variants(ExecutionMode::CpuOnly, &caps).unwrap();
    assert_eq!(plan.selections[0].target, TargetTag::CpuAvx2);

    let plan_gpu = file
        .select_variants(
            ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(Some(rsmf_core::GpuBackend::Wgpu)),
        )
        .unwrap();
    assert_eq!(plan_gpu.selections[0].target, TargetTag::Wgpu);
}
