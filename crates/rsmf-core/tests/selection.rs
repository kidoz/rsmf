//! Test requirement #5: variant selection across CpuOnly / GpuOnly /
//! HybridAuto, including canonical fallback.

mod common;

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput, convert_f32_to_f16_bytes};
use rsmf_core::{
    Capabilities, EncodingKind, ExecutionMode, GpuBackend, LayoutTag, LogicalDtype, RsmfFile,
    StorageDtype, TargetTag, Tier, VariantMeta,
};
use std::io::Write;
use tempfile::NamedTempFile;

fn open_fixture() -> (NamedTempFile, RsmfFile) {
    let bytes = common::build_basic_file_bytes();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let file = RsmfFile::open(tmp.path()).unwrap();
    (tmp, file)
}

#[test]
fn cpu_only_falls_back_to_canonical_when_no_cpu_variant() {
    let (_tmp, file) = open_fixture();
    let plan = file
        .select_variants(
            ExecutionMode::CpuOnly,
            &Capabilities::detect().with_gpu(None),
        )
        .unwrap();
    for sel in &plan.selections {
        // Fixture has no explicit CpuGeneric variant, so canonical wins.
        assert_eq!(sel.target, TargetTag::Canonical, "for {}", sel.tensor_name);
    }
}

#[test]
fn gpu_only_prefers_wgpu_variant_when_present() {
    let (_tmp, file) = open_fixture();
    let plan = file
        .select_variants(
            ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(Some(GpuBackend::Wgpu)),
        )
        .unwrap();
    let weight_sel = plan
        .selections
        .iter()
        .find(|s| s.tensor_name == "weight")
        .unwrap();
    assert_eq!(weight_sel.target, TargetTag::Wgpu);

    let bias_sel = plan
        .selections
        .iter()
        .find(|s| s.tensor_name == "bias")
        .unwrap();
    // Bias has no wgpu variant, so it falls back to canonical.
    assert_eq!(bias_sel.target, TargetTag::Canonical);
}

#[test]
fn gpu_only_falls_back_when_no_gpu_available() {
    let (_tmp, file) = open_fixture();
    let plan = file
        .select_variants(
            ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(None),
        )
        .unwrap();
    for sel in &plan.selections {
        assert_eq!(sel.target, TargetTag::Canonical);
    }
}

#[test]
fn hybrid_auto_picks_wgpu_when_available() {
    let (_tmp, file) = open_fixture();
    let plan = file
        .select_variants(
            ExecutionMode::HybridAuto,
            &Capabilities::detect().with_gpu(Some(GpuBackend::Wgpu)),
        )
        .unwrap();
    let weight_sel = plan
        .selections
        .iter()
        .find(|s| s.tensor_name == "weight")
        .unwrap();
    assert_eq!(weight_sel.target, TargetTag::Wgpu);
}

fn open_tier_fixture() -> (NamedTempFile, RsmfFile) {
    let canonical: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let f16 = convert_f32_to_f16_bytes(&canonical);
    let nvme_q4 = vec![0u8; 18];

    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "experts.0.ffn.up".into(),
        dtype: LogicalDtype::F32,
        shape: vec![4],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(canonical),
        packed: vec![
            VariantInput::packed_cast_f16(TargetTag::Wgpu, f16)
                .with_tier_intent(Tier::Vram)
                .with_tier_class("hot"),
            VariantInput::packed_q4_0(TargetTag::Wgpu, nvme_q4)
                .with_tier_intent(Tier::Nvme)
                .with_tier_class("cold"),
        ],
    });

    let bytes = writer.write_to_bytes().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let file = RsmfFile::open(tmp.path()).unwrap();
    (tmp, file)
}

#[test]
fn gpu_selection_without_tier_keeps_existing_backend_preference() {
    let (_tmp, file) = open_tier_fixture();
    let plan = file
        .select_variants(
            ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(Some(GpuBackend::Wgpu)),
        )
        .unwrap();
    let selected = &plan.selections[0];
    assert_eq!(selected.target, TargetTag::Wgpu);
    assert_eq!(selected.encoding, EncodingKind::CastF16);
    assert_eq!(selected.tier_intent, Some(Tier::Vram));
    assert_eq!(selected.tier_class.as_deref(), Some("hot"));
}

#[test]
fn gpu_selection_with_nvme_tier_picks_nvme_variant() {
    let (_tmp, file) = open_tier_fixture();
    let plan = file
        .select_variants_for_tier(
            ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(Some(GpuBackend::Wgpu)),
            Tier::Nvme,
        )
        .unwrap();
    let selected = &plan.selections[0];
    assert_eq!(selected.target, TargetTag::Wgpu);
    assert_eq!(selected.encoding, EncodingKind::BlockQuantized);
    assert_eq!(selected.tier_intent, Some(Tier::Nvme));
    assert_eq!(selected.tier_class.as_deref(), Some("cold"));
}

#[test]
fn absent_requested_tier_falls_back_to_backend_selection() {
    let (_tmp, file) = open_tier_fixture();
    let plan = file
        .select_variants_for_tier(
            ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(Some(GpuBackend::Wgpu)),
            Tier::Ram,
        )
        .unwrap();
    let selected = &plan.selections[0];
    assert_eq!(selected.encoding, EncodingKind::CastF16);
    assert_eq!(selected.tier_intent, Some(Tier::Vram));
}

#[test]
fn reader_rejects_unknown_tier_intent_metadata() {
    let canonical: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let f16 = convert_f32_to_f16_bytes(&canonical);
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "bad-tier".into(),
        dtype: LogicalDtype::F32,
        shape: vec![4],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(canonical),
        packed: vec![VariantInput {
            target: TargetTag::Wgpu,
            encoding: EncodingKind::CastF16,
            storage_dtype: Some(StorageDtype::Logical(LogicalDtype::F16)),
            layout: LayoutTag::RowMajor,
            alignment: 64,
            bytes: f16,
            meta: VariantMeta {
                extra: vec![("tier.intent".into(), "hbm".into())],
                ..VariantMeta::default()
            },
        }],
    });
    let bytes = writer.write_to_bytes().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let err = RsmfFile::open(tmp.path()).unwrap_err();
    assert!(
        err.to_string().contains("unknown tier intent"),
        "unexpected error: {err}"
    );
}
