//! Smoke tests for tier-aware `rsmf select`.

use std::process::Command;

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput, convert_f32_to_f16_bytes};
use rsmf_core::{LogicalDtype, TargetTag, Tier};
use tempfile::tempdir;

fn rsmf_bin() -> &'static str {
    env!("CARGO_BIN_EXE_rsmf")
}

fn write_tier_fixture(path: &std::path::Path) {
    let canonical: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let f16 = convert_f32_to_f16_bytes(&canonical);
    let q4 = vec![0u8; 18];

    RsmfWriter::new()
        .with_tensor(TensorInput {
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
                VariantInput::packed_q4_0(TargetTag::Wgpu, q4)
                    .with_tier_intent(Tier::Nvme)
                    .with_tier_class("cold"),
            ],
        })
        .write_to_path(path)
        .unwrap();
}

#[test]
fn select_tier_nvme_prefers_nvme_variant() {
    let dir = tempdir().unwrap();
    let rsmf_path = dir.path().join("tiered.rsmf");
    write_tier_fixture(&rsmf_path);

    let out = Command::new(rsmf_bin())
        .args(["select", "--mode", "gpu", "--assume-wgpu", "--tier", "nvme"])
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "select failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("tier=nvme"), "stdout: {stdout}");
    assert!(
        stdout.contains("encoding=block_quantized tier=nvme class=cold"),
        "stdout: {stdout}"
    );
}
