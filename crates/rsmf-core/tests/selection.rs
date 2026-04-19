//! Test requirement #5: variant selection across CpuOnly / GpuOnly /
//! HybridAuto, including canonical fallback.

mod common;

use rsmf_core::{Capabilities, ExecutionMode, GpuBackend, RsmfFile, TargetTag};
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
