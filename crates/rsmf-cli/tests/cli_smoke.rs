//! End-to-end smoke test that exercises the `rsmf` binary through `Command`.
//!
//! Creates a safetensors fixture, packs it, then runs `inspect`, `verify
//! --full`, `select`, and `extract` — asserting zero exit codes and expected
//! output markers.

use std::collections::HashMap;
use std::process::Command;

use safetensors::tensor::{Dtype, TensorView};
use tempfile::tempdir;

fn rsmf_bin() -> &'static str {
    env!("CARGO_BIN_EXE_rsmf")
}

fn build_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let w: Vec<u8> = (0..12u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert(
        "weight".into(),
        TensorView::new(Dtype::F32, vec![3, 4], &w).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, &None).unwrap();
    let path = dir.join("model.safetensors");
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn end_to_end_pack_inspect_verify_select_extract() {
    let dir = tempdir().unwrap();
    let st_path = build_fixture(dir.path());
    let rsmf_path = dir.path().join("model.rsmf");
    let extract_path = dir.path().join("weight.bin");

    // pack
    let out = Command::new(rsmf_bin())
        .args(["pack", "--from-safetensors"])
        .arg(&st_path)
        .arg("--out")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "pack failed: {out:?}");

    // inspect
    let out = Command::new(rsmf_bin())
        .arg("inspect")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "inspect failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("weight"));
    assert!(stdout.contains("Tensors:  1"));

    // verify --full
    let out = Command::new(rsmf_bin())
        .args(["verify", "--full"])
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("structural: ok"));
    assert!(stdout.contains("full checksum: ok"));

    // select --mode cpu
    let out = Command::new(rsmf_bin())
        .args(["select", "--mode", "cpu"])
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "select failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("weight"));

    // extract
    let out = Command::new(rsmf_bin())
        .args(["extract", "--tensor", "weight"])
        .arg(&rsmf_path)
        .arg(&extract_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "extract failed: {out:?}");
    let extracted = std::fs::read(&extract_path).unwrap();
    assert_eq!(extracted.len(), 12 * 4);
    for i in 0..12 {
        let off = i * 4;
        let v = f32::from_le_bytes([
            extracted[off],
            extracted[off + 1],
            extracted[off + 2],
            extracted[off + 3],
        ]);
        assert!((v - i as f32).abs() < f32::EPSILON);
    }
}
