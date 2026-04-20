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

fn build_multi_tensor_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    // Two tensors with different shapes so the round-trip verifies
    // alignment padding between tensors inside the canonical arena.
    let w_a: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let w_b: Vec<u8> = (100..106u32)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert(
        "encoder.weight".into(),
        TensorView::new(Dtype::F32, vec![2, 4], &w_a).unwrap(),
    );
    tensors.insert(
        "encoder.bias".into(),
        TensorView::new(Dtype::F32, vec![6], &w_b).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, &None).unwrap();
    let path = dir.join("model_multi.safetensors");
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn stream_pack_roundtrips_multi_tensor_safetensors() {
    let dir = tempdir().unwrap();
    let st_path = build_multi_tensor_fixture(dir.path());
    let rsmf_path = dir.path().join("model_stream.rsmf");

    // --stream pack
    let out = Command::new(rsmf_bin())
        .args(["pack", "--stream", "--from-safetensors"])
        .arg(&st_path)
        .arg("--out")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "stream pack failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("packed (stream)"),
        "expected stream marker in stdout, got: {stdout}"
    );

    // verify --full confirms the on-disk layout is sound.
    let out = Command::new(rsmf_bin())
        .args(["verify", "--full"])
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("structural: ok"));
    assert!(stdout.contains("full checksum: ok"));

    // inspect should report two tensors.
    let out = Command::new(rsmf_bin())
        .arg("inspect")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "inspect failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("encoder.weight"));
    assert!(stdout.contains("encoder.bias"));
    assert!(stdout.contains("Tensors:  2"));

    // extract one and verify bytes match the source.
    let extract_path = dir.path().join("encoder_weight.bin");
    let out = Command::new(rsmf_bin())
        .args(["extract", "--tensor", "encoder.weight"])
        .arg(&rsmf_path)
        .arg(&extract_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "extract failed: {out:?}");
    let extracted = std::fs::read(&extract_path).unwrap();
    assert_eq!(extracted.len(), 8 * 4);
    for i in 0..8 {
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

#[test]
fn stream_pack_bundles_graph_and_asset() {
    let dir = tempdir().unwrap();
    let st_path = build_multi_tensor_fixture(dir.path());
    let graph_path = dir.path().join("intent.onnx");
    let asset_path = dir.path().join("tokenizer.json");
    std::fs::write(&graph_path, b"ONNX-stream-bundle-graph-bytes").unwrap();
    std::fs::write(&asset_path, br#"{"vocab":["<pad>","<bos>"]}"#).unwrap();
    let rsmf_path = dir.path().join("bundle_stream.rsmf");

    let out = Command::new(rsmf_bin())
        .args(["pack", "--stream", "--from-safetensors"])
        .arg(&st_path)
        .args(["--graph"])
        .arg(&graph_path)
        .args(["--asset"])
        .arg(&asset_path)
        .arg("--out")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "stream bundle pack failed: {out:?}");

    // verify --full
    let out = Command::new(rsmf_bin())
        .args(["verify", "--full"])
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");

    // inspect: both tensors + 1 graph + 1 asset.
    let out = Command::new(rsmf_bin())
        .arg("inspect")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "inspect failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Tensors:  2"));
    assert!(stdout.contains("Assets:   1"), "inspect stdout: {stdout}");

    // extract-asset round-trips the tokenizer.
    let extracted_asset = dir.path().join("tokenizer_out.json");
    let out = Command::new(rsmf_bin())
        .args(["extract-asset", "--name", "tokenizer.json"])
        .arg(&rsmf_path)
        .arg(&extracted_asset)
        .output()
        .unwrap();
    assert!(out.status.success(), "extract-asset failed: {out:?}");
    assert_eq!(
        std::fs::read(&extracted_asset).unwrap(),
        br#"{"vocab":["<pad>","<bos>"]}"#.to_vec()
    );
}

#[test]
fn stream_pack_with_compress_tensors_round_trips() {
    let dir = tempdir().unwrap();
    let st_path = build_multi_tensor_fixture(dir.path());
    let rsmf_path = dir.path().join("model_stream_compressed.rsmf");

    let out = Command::new(rsmf_bin())
        .args([
            "pack",
            "--stream",
            "--compress-tensors",
            "--from-safetensors",
        ])
        .arg(&st_path)
        .arg("--out")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stream + compress-tensors pack failed: {out:?}"
    );

    let out = Command::new(rsmf_bin())
        .args(["verify", "--full"])
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("structural: ok"));
    assert!(stdout.contains("full checksum: ok"));

    let extract_path = dir.path().join("ecw.bin");
    let out = Command::new(rsmf_bin())
        .args(["extract", "--tensor", "encoder.weight"])
        .arg(&rsmf_path)
        .arg(&extract_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "extract failed: {out:?}");
    let extracted = std::fs::read(&extract_path).unwrap();
    for i in 0..8 {
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

#[test]
fn stream_pack_rejects_incompatible_flags() {
    let dir = tempdir().unwrap();
    let st_path = build_fixture(dir.path());
    let rsmf_path = dir.path().join("model_stream_reject.rsmf");

    // --stream + --quantize-q4_0 must fail loudly rather than silently
    // drop the quantization.
    let out = Command::new(rsmf_bin())
        .args(["pack", "--stream", "--from-safetensors"])
        .arg(&st_path)
        .args(["--quantize-q4_0", "cpu_generic"])
        .arg("--out")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "pack must reject --stream + --quantize-q4_0"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--stream"),
        "rejection should mention --stream in stderr: {stderr}"
    );

    assert!(!rsmf_path.exists(), "no output should have been written");
}
