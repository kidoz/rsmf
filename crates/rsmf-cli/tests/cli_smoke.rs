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
fn rewrite_strips_named_variant_and_asset() {
    let dir = tempdir().unwrap();
    let st_path = build_fixture(dir.path());
    let asset_path = dir.path().join("config.json");
    std::fs::write(&asset_path, br#"{"model":"tiny"}"#).unwrap();
    let src_rsmf = dir.path().join("src.rsmf");
    let out_rsmf = dir.path().join("out.rsmf");

    // Pack with a Q4_0 cpu_generic variant and the config asset.
    let out = Command::new(rsmf_bin())
        .args(["pack", "--from-safetensors"])
        .arg(&st_path)
        .args(["--quantize-q4_0", "cpu_generic"])
        .args(["--asset"])
        .arg(&asset_path)
        .arg("--out")
        .arg(&src_rsmf)
        .output()
        .unwrap();
    assert!(out.status.success(), "pack failed: {out:?}");

    // Sanity: source has the cpu_generic variant + the asset.
    let out = Command::new(rsmf_bin())
        .arg("inspect")
        .arg(&src_rsmf)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("cpu_generic"),
        "src missing cpu_generic: {stdout}"
    );
    assert!(
        stdout.contains("config.json"),
        "src missing config.json: {stdout}"
    );

    // Rewrite: strip cpu_generic variant + config.json asset.
    let out = Command::new(rsmf_bin())
        .args(["rewrite"])
        .arg(&src_rsmf)
        .arg(&out_rsmf)
        .args(["--strip-variants", "cpu_generic"])
        .args(["--strip-asset", "config.json"])
        .output()
        .unwrap();
    assert!(out.status.success(), "rewrite failed: {out:?}");

    // Destination must still verify.
    let out = Command::new(rsmf_bin())
        .args(["verify", "--full"])
        .arg(&out_rsmf)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("full checksum: ok"));

    // Destination must NOT contain the stripped items.
    let out = Command::new(rsmf_bin())
        .arg("inspect")
        .arg(&out_rsmf)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("cpu_generic"),
        "cpu_generic survived rewrite: {stdout}"
    );
    assert!(
        !stdout.contains("config.json"),
        "config.json survived rewrite: {stdout}"
    );
    // Canonical tensor must still be present.
    assert!(
        stdout.contains("weight"),
        "canonical weight missing: {stdout}"
    );
}

#[test]
fn rewrite_keep_only_canonical_removes_all_packed() {
    let dir = tempdir().unwrap();
    let st_path = build_fixture(dir.path());
    let src_rsmf = dir.path().join("multi_variant.rsmf");
    let out_rsmf = dir.path().join("canonical_only.rsmf");

    // Pack with two packed variants: cpu_generic (Q4_0) + wgpu (f16 cast).
    let out = Command::new(rsmf_bin())
        .args(["pack", "--from-safetensors"])
        .arg(&st_path)
        .args(["--quantize-q4_0", "cpu_generic"])
        .args(["--cast-f16", "wgpu"])
        .arg("--out")
        .arg(&src_rsmf)
        .output()
        .unwrap();
    assert!(out.status.success(), "pack failed: {out:?}");

    let out = Command::new(rsmf_bin())
        .args(["rewrite"])
        .arg(&src_rsmf)
        .arg(&out_rsmf)
        .arg("--keep-only-canonical")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "rewrite --keep-only-canonical failed: {out:?}"
    );

    let out = Command::new(rsmf_bin())
        .args(["verify", "--full"])
        .arg(&out_rsmf)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");

    let out = Command::new(rsmf_bin())
        .arg("inspect")
        .arg(&out_rsmf)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // No packed variants left.
    assert!(
        !stdout.contains("cpu_generic") && !stdout.contains("wgpu"),
        "expected no packed variants after --keep-only-canonical: {stdout}"
    );
    // Still has the tensor via its canonical.
    assert!(stdout.contains("weight"));
    assert!(stdout.contains("Variants: 1"));
}

#[test]
fn rewrite_dedup_shrinks_file_with_tied_tensors() {
    // Safetensors fixture with two tensors that share the exact same byte
    // payload (the tied-embeddings case). After `rsmf rewrite --dedup` the
    // output should be smaller than a plain rewrite and still verify.
    let dir = tempdir().unwrap();
    let shared: Vec<u8> = (0..64u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let st_path = dir.path().join("tied.safetensors");
    {
        let mut tensors: HashMap<String, TensorView> = HashMap::new();
        tensors.insert(
            "embed".into(),
            TensorView::new(Dtype::F32, vec![8, 8], &shared).unwrap(),
        );
        tensors.insert(
            "lm_head".into(),
            TensorView::new(Dtype::F32, vec![8, 8], &shared).unwrap(),
        );
        let bytes = safetensors::serialize(&tensors, &None).unwrap();
        std::fs::write(&st_path, bytes).unwrap();
    }

    let src_rsmf = dir.path().join("tied.rsmf");
    let plain_out = dir.path().join("plain.rsmf");
    let dedup_out = dir.path().join("dedup.rsmf");

    let out = Command::new(rsmf_bin())
        .args(["pack", "--from-safetensors"])
        .arg(&st_path)
        .arg("--out")
        .arg(&src_rsmf)
        .output()
        .unwrap();
    assert!(out.status.success(), "pack failed: {out:?}");

    // Rewrite without dedup.
    let out = Command::new(rsmf_bin())
        .args(["rewrite"])
        .arg(&src_rsmf)
        .arg(&plain_out)
        .output()
        .unwrap();
    assert!(out.status.success(), "plain rewrite failed: {out:?}");

    // Rewrite with dedup.
    let out = Command::new(rsmf_bin())
        .args(["rewrite"])
        .arg(&src_rsmf)
        .arg(&dedup_out)
        .arg("--dedup")
        .output()
        .unwrap();
    assert!(out.status.success(), "dedup rewrite failed: {out:?}");

    let plain_len = std::fs::metadata(&plain_out).unwrap().len();
    let dedup_len = std::fs::metadata(&dedup_out).unwrap().len();
    assert!(
        dedup_len < plain_len,
        "dedup should shrink the file: dedup={dedup_len} plain={plain_len}"
    );

    // Dedup output must still pass full verification.
    let out = Command::new(rsmf_bin())
        .args(["verify", "--full"])
        .arg(&dedup_out)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("full checksum: ok"));
}

#[test]
fn rewrite_rejects_same_input_and_output() {
    let dir = tempdir().unwrap();
    let st_path = build_fixture(dir.path());
    let rsmf_path = dir.path().join("self_rewrite.rsmf");

    let out = Command::new(rsmf_bin())
        .args(["pack", "--from-safetensors"])
        .arg(&st_path)
        .arg("--out")
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success());

    let out = Command::new(rsmf_bin())
        .args(["rewrite"])
        .arg(&rsmf_path)
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "rewrite must reject same input/output path"
    );
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
