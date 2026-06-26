//! Smoke tests for `rsmf shard`.

use std::process::Command;

use rsmf_core::LogicalDtype;
use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use tempfile::tempdir;

fn rsmf_bin() -> &'static str {
    env!("CARGO_BIN_EXE_rsmf")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn write_fixture(path: &std::path::Path) {
    RsmfWriter::new()
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "a".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            packed: Vec::new(),
        })
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "b".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, 20.0, 30.0, 40.0])),
            packed: Vec::new(),
        })
        .write_to_path(path)
        .unwrap();
}

#[test]
fn shard_then_verify_with_attached_shards() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("source.rsmf");
    let out_dir = dir.path().join("out");
    write_fixture(&src);

    let out = Command::new(rsmf_bin())
        .args(["shard", "--by", "size", "--shards", "2", "--out-dir"])
        .arg(&out_dir)
        .arg(&src)
        .output()
        .unwrap();
    assert!(out.status.success(), "shard failed: {out:?}");

    let master = out_dir.join("master.rsmf");
    let shard_1 = out_dir.join("shard-1.bin");
    let shard_2 = out_dir.join("shard-2.bin");
    assert!(master.exists());
    assert!(shard_1.exists());
    assert!(shard_2.exists());

    let out = Command::new(rsmf_bin())
        .args(["verify", "--full", "--shard"])
        .arg(format!("1={}", shard_1.display()))
        .arg("--shard")
        .arg(format!("2={}", shard_2.display()))
        .arg(&master)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("structural: ok"), "stdout: {stdout}");
    assert!(stdout.contains("full checksum: ok"), "stdout: {stdout}");
}
