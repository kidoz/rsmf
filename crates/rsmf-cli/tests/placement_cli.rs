//! Smoke tests for `rsmf placement`.

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
            name: "weights".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            packed: Vec::new(),
        })
        .write_to_path(path)
        .unwrap();
}

#[test]
fn placement_set_inspect_and_verify_round_trip() {
    let dir = tempdir().unwrap();
    let rsmf_path = dir.path().join("model.rsmf");
    let plan_path = dir.path().join("placement.toml");
    write_fixture(&rsmf_path);
    std::fs::write(
        &plan_path,
        r#"
[metadata]
planner = "test"

[[devices]]
id = 0
kind = "cpu"
tier = "ram"
capacity_bytes = 1024
bandwidth_mbps = 100

[[placements]]
shard_id = 0
primary_device = 0
prefetch_priority = 3
flags = ["pin"]
"#,
    )
    .unwrap();

    let out = Command::new(rsmf_bin())
        .args(["placement", "set"])
        .arg(&rsmf_path)
        .arg("--plan")
        .arg(&plan_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "placement set failed: {out:?}");

    let out = Command::new(rsmf_bin())
        .args(["placement", "inspect"])
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "placement inspect failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("PlacementManifest: version 1"),
        "inspect stdout: {stdout}"
    );
    assert!(
        stdout.contains("id=0 kind=cpu tier=ram"),
        "inspect stdout: {stdout}"
    );
    assert!(
        stdout.contains("shard_id=0 primary_device=0"),
        "inspect stdout: {stdout}"
    );

    let out = Command::new(rsmf_bin())
        .args(["verify", "--full"])
        .arg(&rsmf_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "verify failed: {out:?}");
}

#[test]
fn placement_set_rejects_unknown_shard_id() {
    let dir = tempdir().unwrap();
    let rsmf_path = dir.path().join("model.rsmf");
    let plan_path = dir.path().join("placement.toml");
    write_fixture(&rsmf_path);
    std::fs::write(
        &plan_path,
        r#"
[[devices]]
id = 0
kind = "cpu"
tier = "ram"

[[placements]]
shard_id = 42
primary_device = 0
"#,
    )
    .unwrap();

    let out = Command::new(rsmf_bin())
        .args(["placement", "set"])
        .arg(&rsmf_path)
        .arg("--plan")
        .arg(&plan_path)
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "placement set unexpectedly succeeded: {out:?}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("placement references shard_id 42"),
        "stderr: {stderr}"
    );
}
