//! Integration tests for the experimental multi-file sharding path.
//!
//! Shard files are raw **arena byte buffers**: a variant that lives at
//! `section_relative_offset = K` and has `length = L` in the manifest
//! is resolved against `shard_bytes[K..K + L]`. The master file still
//! carries a canonical arena descriptor so structural validation
//! passes; the bytes at that position in the master are placeholder
//! once a shard is attached.
//!
//! These tests exercise that contract end-to-end. v1 of RSMF does not
//! offer a writer-side sharding API — both sides are hand-assembled.

use memmap2::MmapMut;
use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfError, RsmfFile};
use std::fs;
use tempfile::tempdir;

fn build_master_with_sharded_tensor(path: &std::path::Path, placeholder: Vec<u8>) {
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 1,
        name: "weights".into(),
        dtype: LogicalDtype::F32,
        shape: vec![4],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(placeholder),
        packed: Vec::new(),
    });
    writer.write_to_path(path).expect("write master");
}

#[test]
fn sharded_read_returns_shard_bytes_not_master_placeholder() {
    let dir = tempdir().unwrap();

    // The real tensor bytes live in the shard.
    let real_values = [1.0_f32, 2.0, 3.0, 4.0];
    let real_bytes: Vec<u8> = real_values.iter().flat_map(|v| v.to_le_bytes()).collect();

    // The master keeps placeholder bytes — distinct from the shard so
    // the test would fail if the reader silently fell back to them.
    let master_path = dir.path().join("master.rsmf");
    let placeholder: Vec<u8> = (0..16).map(|i| 0xA0 + i).collect();
    build_master_with_sharded_tensor(&master_path, placeholder);

    let shard_path = dir.path().join("shard1.bin");
    fs::write(&shard_path, &real_bytes).unwrap();

    let master = RsmfFile::open(&master_path).unwrap();
    let shard_file = fs::File::open(&shard_path).unwrap();
    // SAFETY: shard file not mutated for the lifetime of the mmap.
    let shard_mmap = unsafe { memmap2::Mmap::map(&shard_file).unwrap() };
    let master = master.with_shard(1, shard_mmap);

    let view = master.tensor_view("weights").unwrap();
    let got = view.as_slice::<f32>().unwrap();
    assert_eq!(
        got, real_values,
        "shard bytes must override master placeholder"
    );
}

#[test]
fn sharded_read_without_attached_shard_fails_with_typed_error() {
    let dir = tempdir().unwrap();
    let master_path = dir.path().join("master.rsmf");
    build_master_with_sharded_tensor(&master_path, vec![0u8; 16]);

    let master = RsmfFile::open(&master_path).unwrap();
    let err = master.tensor_view("weights").unwrap_err();
    match err {
        RsmfError::Unsupported(msg) => {
            assert!(
                msg.contains("shard 1") && msg.contains("not loaded"),
                "error message should name the missing shard, got: {msg}"
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn short_shard_fails_with_structural_error_naming_length() {
    let dir = tempdir().unwrap();
    let master_path = dir.path().join("master.rsmf");
    build_master_with_sharded_tensor(&master_path, vec![0u8; 16]);

    // Shard intentionally shorter than the variant length (4×f32 = 16).
    let shard_path = dir.path().join("shard1.bin");
    fs::write(&shard_path, vec![0u8; 4]).unwrap();

    let master = RsmfFile::open(&master_path).unwrap();
    let shard_file = fs::File::open(&shard_path).unwrap();
    // SAFETY: shard file not mutated for the lifetime of the mmap.
    let shard_mmap = unsafe { memmap2::Mmap::map(&shard_file).unwrap() };
    let master = master.with_shard(1, shard_mmap);

    let err = master.tensor_view("weights").unwrap_err();
    match err {
        RsmfError::Structural(msg) => {
            assert!(
                msg.contains("past shard") && msg.contains("shard has 4"),
                "message should mention shard length, got: {msg}"
            );
        }
        other => panic!("expected Structural, got {other:?}"),
    }
}

#[test]
fn unsharded_tensor_still_reads_from_master() {
    // Regression check: shard_id=0 must keep reading from the master's
    // canonical arena unchanged even when the reader has shards attached.
    let dir = tempdir().unwrap();
    let master_path = dir.path().join("master.rsmf");
    let real_values = [10.0_f32, 20.0, 30.0, 40.0];
    let real_bytes: Vec<u8> = real_values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "weights".into(),
        dtype: LogicalDtype::F32,
        shape: vec![4],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(real_bytes.clone()),
        packed: Vec::new(),
    });
    writer.write_to_path(&master_path).unwrap();

    // Attach an unrelated shard — it must not interfere.
    let shard_path = dir.path().join("shard9.bin");
    fs::write(&shard_path, vec![0xFFu8; 32]).unwrap();
    let master = RsmfFile::open(&master_path).unwrap();
    let shard_file = fs::File::open(&shard_path).unwrap();
    // SAFETY: shard file not mutated for the lifetime of the mmap.
    let shard_mmap = unsafe { memmap2::Mmap::map(&shard_file).unwrap() };
    let master = master.with_shard(9, shard_mmap);

    let view = master.tensor_view("weights").unwrap();
    assert_eq!(view.as_slice::<f32>().unwrap(), real_values);
}

// Keep the unused-import warning silent on platforms that don't need
// MmapMut explicitly; the import is here so a future "mutable shard"
// follow-up has the type available without re-importing.
#[allow(dead_code)]
fn _keep_mmap_mut_in_scope(_: MmapMut) {}
