//! End-to-end tests for `StreamingRsmfWriter`.
//!
//! The streaming writer emits a different on-disk section order to the
//! batch writer (manifest last instead of first). These tests prove the
//! output still opens via `RsmfFile::open`, validates, verifies, and
//! serves the streamed bytes correctly.

use std::io::Cursor;

use rsmf_core::{LogicalDtype, RsmfFile, StreamingRsmfWriter};
use tempfile::tempdir;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn single_tensor_streams_and_reads_back() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("single.rsmf");

    let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let bytes = f32_bytes(&values);

    let mut w = StreamingRsmfWriter::new(&path).expect("new");
    w.stream_canonical_tensor(
        "weights",
        LogicalDtype::F32,
        vec![4, 4],
        Cursor::new(&bytes),
    )
    .expect("stream tensor");
    w.finish().expect("finish");

    let file = RsmfFile::open(&path).expect("open");
    let summary = file.inspect();
    assert_eq!(summary.tensor_count, 1);
    assert_eq!(summary.variant_count, 1);
    assert_eq!(summary.asset_count, 0);
    assert_eq!(summary.graph_kinds.len(), 0);

    let view = file.tensor_view("weights").expect("tensor_view");
    assert_eq!(view.dtype(), LogicalDtype::F32);
    assert_eq!(view.shape(), &[4u64, 4]);
    assert_eq!(view.bytes(), bytes.as_slice());

    file.full_verify().expect("full_verify");
}

#[test]
fn multiple_tensors_round_trip_in_declared_order() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multi.rsmf");

    let a_bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let b_bytes = f32_bytes(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    let mut w = StreamingRsmfWriter::new(&path).expect("new");
    w = w.with_metadata("source", "streaming_test");
    w.stream_canonical_tensor("a", LogicalDtype::F32, vec![2, 2], Cursor::new(&a_bytes))
        .expect("stream a");
    w.stream_canonical_tensor("b", LogicalDtype::F32, vec![3, 2], Cursor::new(&b_bytes))
        .expect("stream b");
    w.finish().expect("finish");

    let file = RsmfFile::open(&path).expect("open");
    let summary = file.inspect();
    assert_eq!(summary.tensor_count, 2);
    let meta: std::collections::HashMap<_, _> = summary.metadata.into_iter().collect();
    assert_eq!(
        meta.get("source").map(String::as_str),
        Some("streaming_test")
    );

    let a_view = file.tensor_view("a").expect("a");
    assert_eq!(a_view.shape(), &[2u64, 2]);
    assert_eq!(a_view.bytes(), a_bytes.as_slice());

    let b_view = file.tensor_view("b").expect("b");
    assert_eq!(b_view.shape(), &[3u64, 2]);
    assert_eq!(b_view.bytes(), b_bytes.as_slice());

    file.full_verify().expect("full_verify");
}

#[test]
fn duplicate_tensor_name_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dup.rsmf");

    let bytes = f32_bytes(&[0.0, 1.0]);
    let mut w = StreamingRsmfWriter::new(&path).expect("new");
    w.stream_canonical_tensor("x", LogicalDtype::F32, vec![2], Cursor::new(&bytes))
        .expect("first stream");
    let err = w
        .stream_canonical_tensor("x", LogicalDtype::F32, vec![2], Cursor::new(&bytes))
        .expect_err("dup");
    assert!(matches!(err, rsmf_core::RsmfError::Structural(ref msg) if msg.contains("duplicate")));
}

#[test]
fn finish_with_no_tensors_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.rsmf");

    let w = StreamingRsmfWriter::new(&path).expect("new");
    let err = w.finish().expect_err("finish must reject empty");
    assert!(matches!(err, rsmf_core::RsmfError::Structural(_)));
}

#[test]
fn different_logical_dtypes_round_trip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.rsmf");

    let f32_vec = f32_bytes(&[1.5, 2.5]);
    let i32_vec: Vec<u8> = [100i32, -100, 42, -42]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let u8_vec: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];

    let mut w = StreamingRsmfWriter::new(&path).expect("new");
    w.stream_canonical_tensor("f", LogicalDtype::F32, vec![2], Cursor::new(&f32_vec))
        .expect("f");
    w.stream_canonical_tensor("i", LogicalDtype::I32, vec![4], Cursor::new(&i32_vec))
        .expect("i");
    w.stream_canonical_tensor("u", LogicalDtype::U8, vec![8], Cursor::new(&u8_vec))
        .expect("u");
    w.finish().expect("finish");

    let file = RsmfFile::open(&path).expect("open");
    assert_eq!(file.inspect().tensor_count, 3);
    assert_eq!(file.tensor_view("f").unwrap().bytes(), f32_vec.as_slice());
    assert_eq!(file.tensor_view("i").unwrap().bytes(), i32_vec.as_slice());
    assert_eq!(file.tensor_view("u").unwrap().bytes(), u8_vec.as_slice());
    file.full_verify().expect("full_verify");
}
