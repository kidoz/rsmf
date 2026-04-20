//! End-to-end tests for `StreamingRsmfWriter`.
//!
//! The streaming writer emits a different on-disk section order to the
//! batch writer (manifest last instead of first). These tests prove the
//! output still opens via `RsmfFile::open`, validates, verifies, and
//! serves the streamed bytes correctly.

use std::io::Cursor;

use rsmf_core::manifest::GraphKind;
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
fn tensors_and_graph_and_assets_round_trip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bundle.rsmf");

    let tensor_bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let graph_bytes = b"ONNX\x00fake-graph-for-streaming-test".to_vec();
    let asset_a = b"{\"vocab\":[\"<pad>\"]}".to_vec();
    let asset_b = b"{\"rope_theta\":10000.0}".to_vec();

    let mut w = StreamingRsmfWriter::new(&path)
        .expect("new")
        .with_metadata("source", "streaming_bundle");
    w.stream_canonical_tensor(
        "weights",
        LogicalDtype::F32,
        vec![2, 4],
        Cursor::new(&tensor_bytes),
    )
    .expect("tensor");
    w.stream_graph(GraphKind::Onnx, Cursor::new(&graph_bytes))
        .expect("graph");
    w.stream_asset("tokenizer.json", Cursor::new(&asset_a))
        .expect("asset a");
    w.stream_asset("config.json", Cursor::new(&asset_b))
        .expect("asset b");
    w.finish().expect("finish");

    let file = RsmfFile::open(&path).expect("open");
    let summary = file.inspect();
    assert_eq!(summary.tensor_count, 1);
    assert_eq!(summary.graph_kinds.len(), 1);
    assert_eq!(summary.asset_count, 2);

    let weights = file.tensor_view("weights").expect("weights");
    assert_eq!(weights.bytes(), tensor_bytes.as_slice());

    let payloads = file.graph_payloads();
    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0].kind, GraphKind::Onnx);
    assert_eq!(payloads[0].bytes, graph_bytes.as_slice());

    let tok = file.asset("tokenizer.json").expect("tok");
    assert_eq!(tok.bytes, asset_a.as_slice());
    let cfg = file.asset("config.json").expect("cfg");
    assert_eq!(cfg.bytes, asset_b.as_slice());

    file.full_verify().expect("full_verify");
}

#[test]
fn graph_after_asset_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_order.rsmf");

    let bytes = f32_bytes(&[1.0, 2.0]);
    let mut w = StreamingRsmfWriter::new(&path).expect("new");
    w.stream_canonical_tensor("a", LogicalDtype::F32, vec![2], Cursor::new(&bytes))
        .expect("tensor");
    w.stream_asset("cfg.json", Cursor::new(b"{}".to_vec()))
        .expect("asset");
    let err = w
        .stream_graph(GraphKind::Onnx, Cursor::new(b"fake".to_vec()))
        .expect_err("graph must be rejected after asset");
    assert!(matches!(err, rsmf_core::RsmfError::Structural(_)));
}

#[test]
fn tensor_after_graph_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_tensor.rsmf");

    let bytes = f32_bytes(&[1.0, 2.0]);
    let mut w = StreamingRsmfWriter::new(&path).expect("new");
    w.stream_canonical_tensor("a", LogicalDtype::F32, vec![2], Cursor::new(&bytes))
        .expect("tensor");
    w.stream_graph(GraphKind::Onnx, Cursor::new(b"fake".to_vec()))
        .expect("graph");
    let err = w
        .stream_canonical_tensor("b", LogicalDtype::F32, vec![2], Cursor::new(&bytes))
        .expect_err("tensor must be rejected after graph");
    assert!(matches!(err, rsmf_core::RsmfError::Structural(_)));
}

#[test]
fn multiple_graphs_round_trip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multi_graph.rsmf");

    let tensor_bytes = f32_bytes(&[1.0, 2.0]);
    let g1 = b"ONNX-first".to_vec();
    let g2 = b"ORT-second".to_vec();

    let mut w = StreamingRsmfWriter::new(&path).expect("new");
    w.stream_canonical_tensor("x", LogicalDtype::F32, vec![2], Cursor::new(&tensor_bytes))
        .expect("tensor");
    w.stream_graph(GraphKind::Onnx, Cursor::new(&g1))
        .expect("g1");
    w.stream_graph(GraphKind::Ort, Cursor::new(&g2))
        .expect("g2");
    w.finish().expect("finish");

    let file = RsmfFile::open(&path).expect("open");
    let payloads = file.graph_payloads();
    assert_eq!(payloads.len(), 2);
    assert_eq!(payloads[0].kind, GraphKind::Onnx);
    assert_eq!(payloads[0].bytes, g1.as_slice());
    assert_eq!(payloads[1].kind, GraphKind::Ort);
    assert_eq!(payloads[1].bytes, g2.as_slice());
    file.full_verify().expect("full_verify");
}

#[cfg(feature = "compression")]
#[test]
fn compressed_canonical_arena_round_trips() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("compressed_canonical.rsmf");

    // Enough tensors that zstd produces a noticeably-shrunk arena.
    let a_bytes = f32_bytes(&(0..256).map(|i| (i as f32) * 0.125).collect::<Vec<_>>());
    let b_bytes = f32_bytes(&(0..128).map(|i| (i as f32).sin()).collect::<Vec<_>>());

    let mut w = StreamingRsmfWriter::new(&path)
        .expect("new")
        .with_canonical_compression(3);
    w.stream_canonical_tensor("a", LogicalDtype::F32, vec![16, 16], Cursor::new(&a_bytes))
        .expect("a");
    w.stream_canonical_tensor("b", LogicalDtype::F32, vec![128], Cursor::new(&b_bytes))
        .expect("b");
    w.finish().expect("finish");

    let file = RsmfFile::open(&path).expect("open");
    assert_eq!(file.tensor_view("a").unwrap().bytes(), a_bytes.as_slice());
    assert_eq!(file.tensor_view("b").unwrap().bytes(), b_bytes.as_slice());
    file.full_verify().expect("full_verify");
}

#[cfg(feature = "compression")]
#[test]
fn compressed_graph_and_assets_round_trip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("compressed_bundle.rsmf");

    let tensor_bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let graph_bytes = b"ONNX-compressible-graph-bytes-".repeat(64);
    let asset_bytes = b"{\"vocab\":[\"<pad>\"]}".to_vec();

    let mut w = StreamingRsmfWriter::new(&path)
        .expect("new")
        .with_graph_compression(5)
        .with_assets_compression(3);
    w.stream_canonical_tensor(
        "w",
        LogicalDtype::F32,
        vec![2, 2],
        Cursor::new(&tensor_bytes),
    )
    .expect("tensor");
    w.stream_graph(GraphKind::Onnx, Cursor::new(&graph_bytes))
        .expect("graph");
    w.stream_asset("tokenizer.json", Cursor::new(&asset_bytes))
        .expect("asset");
    w.finish().expect("finish");

    let file = RsmfFile::open(&path).expect("open");
    let payloads = file.graph_payloads();
    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0].bytes, graph_bytes.as_slice());
    assert_eq!(
        file.asset("tokenizer.json").unwrap().bytes,
        asset_bytes.as_slice()
    );
    file.full_verify().expect("full_verify");
}

#[cfg(feature = "compression")]
#[test]
fn all_sections_compressed_round_trip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("all_compressed.rsmf");

    let tensor_bytes = f32_bytes(&(0..1024).map(|i| i as f32).collect::<Vec<_>>());
    let graph_bytes = b"ORT-graph-payload-".repeat(32);
    let asset_bytes = b"{\"config\":42}".to_vec();

    let mut w = StreamingRsmfWriter::new(&path)
        .expect("new")
        .with_canonical_compression(3)
        .with_graph_compression(3)
        .with_assets_compression(3);
    w.stream_canonical_tensor(
        "t",
        LogicalDtype::F32,
        vec![1024],
        Cursor::new(&tensor_bytes),
    )
    .expect("t");
    w.stream_graph(GraphKind::Ort, Cursor::new(&graph_bytes))
        .expect("g");
    w.stream_asset("config.json", Cursor::new(&asset_bytes))
        .expect("a");
    w.finish().expect("finish");

    let file = RsmfFile::open(&path).expect("open");
    assert_eq!(
        file.tensor_view("t").unwrap().bytes(),
        tensor_bytes.as_slice()
    );
    assert_eq!(file.graph_payloads()[0].bytes, graph_bytes.as_slice());
    assert_eq!(
        file.asset("config.json").unwrap().bytes,
        asset_bytes.as_slice()
    );
    file.full_verify().expect("full_verify");
}

#[test]
fn duplicate_asset_name_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dup_asset.rsmf");

    let bytes = f32_bytes(&[1.0]);
    let mut w = StreamingRsmfWriter::new(&path).expect("new");
    w.stream_canonical_tensor("a", LogicalDtype::F32, vec![1], Cursor::new(&bytes))
        .expect("tensor");
    w.stream_asset("dup.json", Cursor::new(b"first".to_vec()))
        .expect("asset a");
    let err = w
        .stream_asset("dup.json", Cursor::new(b"second".to_vec()))
        .expect_err("dup asset name");
    assert!(matches!(err, rsmf_core::RsmfError::Structural(ref m) if m.contains("duplicate")));
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
