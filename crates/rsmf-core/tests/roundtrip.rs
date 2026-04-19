//! Test requirement #1: round-trip write/read.

mod common;

use std::io::Write;

use rsmf_core::RsmfFile;
use tempfile::NamedTempFile;

#[test]
fn write_then_read_round_trip_bytes() {
    let bytes = common::build_basic_file_bytes();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).expect("open");
    let summary = file.inspect();
    assert_eq!(summary.tensor_count, 2);
    assert_eq!(summary.asset_count, 1);
    assert!(!summary.graph_kinds.is_empty());
    assert_eq!(summary.metadata.len(), 2);

    // Tensor bytes match what we wrote.
    let w = file.tensor_view("weight").expect("weight");
    assert_eq!(w.bytes().len(), 16 * 4);

    let b = file.tensor_view("bias").expect("bias");
    assert_eq!(b.bytes().len(), 4 * 4);

    // Graph / asset round-trip.
    let g = file.graph_payloads().pop().expect("graph");
    assert_eq!(g.bytes, b"fake-onnx-bytes");
    let a = file.asset("tokenizer.json").expect("asset");
    assert_eq!(a.bytes, b"{\"t\":1}");
}
