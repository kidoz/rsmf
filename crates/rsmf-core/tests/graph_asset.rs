//! Test requirements #7, #8: graph payload + asset round-trip.

mod common;

use rsmf_core::RsmfFile;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn graph_payload_round_trip_bytes_exact() {
    let bytes = common::build_basic_file_bytes();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let payloads = file.graph_payloads();
    let graph = payloads.first().expect("graph present");
    assert_eq!(graph.bytes, b"fake-onnx-bytes");
}

#[test]
fn asset_round_trip_bytes_exact() {
    let bytes = common::build_basic_file_bytes();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let asset = file.asset("tokenizer.json").expect("asset present");
    assert_eq!(asset.bytes, b"{\"t\":1}");
    assert!(file.asset("does-not-exist").is_none());
}
