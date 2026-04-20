//! Zstd compression tests for graph and asset sections.
//!
//! These tests compile unconditionally but only exercise the compression
//! path when the `compression` feature is enabled. When disabled, we verify
//! that requesting compression returns an error.

use std::io::Write;

#[cfg(feature = "compression")]
use rsmf_core::SectionKind;
#[cfg(feature = "compression")]
use rsmf_core::writer::AssetInput;
use rsmf_core::writer::{GraphInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile};
use tempfile::NamedTempFile;

fn basic_tensor() -> TensorInput {
    let bytes: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    TensorInput {
        shard_id: 0,

        name: "w".into(),
        dtype: LogicalDtype::F32,
        shape: vec![4],
        metadata: vec![],
        canonical: VariantInput::canonical_raw(bytes),
        packed: vec![],
    }
}

#[test]
#[cfg(feature = "compression")]
fn compressed_graph_round_trip() {
    let graph_bytes = vec![0xCCu8; 1000]; // Very compressible
    let writer = RsmfWriter::new()
        .with_tensor(basic_tensor())
        .with_graph(GraphInput::onnx(graph_bytes.clone()).with_compression(3));

    let file_bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&file_bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let section = file
        .sections()
        .iter()
        .find(|s| s.kind == SectionKind::Graph)
        .expect("graph section");

    assert!(section.is_compressed());
    // On-disk length is small due to compression.
    assert!(section.length < 100);

    let payloads = file.graph_payloads();
    let payload = payloads.first().expect("graph present");
    assert_eq!(payload.bytes, &graph_bytes[..]);

    // Full verify passes (hashes match after decompression).
    file.full_verify().unwrap();
}

#[test]
#[cfg(feature = "compression")]
fn compressed_asset_round_trip() {
    let asset_bytes = vec![0xDDu8; 500];
    let writer = RsmfWriter::new()
        .with_tensor(basic_tensor())
        .with_asset(AssetInput::new("data", asset_bytes.clone()).with_compression(3));

    let file_bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&file_bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let section = file
        .sections()
        .iter()
        .find(|s| s.kind == SectionKind::Assets)
        .expect("assets section");

    assert!(section.is_compressed());
    assert!(section.length < 100);

    let asset = file.asset("data").unwrap();
    assert_eq!(asset.bytes, &asset_bytes[..]);
    file.full_verify().unwrap();
}

#[test]
#[cfg(feature = "compression")]
fn compressed_tensor_arenas_round_trip() {
    let bytes: Vec<u8> = vec![0x42; 1000]; // Highly compressible
    let tensor = TensorInput {
        name: "w".into(),
        dtype: LogicalDtype::F32,
        shape: vec![250],
        shard_id: 0,
        metadata: vec![],
        canonical: VariantInput::canonical_raw(bytes.clone()),
        packed: vec![VariantInput::packed_cast_f16(
            rsmf_core::TargetTag::CpuGeneric,
            vec![0x42; 500],
        )],
    };

    let writer = RsmfWriter::new()
        .with_canonical_compression(3)
        .with_packed_compression(3)
        .with_tensor(tensor);

    let file_bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&file_bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();

    // Check canonical arena
    let canonical_section = file
        .sections()
        .iter()
        .find(|s| s.kind == SectionKind::CanonicalArena)
        .expect("canonical section");
    assert!(
        canonical_section.is_compressed(),
        "canonical arena should be compressed"
    );

    // Check packed arena
    let packed_section = file
        .sections()
        .iter()
        .find(|s| s.kind == SectionKind::PackedArena)
        .expect("packed section");
    assert!(
        packed_section.is_compressed(),
        "packed arena should be compressed"
    );

    // Read back: bytes match
    let view = file.tensor_view("w").unwrap();
    assert_eq!(view.bytes, &bytes[..]);

    let packed_view = file.tensor_view_variant("w", 1).unwrap();
    assert_eq!(packed_view.bytes.len(), 500);

    // Full verify passes.
    file.full_verify().unwrap();
}

#[test]
fn uncompressed_files_still_open_unchanged() {
    // Regression: uncompressed files (flags=0) should not be affected by the
    // v2 compression support.
    let writer = RsmfWriter::new()
        .with_tensor(basic_tensor())
        .with_graph(GraphInput::ort(b"hello".to_vec()));
    let bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    assert_eq!(file.graph_payloads().first().unwrap().bytes, b"hello");
    file.full_verify().unwrap();
}

#[test]
#[cfg(not(feature = "compression"))]
fn fail_when_compression_requested_without_feature() {
    let writer = RsmfWriter::new()
        .with_tensor(basic_tensor())
        .with_graph(GraphInput::ort(b"data".to_vec()).with_compression(3));
    let err = writer.write_to_bytes().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("compression") && msg.contains("not enabled"),
        "got: {msg}"
    );
}
