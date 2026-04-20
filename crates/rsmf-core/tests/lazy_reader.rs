//! Round-trip tests for `LazyRsmfFile` over a `SliceRangeReader`.
//!
//! The in-memory `SliceRangeReader` lets us exercise the "fetch by range"
//! contract without a network dependency. A tiny counting wrapper tracks
//! byte volume per request so we can assert that `open()` pulls only
//! preamble + section table + manifest, and that `fetch_tensor_bytes`
//! fetches exactly one variant's worth of bytes afterward.

use std::sync::atomic::{AtomicU64, Ordering};

use rsmf_core::writer::{AssetInput, GraphInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{
    LazyRsmfFile, LogicalDtype, RangeReader, Result, RsmfFile, SliceRangeReader, TargetTag,
};

struct CountingReader {
    inner: SliceRangeReader,
    calls: AtomicU64,
    bytes: AtomicU64,
}

impl CountingReader {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            inner: SliceRangeReader::new(bytes),
            calls: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }

    fn call_count(&self) -> u64 {
        self.calls.load(Ordering::SeqCst)
    }

    fn byte_count(&self) -> u64 {
        self.bytes.load(Ordering::SeqCst)
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::SeqCst);
        self.bytes.store(0, Ordering::SeqCst);
    }
}

impl RangeReader for CountingReader {
    fn len(&self) -> u64 {
        self.inner.len()
    }
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.bytes.fetch_add(length, Ordering::SeqCst);
        self.inner.read_range(offset, length)
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn base_tensor(name: &str, values: &[f32]) -> TensorInput {
    TensorInput {
        shard_id: 0,
        name: name.into(),
        dtype: LogicalDtype::F32,
        shape: vec![values.len() as u64],
        metadata: vec![],
        canonical: VariantInput::canonical_raw(f32_bytes(values)),
        packed: vec![],
    }
}

#[test]
fn open_issues_exactly_three_range_reads() {
    let bytes = RsmfWriter::new()
        .with_tensor(base_tensor("w", &[1.0, 2.0, 3.0, 4.0]))
        .write_to_bytes()
        .unwrap();
    let total_len = bytes.len() as u64;

    let reader = CountingReader::new(bytes);
    let _lazy = LazyRsmfFile::open(&reader).unwrap();

    // Three reads: preamble, section table, manifest. No tensor, graph, or
    // asset bytes should have been fetched — that's the whole point of lazy.
    assert_eq!(reader.call_count(), 3, "expected exactly 3 range reads");
    assert!(
        reader.byte_count() < total_len,
        "metadata reads should be a fraction of the file: read {} / {total_len}",
        reader.byte_count()
    );
}

#[test]
fn fetch_tensor_bytes_matches_rsmf_file() {
    let values: Vec<f32> = (0..128).map(|i| i as f32 * 0.25).collect();
    let bytes = RsmfWriter::new()
        .with_tensor(base_tensor("w", &values))
        .with_tensor(base_tensor("b", &[-1.0, -2.0]))
        .write_to_bytes()
        .unwrap();

    // Truth source: the existing RsmfFile.
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    use std::io::Write;
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let file = RsmfFile::open(tmp.path()).unwrap();
    let want_w = file.tensor_view("w").unwrap().bytes.to_vec();
    let want_b = file.tensor_view("b").unwrap().bytes.to_vec();

    // Lazy path:
    let lazy = LazyRsmfFile::open(SliceRangeReader::new(bytes)).unwrap();
    let got_w = lazy.fetch_tensor_bytes("w").unwrap();
    let got_b = lazy.fetch_tensor_bytes("b").unwrap();

    assert_eq!(got_w, want_w);
    assert_eq!(got_b, want_b);
}

#[test]
fn fetch_only_reads_variant_range() {
    // Build a file with one large tensor and one small tensor. Fetch only
    // the small one and assert we read ~small bytes, not the whole file.
    let big: Vec<f32> = vec![0.5_f32; 16 * 1024];
    let small: Vec<f32> = vec![0.1_f32; 8];
    let bytes = RsmfWriter::new()
        .with_tensor(base_tensor("big", &big))
        .with_tensor(base_tensor("small", &small))
        .write_to_bytes()
        .unwrap();
    let total = bytes.len() as u64;

    let reader = CountingReader::new(bytes);
    let lazy = LazyRsmfFile::open(&reader).unwrap();
    reader.reset();

    let fetched = lazy.fetch_tensor_bytes("small").unwrap();
    assert_eq!(fetched.len(), small.len() * 4);

    // Only one range read for the small variant, and it must read far
    // fewer bytes than the full file.
    assert_eq!(reader.call_count(), 1, "exactly one fetch for one tensor");
    assert!(
        reader.byte_count() < total / 10,
        "small tensor fetch pulled {} of {total} bytes",
        reader.byte_count()
    );
}

#[test]
fn fetch_packed_variant_works() {
    let bytes = RsmfWriter::new()
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "w".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],
            metadata: vec![],
            canonical: VariantInput::canonical_raw(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            packed: vec![VariantInput::packed_cast_f16(
                TargetTag::CpuGeneric,
                vec![0u8; 8],
            )],
        })
        .write_to_bytes()
        .unwrap();

    let lazy = LazyRsmfFile::open(SliceRangeReader::new(bytes)).unwrap();
    let tensor = lazy
        .manifest()
        .tensors
        .iter()
        .find(|t| t.name == "w")
        .unwrap();
    let packed_idx = tensor.packed_variants[0];
    let got = lazy.fetch_tensor_variant_bytes("w", packed_idx).unwrap();
    assert_eq!(got, vec![0u8; 8]);
}

#[test]
fn fetch_asset_and_graph_bytes() {
    let bytes = RsmfWriter::new()
        .with_tensor(base_tensor("w", &[0.0; 4]))
        .with_asset(AssetInput::new(
            "config.json",
            b"{\"hello\":\"world\"}".to_vec(),
        ))
        .with_graph(GraphInput::onnx(b"fake-onnx".to_vec()))
        .write_to_bytes()
        .unwrap();

    let lazy = LazyRsmfFile::open(SliceRangeReader::new(bytes)).unwrap();

    let asset = lazy.fetch_asset_bytes("config.json").unwrap();
    assert_eq!(asset, b"{\"hello\":\"world\"}");

    let graph = lazy.fetch_graph_bytes(0).unwrap();
    assert_eq!(graph, b"fake-onnx");
}

#[test]
#[cfg(feature = "compression")]
fn fetch_from_compressed_canonical_arena() {
    let values: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let bytes = RsmfWriter::new()
        .with_canonical_compression(3)
        .with_tensor(base_tensor("w", &values))
        .write_to_bytes()
        .unwrap();

    let lazy = LazyRsmfFile::open(SliceRangeReader::new(bytes)).unwrap();
    let got = lazy.fetch_tensor_bytes("w").unwrap();
    assert_eq!(got, f32_bytes(&values));
}

#[test]
fn unknown_tensor_returns_not_found() {
    let bytes = RsmfWriter::new()
        .with_tensor(base_tensor("w", &[1.0]))
        .write_to_bytes()
        .unwrap();
    let lazy = LazyRsmfFile::open(SliceRangeReader::new(bytes)).unwrap();
    let err = lazy.fetch_tensor_bytes("missing").unwrap_err();
    assert!(err.to_string().contains("missing"), "got: {err}");
}

#[test]
fn short_source_is_rejected_early() {
    // A few bytes: too short for the preamble.
    let reader = SliceRangeReader::new(vec![0u8; 8]);
    let err = LazyRsmfFile::open(reader).unwrap_err();
    assert!(err.to_string().contains("too short"), "got: {err}");
}

#[test]
fn manifest_round_trips_intact() {
    // Lazy and eager readers should report identical manifests.
    let bytes = RsmfWriter::new()
        .with_metadata("rsmf.creator", "test")
        .with_tensor(base_tensor("w", &[1.0, 2.0, 3.0, 4.0]))
        .write_to_bytes()
        .unwrap();

    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    use std::io::Write;
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let eager = RsmfFile::open(tmp.path()).unwrap();

    let lazy = LazyRsmfFile::open(SliceRangeReader::new(bytes)).unwrap();
    assert_eq!(eager.manifest(), lazy.manifest());
    assert_eq!(eager.sections(), lazy.sections());
    assert_eq!(eager.preamble(), lazy.preamble());
}
