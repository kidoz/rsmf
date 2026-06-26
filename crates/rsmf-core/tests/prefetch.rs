//! Round-trip tests for the `prefetch.*` metadata convention.

use std::io::Write;

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LazyRsmfFile, LogicalDtype, RsmfError, RsmfFile, SliceRangeReader, TargetTag};
use tempfile::NamedTempFile;

fn f32_bytes(n: usize, fill: f32) -> Vec<u8> {
    (0..n).flat_map(|_| fill.to_le_bytes()).collect()
}

fn write_file(writer: RsmfWriter) -> (NamedTempFile, Vec<u8>) {
    let bytes = writer.write_to_bytes().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    (tmp, bytes)
}

fn tensor_with_prefetch(name: &str) -> TensorInput {
    TensorInput {
        shard_id: 0,
        name: name.into(),
        dtype: LogicalDtype::F32,
        shape: vec![2, 2],
        metadata: vec![],
        canonical: VariantInput::canonical_raw(f32_bytes(4, 1.0))
            .with_prefetch_group("layer0.expert3")
            .with_prefetch_affinity("shard:1,expert:0:2"),
        packed: vec![
            VariantInput::packed_cast_f16(TargetTag::CpuGeneric, vec![0; 8])
                .with_prefetch_group("layer0.expert3")
                .with_prefetch_affinity("tier:nvme"),
        ],
    }
}

#[test]
fn round_trips_variant_prefetch_groups_and_affinity() {
    let writer = RsmfWriter::new()
        .with_tensor(tensor_with_prefetch("layers.0.experts.3.up"))
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "dense.weight".into(),
            dtype: LogicalDtype::F32,
            shape: vec![2, 2],
            metadata: vec![],
            canonical: VariantInput::canonical_raw(f32_bytes(4, 2.0))
                .with_prefetch_group("dense.hot"),
            packed: vec![],
        });

    let (tmp, bytes) = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();

    let idx = file.prefetch_hints().unwrap();
    assert_eq!(idx.len(), 3);
    assert_eq!(idx.groups.len(), 2);
    assert_eq!(idx.entries[0].tensor_name, "layers.0.experts.3.up");
    assert_eq!(idx.entries[0].variant_index, 0);
    assert_eq!(idx.entries[0].group, "layer0.expert3");
    assert_eq!(idx.entries[0].affinity, vec!["shard:1", "expert:0:2"]);

    let expert_group = idx.group("layer0.expert3").unwrap();
    assert_eq!(
        expert_group.variants,
        vec![
            rsmf_core::PrefetchVariantRef {
                tensor_name: "layers.0.experts.3.up".into(),
                variant_index: 0,
            },
            rsmf_core::PrefetchVariantRef {
                tensor_name: "layers.0.experts.3.up".into(),
                variant_index: 2,
            },
        ]
    );
    assert_eq!(
        expert_group.affinity,
        vec!["shard:1", "expert:0:2", "tier:nvme"]
    );

    let lazy = LazyRsmfFile::open(SliceRangeReader::new(bytes)).unwrap();
    assert_eq!(lazy.prefetch_hints().unwrap(), idx);
}

#[test]
fn file_without_prefetch_metadata_yields_empty_index() {
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "dense.weight".into(),
        dtype: LogicalDtype::F32,
        shape: vec![2, 2],
        metadata: vec![],
        canonical: VariantInput::canonical_raw(f32_bytes(4, 1.0)),
        packed: vec![],
    });

    let (tmp, _bytes) = write_file(writer);
    let idx = RsmfFile::open(tmp.path())
        .unwrap()
        .prefetch_hints()
        .unwrap();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
    assert!(idx.groups.is_empty());
}

#[test]
fn missing_group_is_structural_error() {
    let mut variant = VariantInput::canonical_raw(f32_bytes(4, 1.0));
    variant
        .meta
        .extra
        .push(("prefetch.affinity".into(), "shard:1".into()));
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "bad.weight".into(),
        dtype: LogicalDtype::F32,
        shape: vec![2, 2],
        metadata: vec![],
        canonical: variant,
        packed: vec![],
    });

    let (tmp, _bytes) = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let err = file.prefetch_hints().unwrap_err();
    assert!(matches!(err, RsmfError::Structural(_)));
    assert!(err.to_string().contains("no prefetch.group"));
}

#[test]
fn malformed_affinity_is_structural_error() {
    let variant = VariantInput::canonical_raw(f32_bytes(4, 1.0))
        .with_prefetch_group("group0")
        .with_prefetch_affinity("shard:1,,expert:0:3");
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "bad.weight".into(),
        dtype: LogicalDtype::F32,
        shape: vec![2, 2],
        metadata: vec![],
        canonical: variant,
        packed: vec![],
    });

    let (tmp, _bytes) = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let err = file.prefetch_hints().unwrap_err();
    assert!(matches!(err, RsmfError::Structural(_)));
    assert!(err.to_string().contains("empty prefetch.affinity token"));
}

#[test]
fn duplicate_group_key_is_structural_error() {
    let mut variant = VariantInput::canonical_raw(f32_bytes(4, 1.0)).with_prefetch_group("group0");
    variant
        .meta
        .extra
        .push(("prefetch.group".into(), "group1".into()));
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "bad.weight".into(),
        dtype: LogicalDtype::F32,
        shape: vec![2, 2],
        metadata: vec![],
        canonical: variant,
        packed: vec![],
    });

    let (tmp, _bytes) = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let err = file.prefetch_hints().unwrap_err();
    assert!(matches!(err, RsmfError::Structural(_)));
    assert!(err.to_string().contains("duplicate prefetch.group"));
}
