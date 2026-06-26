//! Round-trip tests for the `moe.*` metadata convention.

use std::io::Write;

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LazyRsmfFile, LogicalDtype, MoeRole, RsmfError, RsmfFile, SliceRangeReader};
use tempfile::NamedTempFile;

fn f32_bytes(n: usize, fill: f32) -> Vec<u8> {
    (0..n).flat_map(|_| fill.to_le_bytes()).collect()
}

fn moe_tensor(name: &str, layer: u32, expert: u32, role: &str) -> TensorInput {
    TensorInput {
        shard_id: 0,
        name: name.into(),
        dtype: LogicalDtype::F32,
        shape: vec![2, 2],
        metadata: vec![
            ("moe.layer".into(), layer.to_string()),
            ("moe.expert".into(), expert.to_string()),
            ("moe.role".into(), role.into()),
        ],
        canonical: VariantInput::canonical_raw(f32_bytes(4, layer as f32 + expert as f32)),
        packed: vec![],
    }
}

fn write_file(writer: RsmfWriter) -> (NamedTempFile, Vec<u8>) {
    let bytes = writer.write_to_bytes().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    (tmp, bytes)
}

#[test]
fn groups_two_layer_four_expert_fixture() {
    let mut writer = RsmfWriter::new()
        .with_metadata("moe.n_experts", "4")
        .with_metadata("moe.top_k", "2")
        .with_metadata("moe.n_shared", "1")
        .with_metadata("model.arch", "toy-moe");

    for layer in 0..2 {
        for expert in 0..4 {
            writer = writer
                .with_tensor(moe_tensor(
                    &format!("layers.{layer}.experts.{expert}.gate"),
                    layer,
                    expert,
                    "gate",
                ))
                .with_tensor(moe_tensor(
                    &format!("layers.{layer}.experts.{expert}.up"),
                    layer,
                    expert,
                    "up",
                ))
                .with_tensor(moe_tensor(
                    &format!("layers.{layer}.experts.{expert}.down"),
                    layer,
                    expert,
                    "down",
                ));
        }
    }

    let (tmp, bytes) = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();

    let idx = file.moe_experts().unwrap();
    assert_eq!(idx.n_experts, Some(4));
    assert_eq!(idx.top_k, Some(2));
    assert_eq!(idx.n_shared, Some(1));
    assert_eq!(idx.model_arch.as_deref(), Some("toy-moe"));
    assert_eq!(idx.len(), 2 * 4 * 3);
    assert_eq!(idx.groups.len(), 2 * 4 * 3);

    let group = idx.group(1, Some(3), &MoeRole::Down).unwrap();
    assert_eq!(group.tensor_names, vec!["layers.1.experts.3.down"]);

    let lazy = LazyRsmfFile::open(SliceRangeReader::new(bytes)).unwrap();
    let lazy_idx = lazy.moe_experts().unwrap();
    assert_eq!(lazy_idx, idx);
}

#[test]
fn shared_expert_group_is_indexed_without_expert_id() {
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "layers.0.shared.up".into(),
        dtype: LogicalDtype::F32,
        shape: vec![2, 2],
        metadata: vec![
            ("moe.layer".into(), "0".into()),
            ("moe.shared".into(), "1".into()),
            ("moe.role".into(), "up".into()),
        ],
        canonical: VariantInput::canonical_raw(f32_bytes(4, 1.0)),
        packed: vec![],
    });

    let (tmp, _bytes) = write_file(writer);
    let idx = RsmfFile::open(tmp.path()).unwrap().moe_experts().unwrap();
    assert_eq!(idx.groups.len(), 1);
    assert_eq!(idx.groups[0].expert_id, None);
    assert!(idx.groups[0].shared);
    assert_eq!(idx.groups[0].role, MoeRole::Up);
}

#[test]
fn invalid_decimal_metadata_is_structural_error() {
    let mut tensor = moe_tensor("bad", 0, 0, "gate");
    for (key, value) in &mut tensor.metadata {
        if key == "moe.expert" {
            *value = "zero".into();
        }
    }

    let (tmp, _bytes) = write_file(RsmfWriter::new().with_tensor(tensor));
    let file = RsmfFile::open(tmp.path()).unwrap();
    let err = file.moe_experts().unwrap_err();
    assert!(matches!(err, RsmfError::Structural(_)));
    assert!(err.to_string().contains("moe.expert"));
}

#[test]
fn file_without_moe_metadata_yields_empty_index() {
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
    let idx = RsmfFile::open(tmp.path()).unwrap().moe_experts().unwrap();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
    assert!(idx.groups.is_empty());
}

#[test]
fn unaware_reader_path_still_reads_tensor_bytes() {
    let writer = RsmfWriter::new().with_tensor(moe_tensor("expert.up", 0, 1, "up"));
    let (tmp, _bytes) = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();

    let view = file.tensor_view("expert.up").unwrap();
    let values = view.decode_f32().unwrap();
    assert_eq!(values, vec![1.0; 4]);
}
