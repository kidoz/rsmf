//! Round-trip tests for the `adapter.*` metadata convention.
//!
//! Writers attach adapter metadata to regular tensors; readers group them via
//! [`RsmfFile::adapters`]. The format itself is unchanged — these tests verify
//! that a file packed with the convention produces a correct reader-side
//! index, that conflicting kinds/ranks are rejected, and that files without
//! adapter annotations come back as an empty index.

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{AdapterKind, AdapterRole, LogicalDtype, RsmfFile};
use tempfile::NamedTempFile;

#[allow(clippy::too_many_arguments)]
fn lora_tensor(
    name: &str,
    adapter_name: &str,
    role: &str,
    target: &str,
    rank: u32,
    alpha: f32,
    shape: Vec<u64>,
    bytes: Vec<u8>,
) -> TensorInput {
    TensorInput {
        shard_id: 0,
        name: name.into(),
        dtype: LogicalDtype::F32,
        shape,
        metadata: vec![
            ("adapter.name".into(), adapter_name.into()),
            ("adapter.kind".into(), "lora".into()),
            ("adapter.role".into(), role.into()),
            ("adapter.target".into(), target.into()),
            ("adapter.rank".into(), rank.to_string()),
            ("adapter.alpha".into(), alpha.to_string()),
        ],
        canonical: VariantInput::canonical_raw(bytes),
        packed: vec![],
    }
}

fn f32_bytes(n: usize, fill: f32) -> Vec<u8> {
    (0..n).flat_map(|_| fill.to_le_bytes()).collect()
}

fn write_file(writer: RsmfWriter) -> NamedTempFile {
    let bytes = writer.write_to_bytes().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();
    use std::io::Write;
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    tmp
}

#[test]
fn round_trip_groups_two_lora_adapters() {
    // Two adapters: q_proj and v_proj on layer 0 of a llama-ish target.
    let writer = RsmfWriter::new()
        .with_metadata("adapter.base_model_name", "meta-llama/Llama-3.1-8B")
        .with_metadata("adapter.base_model_sha256", "deadbeef")
        .with_tensor(lora_tensor(
            "q_lora_A",
            "q_proj_lora",
            "lora_a",
            "model.layers.0.self_attn.q_proj.weight",
            8,
            16.0,
            vec![8, 4096],
            f32_bytes(8 * 4096, 0.01),
        ))
        .with_tensor(lora_tensor(
            "q_lora_B",
            "q_proj_lora",
            "lora_b",
            "model.layers.0.self_attn.q_proj.weight",
            8,
            16.0,
            vec![4096, 8],
            f32_bytes(4096 * 8, 0.02),
        ))
        .with_tensor(lora_tensor(
            "v_lora_A",
            "v_proj_lora",
            "lora_a",
            "model.layers.0.self_attn.v_proj.weight",
            4,
            8.0,
            vec![4, 4096],
            f32_bytes(4 * 4096, 0.03),
        ))
        .with_tensor(lora_tensor(
            "v_lora_B",
            "v_proj_lora",
            "lora_b",
            "model.layers.0.self_attn.v_proj.weight",
            4,
            8.0,
            vec![4096, 4],
            f32_bytes(4096 * 4, 0.04),
        ));

    let tmp = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();

    let idx = file.adapters().unwrap();
    assert_eq!(idx.len(), 2);
    assert_eq!(
        idx.base_model_name.as_deref(),
        Some("meta-llama/Llama-3.1-8B")
    );
    assert_eq!(idx.base_model_sha256.as_deref(), Some("deadbeef"));

    let q = idx.get("q_proj_lora").expect("q adapter indexed");
    assert_eq!(q.kind, AdapterKind::Lora);
    assert_eq!(q.rank, Some(8));
    assert_eq!(q.alpha, Some(16.0));
    assert_eq!(q.effective_scale(), Some(2.0));
    assert_eq!(q.entries.len(), 2);

    let a_entry = q.entry_for_role(&AdapterRole::LoraA).unwrap();
    assert_eq!(a_entry.tensor_name, "q_lora_A");
    assert_eq!(
        a_entry.target.as_deref(),
        Some("model.layers.0.self_attn.q_proj.weight")
    );

    let b_entry = q.entry_for_role(&AdapterRole::LoraB).unwrap();
    assert_eq!(b_entry.tensor_name, "q_lora_B");

    // Reader returns the actual bytes back.
    let a_view = file.tensor_view("q_lora_A").unwrap();
    let a_f32 = a_view.decode_f32().unwrap();
    assert_eq!(a_f32.len(), 8 * 4096);
    assert!((a_f32[0] - 0.01).abs() < 1e-6);

    let v = idx.get("v_proj_lora").unwrap();
    assert_eq!(v.rank, Some(4));
    assert_eq!(v.effective_scale(), Some(2.0));
}

#[test]
fn dora_adapter_groups_magnitude_tensor() {
    let writer = RsmfWriter::new()
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "dora_A".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4, 32],
            metadata: vec![
                ("adapter.name".into(), "d".into()),
                ("adapter.kind".into(), "dora".into()),
                ("adapter.role".into(), "lora_a".into()),
                ("adapter.rank".into(), "4".into()),
            ],
            canonical: VariantInput::canonical_raw(f32_bytes(4 * 32, 0.1)),
            packed: vec![],
        })
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "dora_B".into(),
            dtype: LogicalDtype::F32,
            shape: vec![32, 4],
            metadata: vec![
                ("adapter.name".into(), "d".into()),
                ("adapter.kind".into(), "dora".into()),
                ("adapter.role".into(), "lora_b".into()),
                ("adapter.rank".into(), "4".into()),
            ],
            canonical: VariantInput::canonical_raw(f32_bytes(32 * 4, 0.2)),
            packed: vec![],
        })
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "dora_mag".into(),
            dtype: LogicalDtype::F32,
            shape: vec![32],
            metadata: vec![
                ("adapter.name".into(), "d".into()),
                ("adapter.kind".into(), "dora".into()),
                ("adapter.role".into(), "magnitude".into()),
            ],
            canonical: VariantInput::canonical_raw(f32_bytes(32, 1.0)),
            packed: vec![],
        });

    let tmp = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let idx = file.adapters().unwrap();

    let d = idx.get("d").unwrap();
    assert_eq!(d.kind, AdapterKind::Dora);
    assert_eq!(d.entries.len(), 3);
    assert!(d.entry_for_role(&AdapterRole::Magnitude).is_some());
}

#[test]
fn conflicting_ranks_across_same_adapter_are_rejected() {
    let writer = RsmfWriter::new()
        .with_tensor(lora_tensor(
            "A",
            "adapter1",
            "lora_a",
            "w",
            8,
            16.0,
            vec![8, 16],
            f32_bytes(8 * 16, 0.0),
        ))
        // Same adapter name, different rank — this is a bug in the writer.
        .with_tensor(lora_tensor(
            "B",
            "adapter1",
            "lora_b",
            "w",
            4,
            16.0,
            vec![16, 4],
            f32_bytes(16 * 4, 0.0),
        ));

    let tmp = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let err = file.adapters().unwrap_err();
    assert!(
        err.to_string().contains("rank"),
        "expected rank conflict, got {err}"
    );
}

#[test]
fn conflicting_kinds_across_same_adapter_are_rejected() {
    let mut t1 = lora_tensor(
        "A",
        "adapter1",
        "lora_a",
        "w",
        4,
        8.0,
        vec![4, 8],
        f32_bytes(4 * 8, 0.0),
    );
    let mut t2 = lora_tensor(
        "B",
        "adapter1",
        "lora_b",
        "w",
        4,
        8.0,
        vec![8, 4],
        f32_bytes(8 * 4, 0.0),
    );
    // Replace `adapter.kind` on the second tensor with "dora" to force a conflict.
    for (k, v) in t2.metadata.iter_mut() {
        if k == "adapter.kind" {
            *v = "dora".into();
        }
    }
    // Keep t1 as lora.
    let _ = &mut t1;

    let writer = RsmfWriter::new().with_tensor(t1).with_tensor(t2);
    let tmp = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let err = file.adapters().unwrap_err();
    assert!(
        err.to_string().contains("kind"),
        "expected kind conflict, got {err}"
    );
}

#[test]
fn invalid_rank_string_is_a_structural_error() {
    let mut t = lora_tensor(
        "A",
        "adapter1",
        "lora_a",
        "w",
        4,
        8.0,
        vec![4, 8],
        f32_bytes(4 * 8, 0.0),
    );
    for (k, v) in t.metadata.iter_mut() {
        if k == "adapter.rank" {
            *v = "eight".into();
        }
    }

    let writer = RsmfWriter::new().with_tensor(t);
    let tmp = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let err = file.adapters().unwrap_err();
    assert!(
        err.to_string().contains("adapter.rank"),
        "expected rank parse error, got {err}"
    );
}

#[test]
fn file_without_adapter_metadata_yields_empty_index() {
    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "w".into(),
        dtype: LogicalDtype::F32,
        shape: vec![4],
        metadata: vec![],
        canonical: VariantInput::canonical_raw(f32_bytes(4, 1.0)),
        packed: vec![],
    });
    let tmp = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let idx = file.adapters().unwrap();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
    assert!(idx.base_model_name.is_none());
}

#[test]
fn unknown_kind_and_role_are_preserved_verbatim() {
    let mut t = lora_tensor(
        "x",
        "a",
        "custom_role",
        "w",
        2,
        4.0,
        vec![2, 8],
        f32_bytes(2 * 8, 0.0),
    );
    for (k, v) in t.metadata.iter_mut() {
        if k == "adapter.kind" {
            *v = "mystery_adapter".into();
        }
    }

    let writer = RsmfWriter::new().with_tensor(t);
    let tmp = write_file(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    let idx = file.adapters().unwrap();
    let a = idx.get("a").unwrap();
    assert_eq!(a.kind, AdapterKind::Other("mystery_adapter".into()));
    assert_eq!(a.entries[0].role, AdapterRole::Other("custom_role".into()));
}
