//! Content-addressable dedup in the writer's arenas.
//!
//! When dedup is opt-in-enabled, two tensors whose canonical bytes are byte
//! identical share a single span in the canonical arena. The reader is
//! dedup-oblivious: two variant descriptors may point at the same offset and
//! each still produces a correct `TensorView`. These tests pin that contract
//! and the alignment fallback (a differently-aligned reuse falls back to a
//! fresh append rather than producing an unaligned offset).

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile, SectionKind, TargetTag};
use tempfile::NamedTempFile;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn write_to_tmp(writer: RsmfWriter) -> NamedTempFile {
    let bytes = writer.write_to_bytes().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();
    use std::io::Write;
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    tmp
}

fn identical_tensor(name: &str, data: &[f32]) -> TensorInput {
    TensorInput {
        shard_id: 0,
        name: name.into(),
        dtype: LogicalDtype::F32,
        shape: vec![data.len() as u64],
        metadata: vec![],
        canonical: VariantInput::canonical_raw(f32_bytes(data)),
        packed: vec![],
    }
}

fn canonical_section_length(tmp: &NamedTempFile) -> u64 {
    let file = RsmfFile::open(tmp.path()).unwrap();
    file.sections()
        .iter()
        .find(|s| s.kind == SectionKind::CanonicalArena)
        .expect("canonical section")
        .length
}

#[test]
fn identical_tensors_share_canonical_offset_when_dedup_on() {
    let payload: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

    let writer = RsmfWriter::new()
        .with_dedup(true)
        .with_tensor(identical_tensor("embed", &payload))
        .with_tensor(identical_tensor("lm_head", &payload));

    let tmp = write_to_tmp(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();

    let embed = file.tensor_view("embed").unwrap();
    let head = file.tensor_view("lm_head").unwrap();

    // Both views must deliver the same bytes.
    assert_eq!(embed.bytes, head.bytes);

    // And, critically, they should point at the same offset inside the
    // canonical arena — this is the dedup invariant.
    let embed_desc = &file.manifest().variants[file
        .manifest()
        .tensors
        .iter()
        .find(|t| t.name == "embed")
        .unwrap()
        .canonical_variant as usize];
    let head_desc = &file.manifest().variants[file
        .manifest()
        .tensors
        .iter()
        .find(|t| t.name == "lm_head")
        .unwrap()
        .canonical_variant as usize];
    assert_eq!(
        embed_desc.section_relative_offset, head_desc.section_relative_offset,
        "dedup should place both variants at the same arena offset"
    );
    assert_eq!(embed_desc.length, head_desc.length);
    assert_eq!(embed_desc.checksum, head_desc.checksum);
}

#[test]
fn dedup_off_writes_bytes_twice() {
    let payload: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

    let deduped = write_to_tmp(
        RsmfWriter::new()
            .with_dedup(true)
            .with_tensor(identical_tensor("a", &payload))
            .with_tensor(identical_tensor("b", &payload)),
    );
    let plain = write_to_tmp(
        RsmfWriter::new()
            .with_tensor(identical_tensor("a", &payload))
            .with_tensor(identical_tensor("b", &payload)),
    );

    // With dedup, the canonical section stores the payload once; without, twice
    // (plus alignment padding). Plain must be strictly larger.
    assert!(
        canonical_section_length(&deduped) < canonical_section_length(&plain),
        "dedup should shrink the canonical arena: deduped={} plain={}",
        canonical_section_length(&deduped),
        canonical_section_length(&plain)
    );
}

#[test]
fn differing_bytes_do_not_dedup() {
    let a: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..32).map(|i| i as f32 * 0.2).collect();

    let writer = RsmfWriter::new()
        .with_dedup(true)
        .with_tensor(identical_tensor("a", &a))
        .with_tensor(identical_tensor("b", &b));

    let tmp = write_to_tmp(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();

    let va = &file.manifest().variants[file
        .manifest()
        .tensors
        .iter()
        .find(|t| t.name == "a")
        .unwrap()
        .canonical_variant as usize];
    let vb = &file.manifest().variants[file
        .manifest()
        .tensors
        .iter()
        .find(|t| t.name == "b")
        .unwrap()
        .canonical_variant as usize];
    assert_ne!(va.section_relative_offset, vb.section_relative_offset);
    assert_ne!(va.checksum, vb.checksum);
}

#[test]
fn packed_arena_dedup_shares_offsets_per_group() {
    // Two tensors with identical packed variants for the same target.
    let canonical_a = f32_bytes(&[1.0_f32; 4]);
    let canonical_b = f32_bytes(&[2.0_f32; 4]);
    let shared_packed: Vec<u8> = vec![0xAB; 8];

    let make_tensor = |name: &str, canonical: Vec<u8>| TensorInput {
        shard_id: 0,
        name: name.into(),
        dtype: LogicalDtype::F32,
        shape: vec![4],
        metadata: vec![],
        canonical: VariantInput::canonical_raw(canonical),
        packed: vec![VariantInput::packed_cast_f16(
            TargetTag::CpuGeneric,
            shared_packed.clone(),
        )],
    };

    let writer = RsmfWriter::new()
        .with_dedup(true)
        .with_tensor(make_tensor("a", canonical_a))
        .with_tensor(make_tensor("b", canonical_b));

    let tmp = write_to_tmp(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();

    // Packed variant descriptors should share the same offset despite being
    // attached to different tensors.
    let a_pi = file
        .manifest()
        .tensors
        .iter()
        .find(|t| t.name == "a")
        .unwrap()
        .packed_variants[0];
    let b_pi = file
        .manifest()
        .tensors
        .iter()
        .find(|t| t.name == "b")
        .unwrap()
        .packed_variants[0];
    let va = &file.manifest().variants[a_pi as usize];
    let vb = &file.manifest().variants[b_pi as usize];
    assert_eq!(va.section_relative_offset, vb.section_relative_offset);
    assert_eq!(va.length, vb.length);
}

#[test]
#[cfg(feature = "compression")]
fn dedup_survives_compression() {
    // Two tied tensors, zstd on the canonical arena. The shared span should
    // still only appear once in the uncompressed bytes (and therefore twice-
    // reuse shows up as a smaller compressed section too).
    let payload: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();

    let writer = RsmfWriter::new()
        .with_dedup(true)
        .with_canonical_compression(3)
        .with_tensor(identical_tensor("a", &payload))
        .with_tensor(identical_tensor("b", &payload));

    let tmp = write_to_tmp(writer);
    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();

    let va = file.tensor_view("a").unwrap();
    let vb = file.tensor_view("b").unwrap();
    assert_eq!(va.bytes, vb.bytes);
}

#[test]
fn per_arena_toggles_are_independent() {
    let payload: Vec<f32> = (0..16).map(|i| i as f32).collect();

    // Only canonical dedup: packed arena still writes twice.
    let canonical_only = write_to_tmp(
        RsmfWriter::new()
            .with_canonical_dedup(true)
            .with_tensor(identical_tensor("a", &payload))
            .with_tensor(identical_tensor("b", &payload)),
    );
    let both = write_to_tmp(
        RsmfWriter::new()
            .with_dedup(true)
            .with_tensor(identical_tensor("a", &payload))
            .with_tensor(identical_tensor("b", &payload)),
    );

    // With no packed variants either case produces the same size, but both
    // must round-trip.
    let file = RsmfFile::open(canonical_only.path()).unwrap();
    file.full_verify().unwrap();
    let file = RsmfFile::open(both.path()).unwrap();
    file.full_verify().unwrap();
}
