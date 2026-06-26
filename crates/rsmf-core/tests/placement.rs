//! Integration tests for the PlacementManifest custom section.

use std::io::Write;

use rsmf_core::writer::{CustomSectionInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{
    DeviceDescriptor, DeviceKind, LogicalDtype, MemoryTier, PLACEMENT_FLAG_PIN,
    PLACEMENT_SECTION_KIND, PLACEMENT_VERSION, PlacementManifest, PlacementRecord, RsmfFile,
};
use tempfile::NamedTempFile;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn base_writer(shard_id: u64) -> RsmfWriter {
    RsmfWriter::new().with_tensor(TensorInput {
        shard_id,
        name: "weights".into(),
        dtype: LogicalDtype::F32,
        shape: vec![4],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
        packed: Vec::new(),
    })
}

fn placement(shard_id: u64) -> PlacementManifest {
    PlacementManifest {
        version: PLACEMENT_VERSION,
        metadata: vec![("planner".into(), "test".into())],
        devices: vec![
            DeviceDescriptor {
                id: 0,
                kind: DeviceKind::Cuda,
                tier: MemoryTier::Vram,
                capacity_bytes: 1 << 30,
                bandwidth_mbps: 900_000,
                metadata: vec![("name".into(), "cuda:0".into())],
            },
            DeviceDescriptor {
                id: 1,
                kind: DeviceKind::Cpu,
                tier: MemoryTier::Ram,
                capacity_bytes: 16 << 30,
                bandwidth_mbps: 50_000,
                metadata: Vec::new(),
            },
        ],
        placements: vec![PlacementRecord {
            shard_id,
            primary_device: 0,
            prefetch_priority: 7,
            flags: PLACEMENT_FLAG_PIN,
            replicas: vec![1],
        }],
    }
}

#[test]
fn writer_reader_round_trips_placement_manifest_and_custom_sections() {
    let placement = placement(0);
    let bytes = base_writer(0)
        .with_custom_section(CustomSectionInput::new(129, b"sidecar".to_vec()))
        .with_placement_manifest(&placement)
        .unwrap()
        .write_to_bytes()
        .unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();
    assert_eq!(file.placement_manifest().unwrap(), Some(placement));

    let custom = file.custom_sections().unwrap();
    assert_eq!(custom.len(), 2);
    assert!(custom.iter().any(|s| {
        s.kind == PLACEMENT_SECTION_KIND && PlacementManifest::decode(&s.bytes).is_ok()
    }));
    assert!(
        custom
            .iter()
            .any(|s| s.kind == 129 && s.bytes == b"sidecar")
    );
}

#[test]
fn writer_rejects_placement_for_missing_shard_id() {
    let placement = placement(99);
    let err = base_writer(0)
        .with_placement_manifest(&placement)
        .unwrap()
        .write_to_bytes()
        .unwrap_err();
    assert!(
        err.to_string().contains("placement references shard_id 99"),
        "unexpected error: {err}"
    );
}

#[test]
fn encoder_rejects_placement_with_missing_primary_device() {
    let mut placement = placement(0);
    placement.placements[0].primary_device = 9;
    let err = placement.encode().unwrap_err();
    assert!(
        err.to_string().contains("missing primary_device 9"),
        "unexpected error: {err}"
    );
}
