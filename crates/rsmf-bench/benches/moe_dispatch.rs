//! MoE dispatch grouping throughput.

use std::collections::HashMap;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use anyhow::Result;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{
    DeviceDescriptor, DeviceKind, LogicalDtype, MemoryTier, PLACEMENT_FLAG_PIN, PLACEMENT_VERSION,
    PlacementManifest, PlacementRecord, RsmfFile,
};
use rsmf_moe_runtime::{MoeRoutingPolicy, MoeRuntime, MoeRuntimeOptions, batch_by_destination};
use tempfile::{TempDir, tempdir};

fn bench_moe_dispatch(c: &mut Criterion) {
    const TOKENS: usize = 4096;
    const EXPERTS: u32 = 8;

    let assignments: Vec<u32> = (0..TOKENS)
        .map(|token| ((token * 1103515245 + 12345) as u32) % EXPERTS)
        .collect();

    let mut group = c.benchmark_group("moe_dispatch");
    group.throughput(Throughput::Elements(TOKENS as u64));
    group.bench_function("batch_by_destination", |b| {
        b.iter(|| {
            let batches = batch_by_destination(black_box(&assignments));
            black_box(batches);
        });
    });
    match tiny_topk_fixture() {
        Ok(fixture) => {
            let runtime = MoeRuntime::new(
                fixture.open(),
                MoeRuntimeOptions {
                    routing: MoeRoutingPolicy::TopK {
                        k: 2,
                        normalize: true,
                    },
                    ..MoeRuntimeOptions::default()
                },
            )
            .expect("tiny MoE runtime should initialize");
            let plan = runtime
                .prepare_layer(0, 2)
                .expect("tiny MoE layer should prepare");
            let input = vec![2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0];
            group.bench_function("prepared_topk_routed_layer", |b| {
                b.iter(|| {
                    let output = runtime
                        .run_prepared_layer_routed(black_box(&plan), black_box(&input))
                        .expect("tiny routed MoE layer should run");
                    black_box(output);
                });
            });
        }
        Err(error) => {
            eprintln!("skipping prepared_topk_routed_layer bench: {error:#}");
        }
    }
    group.finish();
}

struct TinyFixture {
    _dir: TempDir,
    master_path: PathBuf,
    shards: Vec<(u64, PathBuf)>,
}

impl TinyFixture {
    fn open(&self) -> RsmfFile {
        RsmfFile::open_with_shards(&self.master_path, self.shards.clone())
            .expect("tiny fixture should open")
    }
}

fn tiny_topk_fixture() -> Result<TinyFixture> {
    let dir = tempdir()?;
    let master_path = dir.path().join("tiny-moe.rsmf");
    let tensors = tiny_tensors();
    let tensor_bytes: HashMap<String, Vec<u8>> = tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.canonical.bytes.clone()))
        .collect();
    RsmfWriter::new()
        .with_metadata("moe.top_k", "2")
        .with_placement_manifest(&tiny_placement())?
        .with_tensor(tensors[0].clone())
        .with_tensor(tensors[1].clone())
        .with_tensor(tensors[2].clone())
        .with_tensor(tensors[3].clone())
        .with_tensor(tensors[4].clone())
        .write_to_path(&master_path)?;
    let shards = write_shards(dir.path(), &master_path, &tensor_bytes)?;
    Ok(TinyFixture {
        _dir: dir,
        master_path,
        shards,
    })
}

fn tiny_tensors() -> Vec<TensorInput> {
    vec![
        tensor(
            "layers.0.router.weight",
            0,
            [2, 2],
            moe_meta(0, None, "router"),
            &[1.0, 0.0, 0.0, 1.0],
        ),
        tensor(
            "layers.0.experts.0.up",
            1,
            [2, 2],
            moe_meta(0, Some(0), "up"),
            &[1.0, 0.0, 0.0, 1.0],
        ),
        tensor(
            "layers.0.experts.0.down",
            1,
            [2, 2],
            moe_meta(0, Some(0), "down"),
            &[1.0, 0.0, 0.0, 1.0],
        ),
        tensor(
            "layers.0.experts.1.up",
            2,
            [2, 2],
            moe_meta(0, Some(1), "up"),
            &[2.0, 0.0, 0.0, 2.0],
        ),
        tensor(
            "layers.0.experts.1.down",
            2,
            [2, 2],
            moe_meta(0, Some(1), "down"),
            &[1.0, 0.0, 0.0, 1.0],
        ),
    ]
}

fn tensor(
    name: &str,
    shard_id: u64,
    shape: [u64; 2],
    metadata: Vec<(String, String)>,
    values: &[f32],
) -> TensorInput {
    TensorInput {
        shard_id,
        name: name.into(),
        dtype: LogicalDtype::F32,
        shape: shape.into(),
        metadata,
        canonical: VariantInput::canonical_raw(f32_bytes(values)),
        packed: Vec::new(),
    }
}

fn moe_meta(layer: u32, expert: Option<u32>, role: &str) -> Vec<(String, String)> {
    let mut out = vec![
        ("moe.layer".into(), layer.to_string()),
        ("moe.role".into(), role.into()),
    ];
    if let Some(expert) = expert {
        out.push(("moe.expert".into(), expert.to_string()));
    }
    out
}

fn tiny_placement() -> PlacementManifest {
    PlacementManifest {
        version: PLACEMENT_VERSION,
        metadata: Vec::new(),
        devices: vec![
            DeviceDescriptor {
                id: 0,
                kind: DeviceKind::Wgpu,
                tier: MemoryTier::Vram,
                capacity_bytes: 0,
                bandwidth_mbps: 0,
                metadata: Vec::new(),
            },
            DeviceDescriptor {
                id: 1,
                kind: DeviceKind::Wgpu,
                tier: MemoryTier::Vram,
                capacity_bytes: 0,
                bandwidth_mbps: 0,
                metadata: Vec::new(),
            },
        ],
        placements: vec![
            PlacementRecord {
                shard_id: 1,
                primary_device: 0,
                prefetch_priority: 0,
                flags: PLACEMENT_FLAG_PIN,
                replicas: Vec::new(),
            },
            PlacementRecord {
                shard_id: 2,
                primary_device: 1,
                prefetch_priority: 0,
                flags: PLACEMENT_FLAG_PIN,
                replicas: Vec::new(),
            },
        ],
    }
}

fn write_shards(
    dir: &Path,
    master_path: &Path,
    tensor_bytes: &HashMap<String, Vec<u8>>,
) -> Result<Vec<(u64, PathBuf)>> {
    let master = RsmfFile::open(master_path)?;
    let mut shard_bytes: HashMap<u64, Vec<u8>> = HashMap::new();
    for tensor in &master.manifest().tensors {
        if tensor.shard_id == 0 {
            continue;
        }
        let variant = &master.manifest().variants[tensor.canonical_variant as usize];
        let bytes = &tensor_bytes[&tensor.name];
        let shard = shard_bytes.entry(tensor.shard_id).or_default();
        let start = variant.section_relative_offset as usize;
        let end = start + bytes.len();
        if shard.len() < end {
            shard.resize(end, 0);
        }
        shard[start..end].copy_from_slice(bytes);
    }
    let mut out = Vec::new();
    let mut ids = shard_bytes.keys().copied().collect::<Vec<_>>();
    ids.sort_unstable();
    for shard_id in ids {
        let path = dir.join(format!("shard-{shard_id}.bin"));
        fs::write(&path, &shard_bytes[&shard_id])?;
        out.push((shard_id, path));
    }
    Ok(out)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

criterion_group!(benches, bench_moe_dispatch);
criterion_main!(benches);
