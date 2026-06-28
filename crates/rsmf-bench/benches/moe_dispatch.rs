//! MoE dispatch and prepared-runtime throughput.

use std::collections::HashMap;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{
    DeviceDescriptor, DeviceKind, LogicalDtype, MemoryTier, PLACEMENT_FLAG_PIN, PLACEMENT_VERSION,
    PlacementManifest, PlacementRecord, RsmfFile,
};
#[cfg(feature = "wgpu")]
use rsmf_moe_runtime::RuntimeBackend;
use rsmf_moe_runtime::{MoeRoutingPolicy, MoeRuntime, MoeRuntimeOptions};
use tempfile::{TempDir, tempdir};

fn bench_moe_dispatch(c: &mut Criterion) {
    bench_batch_grouping(c);
    bench_cpu_prepared_runtime(c);
    #[cfg(feature = "wgpu")]
    bench_wgpu_prepared_runtime(c);
}

fn bench_batch_grouping(c: &mut Criterion) {
    const TOKENS: usize = 4096;
    const EXPERTS: u32 = 8;

    let assignments = deterministic_assignments(TOKENS, EXPERTS);
    let mut group = c.benchmark_group("moe_dispatch");
    group.throughput(Throughput::Elements(TOKENS as u64));
    group.bench_function("batch_by_destination", |b| {
        b.iter(|| {
            let batches = rsmf_moe_runtime::batch_by_destination(black_box(&assignments));
            black_box(batches);
        });
    });
    group.finish();
}

fn bench_cpu_prepared_runtime(c: &mut Criterion) {
    let cases = [
        BenchCase {
            name: "tiny",
            tokens: 4,
            experts: 2,
            width: 2,
            hidden: 2,
        },
        BenchCase {
            name: "scale_tokens",
            tokens: 128,
            experts: 4,
            width: 16,
            hidden: 32,
        },
        BenchCase {
            name: "scale_experts",
            tokens: 64,
            experts: 8,
            width: 16,
            hidden: 32,
        },
    ];

    let mut group = c.benchmark_group("moe_runtime_cpu");
    for case in cases {
        let top1_fixture =
            build_fixture(case, None).expect("CPU MoE top-1 benchmark fixture should build");
        let topk_fixture =
            build_fixture(case, Some(2)).expect("CPU MoE top-k benchmark fixture should build");
        let input = deterministic_input(case.tokens, case.width);

        let top1_runtime =
            MoeRuntime::new(top1_fixture.open(), MoeRuntimeOptions::default()).expect("runtime");
        let top1_plan = top1_runtime
            .prepare_layer(0, case.width)
            .expect("top-1 plan");
        group.throughput(Throughput::Elements(case.tokens as u64));
        group.bench_with_input(
            BenchmarkId::new("top1_prepared", case.name),
            &case,
            |b, _| {
                b.iter(|| {
                    let output = top1_runtime
                        .run_prepared_layer_top1(black_box(&top1_plan), black_box(&input))
                        .expect("top-1 prepared run");
                    black_box(output);
                });
            },
        );

        let topk_runtime = MoeRuntime::new(
            topk_fixture.open(),
            MoeRuntimeOptions {
                routing: MoeRoutingPolicy::TopK {
                    k: 2,
                    normalize: true,
                },
                ..MoeRuntimeOptions::default()
            },
        )
        .expect("runtime");
        let topk_plan = topk_runtime
            .prepare_layer(0, case.width)
            .expect("top-k plan");
        group.bench_with_input(
            BenchmarkId::new("topk_prepared", case.name),
            &case,
            |b, _| {
                b.iter(|| {
                    let output = topk_runtime
                        .run_prepared_layer_routed(black_box(&topk_plan), black_box(&input))
                        .expect("top-k prepared run");
                    black_box(output);
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "wgpu")]
fn bench_wgpu_prepared_runtime(c: &mut Criterion) {
    let case = BenchCase {
        name: "scale_tokens",
        tokens: 128,
        experts: 4,
        width: 16,
        hidden: 32,
    };
    let fixture = match build_fixture(case, Some(2)) {
        Ok(fixture) => fixture,
        Err(error) => {
            eprintln!("skipping moe_runtime_wgpu bench: fixture build failed: {error:#}");
            return;
        }
    };
    let input = deterministic_input(case.tokens, case.width);
    let runtime = match MoeRuntime::new(
        fixture.open(),
        MoeRuntimeOptions {
            prefer_wgpu: true,
            routing: MoeRoutingPolicy::TopK {
                k: 2,
                normalize: true,
            },
            ..MoeRuntimeOptions::default()
        },
    ) {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("skipping moe_runtime_wgpu bench: runtime init failed: {error}");
            return;
        }
    };
    if !matches!(runtime.backend(), RuntimeBackend::WgpuCompute { .. }) {
        eprintln!("skipping moe_runtime_wgpu bench: WGPU backend unavailable");
        return;
    }
    let plan = match runtime.prepare_layer(0, case.width) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("skipping moe_runtime_wgpu bench: plan failed: {error}");
            return;
        }
    };

    let mut group = c.benchmark_group("moe_runtime_wgpu");
    group.throughput(Throughput::Elements(case.tokens as u64));
    group.bench_function("topk_cache_miss", |b| {
        b.iter_batched(
            || {
                let fixture = build_fixture(case, Some(2)).expect("fixture");
                let runtime = MoeRuntime::new(
                    fixture.open(),
                    MoeRuntimeOptions {
                        prefer_wgpu: true,
                        routing: MoeRoutingPolicy::TopK {
                            k: 2,
                            normalize: true,
                        },
                        ..MoeRuntimeOptions::default()
                    },
                )
                .expect("runtime");
                let plan = runtime.prepare_layer(0, case.width).expect("plan");
                (fixture, runtime, plan)
            },
            |(_fixture, runtime, plan)| {
                let output = runtime
                    .run_prepared_layer_routed(black_box(&plan), black_box(&input))
                    .expect("WGPU cache-miss run");
                let misses = output
                    .report
                    .device_runs
                    .iter()
                    .map(|run| run.weight_cache_misses)
                    .sum::<usize>();
                assert!(misses > 0);
                black_box(output);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("topk_cache_hit", |b| {
        let _warmup = runtime
            .run_prepared_layer_routed(&plan, &input)
            .expect("WGPU cache warmup");
        b.iter(|| {
            let output = runtime
                .run_prepared_layer_routed(black_box(&plan), black_box(&input))
                .expect("WGPU cache-hit run");
            let misses = output
                .report
                .device_runs
                .iter()
                .map(|run| run.weight_cache_misses)
                .sum::<usize>();
            assert_eq!(misses, 0);
            black_box(output);
        });
    });
    group.finish();
}

#[derive(Debug, Clone, Copy)]
struct BenchCase {
    name: &'static str,
    tokens: usize,
    experts: u32,
    width: usize,
    hidden: usize,
}

struct Fixture {
    _dir: TempDir,
    master_path: PathBuf,
    shards: Vec<(u64, PathBuf)>,
}

impl Fixture {
    fn open(&self) -> RsmfFile {
        RsmfFile::open_with_shards(&self.master_path, self.shards.clone())
            .expect("fixture should open")
    }
}

fn build_fixture(case: BenchCase, top_k: Option<u32>) -> Result<Fixture> {
    let dir = tempdir()?;
    let master_path = dir.path().join(format!("{}-moe.rsmf", case.name));
    let tensors = tensors(case)?;
    let tensor_bytes: HashMap<String, Vec<u8>> = tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.canonical.bytes.clone()))
        .collect();
    let writer = top_k
        .into_iter()
        .fold(RsmfWriter::new(), |writer, top_k| {
            writer.with_metadata("moe.top_k", top_k.to_string())
        })
        .with_metadata("moe.n_experts", case.experts.to_string())
        .with_placement_manifest(&placement(case.experts))?;
    let writer = tensors
        .into_iter()
        .fold(writer, |writer, tensor| writer.with_tensor(tensor));
    writer
        .write_to_path(&master_path)
        .context("write fixture master")?;
    let shards = write_shards(dir.path(), &master_path, &tensor_bytes)?;
    Ok(Fixture {
        _dir: dir,
        master_path,
        shards,
    })
}

fn tensors(case: BenchCase) -> Result<Vec<TensorInput>> {
    let mut tensors = Vec::with_capacity(1 + case.experts as usize * 2);
    tensors.push(tensor(
        "layers.0.router.weight",
        0,
        [u64::from(case.experts), case.width as u64],
        moe_meta(0, None, "router"),
        &router_values(case.experts, case.width),
    ));
    for expert in 0..case.experts {
        let shard_id = u64::from(expert) + 1;
        tensors.push(tensor(
            &format!("layers.0.experts.{expert}.up"),
            shard_id,
            [case.hidden as u64, case.width as u64],
            moe_meta(0, Some(expert), "up"),
            &matrix_values(case.hidden, case.width, expert, 0.07),
        ));
        tensors.push(tensor(
            &format!("layers.0.experts.{expert}.down"),
            shard_id,
            [case.width as u64, case.hidden as u64],
            moe_meta(0, Some(expert), "down"),
            &matrix_values(case.width, case.hidden, expert, 0.03),
        ));
    }
    Ok(tensors)
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

fn placement(experts: u32) -> PlacementManifest {
    let devices = vec![
        DeviceDescriptor {
            id: 0,
            kind: DeviceKind::Wgpu,
            tier: MemoryTier::Vram,
            capacity_bytes: 0,
            bandwidth_mbps: 0,
            metadata: vec![("name".into(), "logical-wgpu-0".into())],
        },
        DeviceDescriptor {
            id: 1,
            kind: DeviceKind::Wgpu,
            tier: MemoryTier::Vram,
            capacity_bytes: 0,
            bandwidth_mbps: 0,
            metadata: vec![("name".into(), "logical-wgpu-1".into())],
        },
    ];
    let placements = (0..experts)
        .map(|expert| PlacementRecord {
            shard_id: u64::from(expert) + 1,
            primary_device: expert % 2,
            prefetch_priority: 0,
            flags: PLACEMENT_FLAG_PIN,
            replicas: Vec::new(),
        })
        .collect();
    PlacementManifest {
        version: PLACEMENT_VERSION,
        metadata: Vec::new(),
        devices,
        placements,
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

fn deterministic_assignments(tokens: usize, experts: u32) -> Vec<u32> {
    (0..tokens)
        .map(|token| ((token * 1_103_515_245 + 12_345) as u32) % experts)
        .collect()
}

fn deterministic_input(tokens: usize, width: usize) -> Vec<f32> {
    (0..tokens * width)
        .map(|idx| ((idx % 17) as f32 - 8.0) / 8.0)
        .collect()
}

fn router_values(experts: u32, width: usize) -> Vec<f32> {
    (0..experts as usize * width)
        .map(|idx| {
            let expert = idx / width;
            let col = idx % width;
            if col == expert % width {
                1.0
            } else {
                ((expert + col) % 5) as f32 * 0.05
            }
        })
        .collect()
}

fn matrix_values(rows: usize, cols: usize, expert: u32, scale: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let row = idx / cols;
            let col = idx % cols;
            if row % cols == col {
                1.0 + expert as f32 * scale
            } else {
                ((row + col + expert as usize) % 7) as f32 * scale
            }
        })
        .collect()
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
