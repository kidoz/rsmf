//! End-to-end tests for the minimal MoE runtime.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{
    DeviceDescriptor, DeviceKind, LogicalDtype, MemoryTier, PLACEMENT_FLAG_PIN, PLACEMENT_VERSION,
    PlacementManifest, PlacementRecord, RsmfFile,
};
use rsmf_moe_runtime::{
    CpuCollectives, ExpertActivation, MoeRoutingPolicy, MoeRuntime, MoeRuntimeError,
    MoeRuntimeOptions, MoeTransferKind, RuntimeBackend, RuntimeLimits,
};
use tempfile::{TempDir, tempdir};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
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

fn tensor(
    name: &str,
    shard_id: u64,
    shape: [u64; 2],
    metadata: Vec<(String, String)>,
    values: &[f32],
    prefetch_group: Option<&str>,
) -> TensorInput {
    let mut canonical = VariantInput::canonical_raw(f32_bytes(values));
    if let Some(group) = prefetch_group {
        canonical = canonical
            .with_prefetch_group(group)
            .with_prefetch_affinity(format!("shard:{shard_id}"));
    }
    TensorInput {
        shard_id,
        name: name.into(),
        dtype: LogicalDtype::F32,
        shape: shape.into(),
        metadata,
        canonical,
        packed: Vec::new(),
    }
}

struct Fixture {
    _dir: TempDir,
    master_path: PathBuf,
    shards: Vec<(u64, PathBuf)>,
}

impl Fixture {
    fn open(&self) -> RsmfFile {
        RsmfFile::open_with_shards(&self.master_path, self.shards.clone()).unwrap()
    }
}

fn fixture_tensors() -> Vec<TensorInput> {
    vec![
        tensor(
            "layers.0.router.weight",
            0,
            [2, 2],
            moe_meta(0, None, "router"),
            &[1.0, 0.0, 0.0, 1.0],
            None,
        ),
        tensor(
            "layers.0.experts.0.up",
            1,
            [2, 2],
            moe_meta(0, Some(0), "up"),
            &[1.0, 0.0, 0.0, 1.0],
            Some("layer0.expert0"),
        ),
        tensor(
            "layers.0.experts.0.down",
            1,
            [2, 2],
            moe_meta(0, Some(0), "down"),
            &[1.0, 0.0, 0.0, 1.0],
            Some("layer0.expert0"),
        ),
        tensor(
            "layers.0.experts.1.up",
            2,
            [2, 2],
            moe_meta(0, Some(1), "up"),
            &[2.0, 0.0, 0.0, 2.0],
            Some("layer0.expert1"),
        ),
        tensor(
            "layers.0.experts.1.down",
            2,
            [2, 2],
            moe_meta(0, Some(1), "down"),
            &[1.0, 0.0, 0.0, 1.0],
            Some("layer0.expert1"),
        ),
    ]
}

fn fixture_placement() -> PlacementManifest {
    PlacementManifest {
        version: PLACEMENT_VERSION,
        metadata: vec![("fixture".into(), "two-shard-moe".into())],
        devices: vec![
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

fn build_fixture() -> Fixture {
    build_fixture_with(Vec::new(), fixture_tensors(), fixture_placement())
}

fn build_fixture_with(
    metadata: Vec<(&str, &str)>,
    tensors: Vec<TensorInput>,
    placement: PlacementManifest,
) -> Fixture {
    let dir = tempdir().unwrap();
    let master_path = dir.path().join("moe.rsmf");

    let tensor_bytes: HashMap<String, Vec<u8>> = tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.canonical.bytes.clone()))
        .collect();

    let writer = metadata
        .into_iter()
        .fold(RsmfWriter::new(), |writer, (key, value)| {
            writer.with_metadata(key, value)
        });
    let writer = tensors
        .into_iter()
        .fold(writer, |writer, tensor| writer.with_tensor(tensor))
        .with_placement_manifest(&placement)
        .unwrap();
    writer.write_to_path(&master_path).unwrap();
    let shards = write_shards(dir.path(), &master_path, &tensor_bytes);

    Fixture {
        _dir: dir,
        master_path,
        shards,
    }
}

fn write_shards(
    dir: &Path,
    master_path: &Path,
    tensor_bytes: &HashMap<String, Vec<u8>>,
) -> Vec<(u64, PathBuf)> {
    let master = RsmfFile::open(master_path).unwrap();
    let mut shard_bytes: HashMap<u64, Vec<u8>> = HashMap::new();
    for tensor in &master.manifest().tensors {
        if tensor.shard_id == 0 {
            continue;
        }
        let variant = &master.manifest().variants[tensor.canonical_variant as usize];
        let bytes = tensor_bytes.get(&tensor.name).unwrap();
        let shard = shard_bytes.entry(tensor.shard_id).or_default();
        let start = variant.section_relative_offset as usize;
        let end = start + bytes.len();
        if shard.len() < end {
            shard.resize(end, 0);
        }
        shard[start..end].copy_from_slice(bytes);
    }

    let mut out = Vec::new();
    let mut ids: Vec<u64> = shard_bytes.keys().copied().collect();
    ids.sort_unstable();
    for shard_id in ids {
        let path = dir.join(format!("shard-{shard_id}.bin"));
        fs::write(&path, &shard_bytes[&shard_id]).unwrap();
        out.push((shard_id, path));
    }
    out
}

#[test]
fn two_shard_runtime_matches_single_device_reference() {
    let fixture = build_fixture();
    let file = fixture.open();
    file.full_verify().unwrap();

    let runtime = MoeRuntime::new(file, MoeRuntimeOptions::default()).unwrap();
    let input = vec![2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0];
    let parallel = runtime.run_layer_top1(0, &input, 2).unwrap();
    let reference = runtime.run_layer_reference_top1(0, &input, 2).unwrap();

    assert_eq!(parallel.output_width, 2);
    assert_eq!(parallel.output, reference);
    assert_eq!(
        parallel.output,
        vec![2.0, 1.0, 2.0, 6.0, 4.0, 0.0, 0.0, 10.0]
    );
    assert_eq!(parallel.report.device_batches.len(), 2);
    assert_eq!(parallel.report.device_batches[0].expert_id, 0);
    assert_eq!(parallel.report.device_batches[0].device_id, 0);
    assert_eq!(parallel.report.device_batches[0].token_indices, vec![0, 2]);
    assert_eq!(
        parallel.report.device_batches[0].token_weights,
        vec![1.0, 1.0]
    );
    assert_eq!(
        parallel.report.device_batches[0].prefetch_groups,
        vec!["layer0.expert0"]
    );
    assert_eq!(parallel.report.device_batches[1].expert_id, 1);
    assert_eq!(parallel.report.device_batches[1].device_id, 1);
    assert_eq!(parallel.report.device_batches[1].token_indices, vec![1, 3]);
    assert_eq!(
        parallel.report.device_batches[1].token_weights,
        vec![1.0, 1.0]
    );
    assert!(matches!(
        parallel.report.backend,
        RuntimeBackend::CpuFallback { .. }
    ));
    assert!(parallel.report.tokens_per_second().is_finite());
}

#[test]
fn prepared_layer_plan_reports_residency_and_reuses_weights() {
    let fixture = build_fixture();
    let runtime = MoeRuntime::new(fixture.open(), MoeRuntimeOptions::default()).unwrap();
    let plan = runtime.prepare_layer(0, 2).unwrap();

    assert_eq!(plan.layer(), 0);
    assert_eq!(plan.input_width(), 2);
    assert_eq!(plan.report().expert_count, 2);
    assert_eq!(plan.report().routing_policy, MoeRoutingPolicy::Top1);
    assert_eq!(plan.report().resident_bytes, 64);
    assert!(plan.report().multi_device);
    assert_eq!(plan.report().devices.len(), 2);
    assert_eq!(plan.report().devices[0].device_id, 0);
    assert_eq!(plan.report().devices[0].resident_bytes, 32);
    assert_eq!(plan.report().devices[0].expert_ids, vec![0]);
    assert_eq!(plan.report().devices[1].device_id, 1);
    assert_eq!(plan.report().devices[1].resident_bytes, 32);
    assert_eq!(plan.report().devices[1].expert_ids, vec![1]);
    assert_eq!(plan.report().transfer_plan.steps.len(), 2);
    assert_eq!(plan.report().transfer_plan.planned_bytes, 64);
    assert_eq!(plan.report().transfer_plan.unsupported_count, 0);
    assert_eq!(
        plan.report().transfer_plan.steps[0].kind,
        MoeTransferKind::HostToDevice
    );
    assert!(!plan.report().collective_plan.required);
    assert!(plan.report().collective_plan.steps.is_empty());

    let input = vec![2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0];
    let output = runtime.run_prepared_layer_top1(&plan, &input).unwrap();
    assert_eq!(output.report.plan, plan.report().clone());
    assert_eq!(output.report.device_runs.len(), 2);
    assert_eq!(output.report.device_runs[0].device_id, 0);
    assert_eq!(output.report.device_runs[0].expert_ids, vec![0]);
    assert_eq!(output.report.device_runs[0].token_count, 2);
    assert_eq!(output.report.device_runs[1].device_id, 1);
    assert_eq!(output.report.device_runs[1].expert_ids, vec![1]);
    assert_eq!(output.report.device_runs[1].token_count, 2);
}

#[test]
fn routed_topk_uses_weighted_expert_combine() {
    let fixture = build_fixture_with(
        vec![("moe.top_k", "2")],
        fixture_tensors(),
        fixture_placement(),
    );
    let runtime = MoeRuntime::new(
        fixture.open(),
        MoeRuntimeOptions {
            routing: MoeRoutingPolicy::TopK {
                k: 2,
                normalize: false,
            },
            ..MoeRuntimeOptions::default()
        },
    )
    .unwrap();
    let input = vec![2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0];

    let routed = runtime.run_layer_routed(0, &input, 2).unwrap();
    let reference = runtime.run_layer_reference_routed(0, &input, 2).unwrap();

    assert_eq!(routed.output_width, 2);
    assert_eq!(routed.output, reference);
    assert_eq!(
        routed.output,
        vec![6.0, 3.0, 3.0, 9.0, 12.0, 0.0, 0.0, 15.0]
    );
    assert_eq!(routed.report.device_batches.len(), 2);
    assert_eq!(routed.report.device_batches[0].expert_id, 0);
    assert_eq!(
        routed.report.device_batches[0].token_indices,
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        routed.report.device_batches[0].token_weights,
        vec![1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(routed.report.device_batches[1].expert_id, 1);
    assert_eq!(
        routed.report.device_batches[1].token_indices,
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        routed.report.device_batches[1].token_weights,
        vec![1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(
        routed.report.plan.routing_policy,
        MoeRoutingPolicy::TopK {
            k: 2,
            normalize: false
        }
    );
}

#[test]
fn routed_topk_checked_reports_reference_match() {
    let fixture = build_fixture_with(
        vec![("moe.top_k", "2")],
        fixture_tensors(),
        fixture_placement(),
    );
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
    .unwrap();
    let input = vec![2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0];

    let checked = runtime.run_layer_routed_checked(0, &input, 2, 0.0).unwrap();

    assert!(checked.comparison.passed);
    assert_eq!(checked.comparison.max_abs_diff, 0.0);
    assert_eq!(checked.routed.output, checked.reference_output);
    assert_eq!(
        checked.routed.report.plan.routing_policy,
        MoeRoutingPolicy::TopK {
            k: 2,
            normalize: true
        }
    );
}

#[test]
fn cpu_collectives_sum_and_gather_partitions() {
    let left = [1.0, 2.0, 3.0];
    let right = [4.0, 5.0, 6.0];

    let reduced = CpuCollectives::all_reduce_sum(&[&left, &right]).unwrap();
    let gathered = CpuCollectives::gather(&[&left, &right]);

    assert_eq!(reduced, vec![5.0, 7.0, 9.0]);
    assert_eq!(gathered, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn cpu_collectives_reject_mismatched_reduce_shapes() {
    let left = [1.0, 2.0];
    let right = [3.0];

    let err = CpuCollectives::all_reduce_sum(&[&left, &right]).unwrap_err();

    assert!(matches!(err, MoeRuntimeError::Shape(message) if message.contains("expected 2")));
}

#[test]
fn checked_layer_run_reports_reference_match_and_speedup() {
    let fixture = build_fixture();
    let runtime = MoeRuntime::new(fixture.open(), MoeRuntimeOptions::default()).unwrap();
    let input = vec![2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0];

    let checked = runtime.run_layer_top1_checked(0, &input, 2, 0.0).unwrap();

    assert!(checked.comparison.passed);
    assert_eq!(checked.comparison.max_abs_diff, 0.0);
    assert_eq!(checked.routed.output, checked.reference_output);
    assert!(checked.comparison.speedup.is_finite() || checked.comparison.speedup.is_infinite());
}

#[test]
fn prepared_layer_rejects_device_capacity_overflow() {
    let mut placement = fixture_placement();
    placement.devices[0].capacity_bytes = 16;
    let fixture = build_fixture_with(Vec::new(), fixture_tensors(), placement);
    let runtime = MoeRuntime::new(fixture.open(), MoeRuntimeOptions::default()).unwrap();

    let err = runtime.prepare_layer(0, 2).unwrap_err();

    assert!(
        matches!(err, MoeRuntimeError::Shape(message) if message.contains("exceeding manifest capacity 16"))
    );
}

#[test]
fn runtime_limits_reject_excessive_token_count() {
    let fixture = build_fixture();
    let runtime = MoeRuntime::new(
        fixture.open(),
        MoeRuntimeOptions {
            limits: RuntimeLimits {
                max_tokens: Some(3),
                ..RuntimeLimits::default()
            },
            ..MoeRuntimeOptions::default()
        },
    )
    .unwrap();

    let err = runtime
        .run_layer_top1(0, &[2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0], 2)
        .unwrap_err();

    assert!(
        matches!(err, MoeRuntimeError::Shape(message) if message.contains("token count 4 exceeds runtime limit 3"))
    );
}

#[test]
fn duplicate_expert_role_is_rejected() {
    let mut tensors = fixture_tensors();
    tensors.push(tensor(
        "layers.0.experts.0.up.duplicate",
        1,
        [2, 2],
        moe_meta(0, Some(0), "up"),
        &[1.0, 0.0, 0.0, 1.0],
        None,
    ));
    let fixture = build_fixture_with(Vec::new(), tensors, fixture_placement());
    let runtime = MoeRuntime::new(fixture.open(), MoeRuntimeOptions::default()).unwrap();

    let err = runtime.run_layer_top1(0, &[2.0, 1.0], 2).unwrap_err();

    assert!(
        matches!(err, MoeRuntimeError::Shape(message) if message.contains("multiple up tensors"))
    );
}

#[test]
fn non_top1_metadata_is_rejected() {
    let fixture = build_fixture_with(
        vec![("moe.top_k", "2")],
        fixture_tensors(),
        fixture_placement(),
    );
    let runtime = MoeRuntime::new(fixture.open(), MoeRuntimeOptions::default()).unwrap();

    let err = runtime.run_layer_top1(0, &[2.0, 1.0], 2).unwrap_err();

    assert!(
        matches!(err, MoeRuntimeError::Unsupported(message) if message.contains("moe.top_k=1"))
    );
}

#[test]
fn n_experts_must_match_router_rows() {
    let fixture = build_fixture_with(
        vec![("moe.n_experts", "3")],
        fixture_tensors(),
        fixture_placement(),
    );
    let runtime = MoeRuntime::new(fixture.open(), MoeRuntimeOptions::default()).unwrap();

    let err = runtime.run_layer_top1(0, &[2.0, 1.0], 2).unwrap_err();

    assert!(matches!(err, MoeRuntimeError::Shape(message) if message.contains("moe.n_experts=3")));
}

#[test]
fn silu_gate_must_share_expert_shard() {
    let mut tensors = fixture_tensors();
    tensors.push(tensor(
        "layers.0.experts.0.gate",
        3,
        [2, 2],
        moe_meta(0, Some(0), "gate"),
        &[1.0, 0.0, 0.0, 1.0],
        Some("layer0.expert0.gate"),
    ));
    tensors.push(tensor(
        "layers.0.experts.1.gate",
        2,
        [2, 2],
        moe_meta(0, Some(1), "gate"),
        &[1.0, 0.0, 0.0, 1.0],
        Some("layer0.expert1"),
    ));
    let fixture = build_fixture_with(Vec::new(), tensors, fixture_placement());
    let runtime = MoeRuntime::new(
        fixture.open(),
        MoeRuntimeOptions {
            activation: ExpertActivation::SiluGated,
            ..MoeRuntimeOptions::default()
        },
    )
    .unwrap();

    let err = runtime.run_layer_top1(0, &[2.0, 1.0], 2).unwrap_err();

    assert!(
        matches!(err, MoeRuntimeError::Shape(message) if message.contains("expert tensors span multiple shards"))
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_preference_matches_reference_or_falls_back() {
    let fixture = build_fixture();
    let file = fixture.open();
    let runtime = MoeRuntime::new(
        file,
        MoeRuntimeOptions {
            prefer_wgpu: true,
            ..MoeRuntimeOptions::default()
        },
    )
    .unwrap();
    let input = vec![2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0];
    let parallel = runtime.run_layer_top1(0, &input, 2).unwrap();
    let reference = runtime.run_layer_reference_top1(0, &input, 2).unwrap();

    assert_close(&parallel.output, &reference, 1e-5);
    match &parallel.report.backend {
        RuntimeBackend::WgpuCompute {
            requested_devices,
            available_adapters,
            adapter_name,
        } => {
            assert_eq!(*requested_devices, 2);
            assert!(*available_adapters >= 1);
            assert!(!adapter_name.is_empty());
        }
        RuntimeBackend::CpuFallback { reason } => {
            assert!(!reason.is_empty());
        }
    }
}

#[test]
fn missing_placement_manifest_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("no-placement.rsmf");
    RsmfWriter::new()
        .with_tensor(tensor(
            "layers.0.router.weight",
            0,
            [2, 2],
            moe_meta(0, None, "router"),
            &[1.0, 0.0, 0.0, 1.0],
            None,
        ))
        .write_to_path(&path)
        .unwrap();

    let err =
        MoeRuntime::new(RsmfFile::open(path).unwrap(), MoeRuntimeOptions::default()).unwrap_err();
    assert!(matches!(err, MoeRuntimeError::Missing(message) if message == "PlacementManifest"));
}

#[cfg(feature = "wgpu")]
fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "value {idx} differs: {left} vs {right}"
        );
    }
}
