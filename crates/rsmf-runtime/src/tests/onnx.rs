use super::super::*;

use std::collections::HashMap;

use ndarray::ArrayD;
use rsmf_core::RsmfFile;
use rsmf_core::tensor::variant::EncodingKind;
use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{GraphInput, LayoutTag, LogicalDtype, StorageDtype, TargetTag, VariantMeta};
use tempfile::tempdir;

#[test]
fn session_options_participate_in_cache_key() {
    let default_key = SessionKey::new(0, SessionOptions::default());
    let tuned_key = SessionKey::new(
        0,
        SessionOptions {
            intra_threads: Some(1),
            ..SessionOptions::default()
        },
    );
    let io_binding_key = SessionKey::new(
        0,
        SessionOptions {
            io_binding: IoBindingPolicy::Cpu,
            ..SessionOptions::default()
        },
    );
    assert_ne!(default_key, tuned_key);
    assert_ne!(default_key, io_binding_key);
}

#[test]
fn tensor_value_shape_mismatch_is_rejected() {
    let err = RuntimeTensor::F32 {
        shape: vec![2, 2],
        data: vec![1.0, 2.0, 3.0],
    }
    .into_ort_value()
    .unwrap_err();
    assert!(matches!(err, RuntimeError::Shape(message) if message.contains("implies 4 elements")));
}

#[test]
fn missing_graph_index_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("no-graph.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "x".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(1.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .write_to_path(&path)
        .unwrap();
    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();

    let err = engine
        .session_handle(0, SessionOptions::default())
        .unwrap_err();

    assert!(matches!(
        err,
        RuntimeError::GraphNotFound {
            graph_idx: 0,
            graph_count: 0
        }
    ));
}

#[test]
fn runs_embedded_onnx_add_graph_from_rsmf() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add.onnx.rsmf");
    let graph = tiny_add_onnx_model();
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(graph.clone()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let handle = engine.session_handle(0, SessionOptions::default()).unwrap();
    assert_eq!(handle.memory_report().graph_payload_bytes, graph.len());
    assert_eq!(handle.memory_report().initializer_count(), 0);
    assert_eq!(handle.memory_report().initializer_materialized_bytes, 0);
    assert_eq!(handle.memory_report().initializer_source_bytes, 0);
    assert_eq!(handle.memory_report().initializer_zero_copy_bytes, 0);
    assert_eq!(handle.memory_report().io_binding, IoBindingPolicy::Disabled);
    assert!(matches!(
        &handle.memory_report().provider_allocator_bytes,
        RuntimeMemoryMeasurement::Unavailable { .. }
    ));
    assert!(matches!(
        &handle.memory_report().provider_allocator_stats,
        RuntimeAllocatorStats::Available { .. } | RuntimeAllocatorStats::Unavailable { .. }
    ));
    assert!(matches!(
        handle.capability_report().ort_cpu_io_binding,
        RuntimeCapability::Available
    ));
    assert!(matches!(
        handle.capability_report().ort_provider_allocator_stats,
        RuntimeCapability::Available
    ));
    assert!(matches!(
        handle.capability_report().mmap_initializer_zero_copy,
        RuntimeCapability::Unavailable { .. }
    ));
    assert!(matches!(
        handle
            .capability_report()
            .native_decoder_i8_q8_q4_direct_kernels,
        RuntimeCapability::Available
    ));
    assert!(matches!(
        handle
            .capability_report()
            .native_decoder_qk_family_direct_kernels,
        RuntimeCapability::Available
    ));
    assert!(matches!(
        handle
            .capability_report()
            .native_decoder_fused_qkv_attention_mlp,
        RuntimeCapability::Available
    ));
    assert!(matches!(
        handle.capability_report().serving_bearer_auth,
        RuntimeCapability::Available
    ));
    assert!(matches!(
        handle.capability_report().serving_load_shedding,
        RuntimeCapability::Available
    ));
    assert!(matches!(
        handle.capability_report().serving_tls,
        RuntimeCapability::Unavailable { .. }
    ));
    let input_names = handle
        .inputs()
        .iter()
        .map(|input| input.name.as_str())
        .collect::<Vec<_>>();
    let output_names = handle
        .outputs()
        .iter()
        .map(|output| output.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(input_names, vec!["x", "y"]);
    assert_eq!(output_names, vec!["z"]);

    let outputs = engine
        .run_f32(
            0,
            HashMap::from([
                (
                    "x".to_string(),
                    ArrayD::from_shape_vec(vec![2], vec![1.5, -2.0]).unwrap(),
                ),
                (
                    "y".to_string(),
                    ArrayD::from_shape_vec(vec![2], vec![2.5, 3.0]).unwrap(),
                ),
            ]),
        )
        .unwrap();

    let z = outputs.get("z").unwrap();
    assert_eq!(z.shape(), &[2]);
    assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![4.0, 1.0]);
}

#[test]
fn runtime_allocator_stats_parse_common_ort_counters() {
    let stats = RuntimeAllocatorStats::from_entries(vec![
        RuntimeAllocatorStat {
            key: "TotalAllocated".to_string(),
            value: "4096".to_string(),
        },
        RuntimeAllocatorStat {
            key: "MaxInUse".to_string(),
            value: "2048".to_string(),
        },
        RuntimeAllocatorStat {
            key: "MaxAllocSize".to_string(),
            value: "1024".to_string(),
        },
        RuntimeAllocatorStat {
            key: "NumAllocs".to_string(),
            value: "7".to_string(),
        },
        RuntimeAllocatorStat {
            key: "NumReserves".to_string(),
            value: "2".to_string(),
        },
    ]);

    assert_eq!(stats.max_in_use_bytes(), Some(2048));
    assert!(matches!(
        stats,
        RuntimeAllocatorStats::Available {
            total_allocated_bytes: Some(4096),
            max_in_use_bytes: Some(2048),
            max_alloc_size_bytes: Some(1024),
            allocation_count: Some(7),
            reserve_count: Some(2),
            ..
        }
    ));
}

#[test]
fn runs_embedded_onnx_add_graph_with_cpu_io_binding_policy() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add-iobinding.onnx.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_add_onnx_model()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let options = SessionOptions {
        io_binding: IoBindingPolicy::Cpu,
        ..SessionOptions::default()
    };
    let handle = engine.session_handle(0, options.clone()).unwrap();
    assert_eq!(handle.memory_report().io_binding, IoBindingPolicy::Cpu);
    let outputs = engine
        .run_f32_with_options(
            0,
            options,
            HashMap::from([
                (
                    "x".to_string(),
                    ArrayD::from_shape_vec(vec![2], vec![1.5, -2.0]).unwrap(),
                ),
                (
                    "y".to_string(),
                    ArrayD::from_shape_vec(vec![2], vec![2.5, 3.0]).unwrap(),
                ),
            ]),
        )
        .unwrap();

    let z = outputs.get("z").unwrap();
    assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![4.0, 1.0]);
}

#[test]
fn runs_onnx_graph_with_rsmf_external_initializer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add-initializer.onnx.rsmf");
    let graph = tiny_add_external_initializer_onnx_model();
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(graph.clone()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let options = SessionOptions {
        initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
        ..SessionOptions::default()
    };
    let handle = engine.session_handle(0, options.clone()).unwrap();
    let memory_report = handle.memory_report().clone();
    assert_eq!(memory_report.graph_payload_bytes, graph.len());
    assert_eq!(memory_report.initializer_count(), 1);
    assert_eq!(memory_report.initializer_materialized_bytes, 8);
    assert_eq!(memory_report.initializer_source_bytes, 8);
    assert_eq!(memory_report.initializer_zero_copy_bytes, 0);
    assert_eq!(
        memory_report.initializers,
        vec![InitializerMemoryReport {
            initializer_name: "bias".to_string(),
            tensor_name: "bias.tensor".to_string(),
            variant_idx: None,
            materialized_bytes: 8,
            source_bytes: 8,
            zero_copy_bytes: 0,
        }]
    );
    let cached_handle = engine.session_handle(0, options.clone()).unwrap();
    assert_eq!(cached_handle.memory_report(), &memory_report);
    let input_names = handle
        .inputs()
        .iter()
        .map(|input| input.name.as_str())
        .collect::<Vec<_>>();
    let output_names = handle
        .outputs()
        .iter()
        .map(|output| output.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(input_names, vec!["x"]);
    assert_eq!(output_names, vec!["z"]);

    let outputs = engine
        .run_f32_with_options(
            0,
            options,
            HashMap::from([(
                "x".to_string(),
                ArrayD::from_shape_vec(vec![2], vec![1.5, 2.0]).unwrap(),
            )]),
        )
        .unwrap();

    let z = outputs.get("z").unwrap();
    assert_eq!(z.shape(), &[2]);
    assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![11.5, -2.0]);
}

#[test]
fn runs_onnx_graph_with_selected_raw_rsmf_initializer_variant() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add-selected-variant.onnx.rsmf");
    let graph = tiny_add_external_initializer_onnx_model();
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[0.0, 0.0])),
            packed: vec![raw_variant(
                StorageDtype::Logical(LogicalDtype::F32),
                LayoutTag::RowMajor,
                f32_bytes(&[10.0, -4.0]),
            )],
        })
        .with_graph(GraphInput::onnx(graph.clone()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let options = SessionOptions {
        initializers: vec![InitializerBinding::new("bias", "bias.tensor").with_variant(1)],
        ..SessionOptions::default()
    };
    let handle = engine.session_handle(0, options.clone()).unwrap();
    assert_eq!(handle.memory_report().initializer_materialized_bytes, 8);
    assert_eq!(handle.memory_report().initializer_source_bytes, 8);
    assert_eq!(handle.memory_report().initializer_zero_copy_bytes, 0);
    assert_eq!(
        handle.memory_report().initializers,
        vec![InitializerMemoryReport {
            initializer_name: "bias".to_string(),
            tensor_name: "bias.tensor".to_string(),
            variant_idx: Some(1),
            materialized_bytes: 8,
            source_bytes: 8,
            zero_copy_bytes: 0,
        }]
    );

    let outputs = engine
        .run_f32_with_options(
            0,
            options,
            HashMap::from([(
                "x".to_string(),
                ArrayD::from_shape_vec(vec![2], vec![1.5, 2.0]).unwrap(),
            )]),
        )
        .unwrap();

    let z = outputs.get("z").unwrap();
    assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![11.5, -2.0]);
}

#[test]
fn blocked_initializer_variant_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("blocked-initializer-variant.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[0.0, 0.0])),
            packed: vec![raw_variant(
                StorageDtype::Logical(LogicalDtype::F32),
                LayoutTag::Blocked,
                f32_bytes(&[10.0, -4.0]),
            )],
        })
        .with_graph(GraphInput::onnx(tiny_add_external_initializer_onnx_model()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let err = engine
        .session_handle(
            0,
            SessionOptions {
                initializers: vec![InitializerBinding::new("bias", "bias.tensor").with_variant(1)],
                ..SessionOptions::default()
            },
        )
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("row-major"))
    );
}

#[test]
fn runs_onnx_graph_with_i64_rsmf_external_initializer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add-i64-initializer.onnx.rsmf");
    let graph = tiny_add_external_initializer_onnx_model_with_dtype_shape(7, &[2]);
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::I64,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(i64_bytes(&[10, -4])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(graph.clone()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let options = SessionOptions {
        initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
        ..SessionOptions::default()
    };
    let handle = engine.session_handle(0, options.clone()).unwrap();
    assert_eq!(handle.memory_report().graph_payload_bytes, graph.len());
    assert_eq!(handle.memory_report().initializer_count(), 1);
    assert_eq!(handle.memory_report().initializer_materialized_bytes, 16);
    assert_eq!(handle.memory_report().initializer_source_bytes, 16);
    assert_eq!(handle.memory_report().initializer_zero_copy_bytes, 0);

    let outputs = engine
        .run(
            0,
            options,
            HashMap::from([(
                "x".to_string(),
                RuntimeTensor::I64 {
                    shape: vec![2],
                    data: vec![2, 9],
                },
            )]),
        )
        .unwrap();

    assert_eq!(
        outputs.get("z"),
        Some(&RuntimeTensor::I64 {
            shape: vec![2],
            data: vec![12, 5],
        })
    );
}

#[test]
fn missing_initializer_tensor_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing-initializer.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_add_external_initializer_onnx_model()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let err = engine
        .session_handle(
            0,
            SessionOptions {
                initializers: vec![InitializerBinding::new("bias", "missing.tensor")],
                ..SessionOptions::default()
            },
        )
        .unwrap_err();

    assert!(matches!(
        err,
        RuntimeError::InitializerTensorNotFound {
            initializer_name,
            tensor_name
        } if initializer_name == "bias" && tensor_name == "missing.tensor"
    ));
}

#[test]
fn initializer_shape_mismatch_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shape-mismatch-initializer.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(
            tiny_add_external_initializer_onnx_model_with_dtype_shape(1, &[3]),
        ))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let err = engine
        .session_handle(
            0,
            SessionOptions {
                initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
                ..SessionOptions::default()
            },
        )
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("shape"))
    );
}

#[test]
fn initializer_dtype_mismatch_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dtype-mismatch-initializer.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(
            tiny_add_external_initializer_onnx_model_with_dtype_shape(7, &[2]),
        ))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let err = engine
        .session_handle(
            0,
            SessionOptions {
                initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
                ..SessionOptions::default()
            },
        )
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("dtype"))
    );
}

fn tiny_add_onnx_model() -> Vec<u8> {
    let mut model = Vec::new();
    push_i64(&mut model, 1, 7);
    push_string(&mut model, 2, "rsmf-runtime-test");
    push_message(&mut model, 7, tiny_add_graph());
    push_message(&mut model, 8, opset_import("", 13));
    model
}

fn tiny_add_external_initializer_onnx_model() -> Vec<u8> {
    tiny_add_external_initializer_onnx_model_with_dtype_shape(1, &[2])
}

fn tiny_add_external_initializer_onnx_model_with_dtype_shape(
    data_type: i32,
    shape: &[i64],
) -> Vec<u8> {
    let mut model = Vec::new();
    push_i64(&mut model, 1, 7);
    push_string(&mut model, 2, "rsmf-runtime-test");
    push_message(
        &mut model,
        7,
        tiny_add_graph_with_external_initializer(data_type, shape),
    );
    push_message(&mut model, 8, opset_import("", 13));
    model
}

fn tiny_add_graph() -> Vec<u8> {
    let mut graph = Vec::new();
    push_message(&mut graph, 1, add_node());
    push_string(&mut graph, 2, "rsmf_add_graph");
    push_message(&mut graph, 11, value_info("x", &[2]));
    push_message(&mut graph, 11, value_info("y", &[2]));
    push_message(&mut graph, 12, value_info("z", &[2]));
    graph
}

fn tiny_dynamic_add_onnx_model() -> Vec<u8> {
    let mut model = Vec::new();
    push_i64(&mut model, 1, 7);
    push_string(&mut model, 2, "rsmf-runtime-test");
    push_message(&mut model, 7, tiny_dynamic_add_graph());
    push_message(&mut model, 8, opset_import("", 13));
    model
}

fn tiny_dynamic_add_graph() -> Vec<u8> {
    let mut graph = Vec::new();
    push_message(&mut graph, 1, add_node());
    push_string(&mut graph, 2, "rsmf_dynamic_add_graph");
    push_message(&mut graph, 11, dynamic_value_info("x"));
    push_message(&mut graph, 11, dynamic_value_info("y"));
    push_message(&mut graph, 12, dynamic_value_info("z"));
    graph
}

fn tiny_add_graph_with_external_initializer(data_type: i32, shape: &[i64]) -> Vec<u8> {
    let mut graph = Vec::new();
    push_message(&mut graph, 1, add_initializer_node());
    push_string(&mut graph, 2, "rsmf_add_initializer_graph");
    push_message(&mut graph, 5, external_tensor("bias", data_type, shape));
    push_message(&mut graph, 11, value_info_typed("x", &[2], data_type));
    push_message(&mut graph, 12, value_info_typed("z", &[2], data_type));
    graph
}

fn add_node() -> Vec<u8> {
    let mut node = Vec::new();
    push_string(&mut node, 1, "x");
    push_string(&mut node, 1, "y");
    push_string(&mut node, 2, "z");
    push_string(&mut node, 3, "add");
    push_string(&mut node, 4, "Add");
    node
}

fn add_initializer_node() -> Vec<u8> {
    let mut node = Vec::new();
    push_string(&mut node, 1, "x");
    push_string(&mut node, 1, "bias");
    push_string(&mut node, 2, "z");
    push_string(&mut node, 3, "add_initializer");
    push_string(&mut node, 4, "Add");
    node
}

fn external_tensor(name: &str, data_type: i32, shape: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    for &dim in shape {
        push_i64(&mut tensor, 1, dim);
    }
    push_i32(&mut tensor, 2, data_type);
    push_string(&mut tensor, 8, name);
    push_message(&mut tensor, 13, string_string_entry("location", "rsmf"));
    push_i32(&mut tensor, 14, 1);
    tensor
}

fn raw_variant(storage_dtype: StorageDtype, layout: LayoutTag, bytes: Vec<u8>) -> VariantInput {
    VariantInput {
        target: TargetTag::CpuGeneric,
        encoding: EncodingKind::Raw,
        storage_dtype: Some(storage_dtype),
        layout,
        alignment: 64,
        bytes,
        meta: VariantMeta::default(),
    }
}

fn string_string_entry(key: &str, value: &str) -> Vec<u8> {
    let mut entry = Vec::new();
    push_string(&mut entry, 1, key);
    push_string(&mut entry, 2, value);
    entry
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub(super) fn add_graph_engine(path: std::path::PathBuf) -> Engine {
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_add_onnx_model()))
        .write_to_path(&path)
        .unwrap();
    Engine::new(RsmfFile::open(path).unwrap()).unwrap()
}

pub(super) fn dynamic_add_graph_engine(path: std::path::PathBuf) -> Engine {
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_dynamic_add_onnx_model()))
        .write_to_path(&path)
        .unwrap();
    Engine::new(RsmfFile::open(path).unwrap()).unwrap()
}
fn value_info(name: &str, shape: &[i64]) -> Vec<u8> {
    value_info_typed(name, shape, 1)
}

fn value_info_typed(name: &str, shape: &[i64], data_type: i32) -> Vec<u8> {
    let mut value = Vec::new();
    push_string(&mut value, 1, name);
    push_message(&mut value, 2, type_proto(data_type, shape));
    value
}

fn dynamic_value_info(name: &str) -> Vec<u8> {
    let mut value = Vec::new();
    push_string(&mut value, 1, name);
    push_message(&mut value, 2, dynamic_type_proto());
    value
}

fn type_proto(data_type: i32, shape: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    push_i32(&mut tensor, 1, data_type);
    push_message(&mut tensor, 2, tensor_shape(shape));

    let mut type_proto = Vec::new();
    push_message(&mut type_proto, 1, tensor);
    type_proto
}

fn dynamic_type_proto() -> Vec<u8> {
    let mut tensor = Vec::new();
    push_i32(&mut tensor, 1, 1);
    push_message(&mut tensor, 2, dynamic_tensor_shape());

    let mut type_proto = Vec::new();
    push_message(&mut type_proto, 1, tensor);
    type_proto
}

fn tensor_shape(shape: &[i64]) -> Vec<u8> {
    let mut tensor_shape = Vec::new();
    for &dim in shape {
        let mut dimension = Vec::new();
        push_i64(&mut dimension, 1, dim);
        push_message(&mut tensor_shape, 1, dimension);
    }
    tensor_shape
}

fn dynamic_tensor_shape() -> Vec<u8> {
    let mut tensor_shape = Vec::new();
    let mut batch = Vec::new();
    push_string(&mut batch, 2, "batch");
    push_message(&mut tensor_shape, 1, batch);

    let mut width = Vec::new();
    push_i64(&mut width, 1, 2);
    push_message(&mut tensor_shape, 1, width);
    tensor_shape
}

fn opset_import(domain: &str, version: i64) -> Vec<u8> {
    let mut opset = Vec::new();
    if !domain.is_empty() {
        push_string(&mut opset, 1, domain);
    }
    push_i64(&mut opset, 2, version);
    opset
}

fn push_i32(out: &mut Vec<u8>, field: u32, value: i32) {
    push_varint_field(out, field, value as u64);
}

fn push_i64(out: &mut Vec<u8>, field: u32, value: i64) {
    push_varint_field(out, field, value as u64);
}

fn push_string(out: &mut Vec<u8>, field: u32, value: &str) {
    push_bytes(out, field, value.as_bytes());
}

fn push_message(out: &mut Vec<u8>, field: u32, message: Vec<u8>) {
    push_bytes(out, field, &message);
}

fn push_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    push_tag(out, field, 2);
    push_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn push_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
    push_tag(out, field, 0);
    push_varint(out, value);
}

fn push_tag(out: &mut Vec<u8>, field: u32, wire_type: u64) {
    push_varint(out, ((field as u64) << 3) | wire_type);
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}
