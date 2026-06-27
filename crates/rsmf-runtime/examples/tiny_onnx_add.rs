use std::collections::HashMap;

use rsmf_core::writer::{GraphInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile};
use rsmf_runtime::{Engine, InitializerBinding, RuntimeTensor, SessionOptions};
use tempfile::tempdir;

fn main() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("tiny-add.rsmf");

    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_add_external_initializer_onnx_model()))
        .write_to_path(&path)?;

    let engine = Engine::new(RsmfFile::open(&path)?)?;
    let options = SessionOptions {
        initializers: vec![InitializerBinding::new("bias", "bias")],
        ..SessionOptions::default()
    };
    let outputs = engine.run(
        0,
        options,
        HashMap::from([(
            "x".to_string(),
            RuntimeTensor::F32 {
                shape: vec![2],
                data: vec![1.0, 2.0],
            },
        )]),
    )?;

    let RuntimeTensor::F32 { shape, data } = outputs.get("z").unwrap() else {
        anyhow::bail!("expected F32 output named z");
    };

    println!("wrote {}", path.display());
    println!("z shape: {shape:?}");
    println!("z data:  {data:?}");
    println!("expected: [11.0, -2.0]");

    Ok(())
}

fn tiny_add_external_initializer_onnx_model() -> Vec<u8> {
    let mut model = Vec::new();
    push_i64(&mut model, 1, 7);
    push_string(&mut model, 2, "rsmf-tiny-add-example");
    push_message(&mut model, 7, tiny_add_graph_with_external_initializer());
    push_message(&mut model, 8, opset_import("", 13));
    model
}

fn tiny_add_graph_with_external_initializer() -> Vec<u8> {
    let mut graph = Vec::new();
    push_message(&mut graph, 1, add_initializer_node());
    push_string(&mut graph, 2, "rsmf_add_initializer_graph");
    push_message(&mut graph, 5, external_tensor("bias", 1, &[2]));
    push_message(&mut graph, 11, value_info("x", &[2]));
    push_message(&mut graph, 12, value_info("z", &[2]));
    graph
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

fn value_info(name: &str, shape: &[i64]) -> Vec<u8> {
    let mut value = Vec::new();
    push_string(&mut value, 1, name);
    push_message(&mut value, 2, type_proto(1, shape));
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

fn tensor_shape(shape: &[i64]) -> Vec<u8> {
    let mut tensor_shape = Vec::new();
    for &dim in shape {
        let mut dimension = Vec::new();
        push_i64(&mut dimension, 1, dim);
        push_message(&mut tensor_shape, 1, dimension);
    }
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
