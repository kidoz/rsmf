use std::collections::HashMap;

use rsmf_core::writer::{GraphInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile};
use rsmf_runtime::{Engine, InitializerBinding, RuntimeTensor, SessionOptions};
use tempfile::tempdir;

const FEATURES: [&str; 8] = [
    "crash", "error", "slow", "refund", "invoice", "price", "demo", "thanks",
];
const LABELS: [&str; 3] = ["support_bug", "billing", "sales"];

fn main() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("tiny-issue-classifier.rsmf");
    write_classifier_bundle(&path)?;

    let engine = Engine::new(RsmfFile::open(&path)?)?;
    let options = SessionOptions {
        initializers: vec![
            InitializerBinding::new("weights", "classifier.weights"),
            InitializerBinding::new("bias", "classifier.bias"),
        ],
        ..SessionOptions::default()
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    let samples = if args.is_empty() {
        vec![
            "the app has a crash and error on startup".to_string(),
            "please refund the invoice from last month".to_string(),
            "can I get a demo and pricing details".to_string(),
        ]
    } else {
        vec![args.join(" ")]
    };

    println!("wrote {}", path.display());
    println!("features: {FEATURES:?}");
    for sample in samples {
        let features = featurize(&sample);
        let logits = run_classifier(&engine, options.clone(), features)?;
        let probabilities = softmax(&logits);
        let best = argmax(&probabilities)?;

        println!();
        println!("text:  {sample}");
        println!(
            "label: {} ({:.1}%)",
            LABELS[best],
            probabilities[best] * 100.0
        );
        println!("scores:");
        for (label, probability) in LABELS.iter().zip(probabilities) {
            println!("  {label:<12} {:>5.1}%", probability * 100.0);
        }
    }

    Ok(())
}

fn write_classifier_bundle(path: &std::path::Path) -> anyhow::Result<()> {
    RsmfWriter::new()
        .with_metadata("model.task", "tiny_issue_intent_classifier")
        .with_tensor(TensorInput {
            name: "classifier.weights".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![FEATURES.len() as u64, LABELS.len() as u64],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&classifier_weights())),
            packed: Vec::new(),
        })
        .with_tensor(TensorInput {
            name: "classifier.bias".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![LABELS.len() as u64],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[-0.5, -0.5, -0.5])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_linear_classifier_onnx_model()))
        .write_to_path(path)?;
    Ok(())
}

fn run_classifier(
    engine: &Engine,
    options: SessionOptions,
    features: Vec<f32>,
) -> anyhow::Result<Vec<f32>> {
    let outputs = engine.run(
        0,
        options,
        HashMap::from([(
            "features".to_string(),
            RuntimeTensor::F32 {
                shape: vec![1, FEATURES.len()],
                data: features,
            },
        )]),
    )?;
    match outputs.get("logits") {
        Some(RuntimeTensor::F32 { data, .. }) => Ok(data.clone()),
        Some(other) => anyhow::bail!("expected F32 logits, got {other:?}"),
        None => anyhow::bail!("missing logits output"),
    }
}

fn featurize(text: &str) -> Vec<f32> {
    let mut values = vec![0.0; FEATURES.len()];
    for token in text
        .to_ascii_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
    {
        if token.is_empty() {
            continue;
        }
        let normalized = token.strip_suffix('s').unwrap_or(token);
        if let Some(index) = FEATURES
            .iter()
            .position(|feature| *feature == token || *feature == normalized)
        {
            values[index] += 1.0;
        }
    }
    values
}

fn classifier_weights() -> Vec<f32> {
    vec![
        3.0, 0.0, 0.0, // crash
        2.5, 0.0, 0.0, // error
        2.0, 0.0, 0.0, // slow
        0.0, 3.0, 0.0, // refund
        0.0, 2.5, 0.0, // invoice
        0.0, 0.0, 2.5, // price
        0.0, 0.0, 3.0, // demo
        -0.5, 0.5, 0.5, // thanks
    ]
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |left, right| left.max(right));
    let mut exp_values: Vec<f32> = logits.iter().map(|value| (*value - max).exp()).collect();
    let total: f32 = exp_values.iter().sum();
    for value in &mut exp_values {
        *value /= total;
    }
    exp_values
}

fn argmax(values: &[f32]) -> anyhow::Result<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow::anyhow!("cannot select from empty scores"))
}

fn tiny_linear_classifier_onnx_model() -> Vec<u8> {
    let mut model = Vec::new();
    push_i64(&mut model, 1, 7);
    push_string(&mut model, 2, "rsmf-tiny-issue-classifier");
    push_message(&mut model, 7, tiny_linear_classifier_graph());
    push_message(&mut model, 8, opset_import("", 13));
    model
}

fn tiny_linear_classifier_graph() -> Vec<u8> {
    let mut graph = Vec::new();
    push_message(&mut graph, 1, matmul_node());
    push_message(&mut graph, 1, add_bias_node());
    push_string(&mut graph, 2, "rsmf_tiny_issue_classifier");
    push_message(
        &mut graph,
        5,
        external_tensor("weights", &[FEATURES.len() as i64, LABELS.len() as i64]),
    );
    push_message(
        &mut graph,
        5,
        external_tensor("bias", &[LABELS.len() as i64]),
    );
    push_message(
        &mut graph,
        11,
        value_info("features", &[1, FEATURES.len() as i64]),
    );
    push_message(
        &mut graph,
        12,
        value_info("logits", &[1, LABELS.len() as i64]),
    );
    graph
}

fn matmul_node() -> Vec<u8> {
    let mut node = Vec::new();
    push_string(&mut node, 1, "features");
    push_string(&mut node, 1, "weights");
    push_string(&mut node, 2, "scores");
    push_string(&mut node, 3, "linear_scores");
    push_string(&mut node, 4, "MatMul");
    node
}

fn add_bias_node() -> Vec<u8> {
    let mut node = Vec::new();
    push_string(&mut node, 1, "scores");
    push_string(&mut node, 1, "bias");
    push_string(&mut node, 2, "logits");
    push_string(&mut node, 3, "add_bias");
    push_string(&mut node, 4, "Add");
    node
}

fn external_tensor(name: &str, shape: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    for &dim in shape {
        push_i64(&mut tensor, 1, dim);
    }
    push_i32(&mut tensor, 2, 1);
    push_string(&mut tensor, 8, name);
    push_message(&mut tensor, 13, string_string_entry("location", "rsmf"));
    push_i32(&mut tensor, 14, 1);
    tensor
}

fn value_info(name: &str, shape: &[i64]) -> Vec<u8> {
    let mut value = Vec::new();
    push_string(&mut value, 1, name);
    push_message(&mut value, 2, type_proto(shape));
    value
}

fn type_proto(shape: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    push_i32(&mut tensor, 1, 1);
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
