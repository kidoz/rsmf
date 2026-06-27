mod executor;
mod native_decoder;
mod network;
mod onnx;

use super::*;

fn f32_output(response: &RuntimeResponse, name: &str) -> Vec<f32> {
    match response.outputs.get(name).unwrap() {
        RuntimeTensor::F32 { data, .. } => data.clone(),
        other => panic!("expected F32 output, got {other:?}"),
    }
}

fn assert_close_slice(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "index {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn f32_output_shape(response: &RuntimeResponse, name: &str) -> Vec<usize> {
    match response.outputs.get(name).unwrap() {
        RuntimeTensor::F32 { shape, .. } => shape.clone(),
        other => panic!("expected F32 output, got {other:?}"),
    }
}
