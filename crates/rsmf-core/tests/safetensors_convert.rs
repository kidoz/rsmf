//! Test requirement #9: safetensors → RSMF conversion preserves names,
//! shapes, dtypes, and bytes.

use std::collections::HashMap;
use std::io::Write;

use rsmf_core::safetensors_convert::writer_from_safetensors_bytes;
use rsmf_core::{LogicalDtype, RsmfFile};
use safetensors::tensor::{Dtype, TensorView};
use tempfile::NamedTempFile;

fn build_safetensors_bytes() -> Vec<u8> {
    let w: Vec<u8> = (0..9u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let b: Vec<u8> = (0..3u32)
        .flat_map(|i| ((i + 10) as f32).to_le_bytes())
        .collect();

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert(
        "w".into(),
        TensorView::new(Dtype::F32, vec![3, 3], &w).unwrap(),
    );
    tensors.insert(
        "b".into(),
        TensorView::new(Dtype::F32, vec![3], &b).unwrap(),
    );
    safetensors::serialize(&tensors, &None).unwrap()
}

fn build_safetensors_with_metadata() -> Vec<u8> {
    let w: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert(
        "x".into(),
        TensorView::new(Dtype::F32, vec![4], &w).unwrap(),
    );
    let mut meta = HashMap::new();
    meta.insert("model_name".into(), "test-model".into());
    meta.insert("version".into(), "42".into());
    safetensors::serialize(&tensors, &Some(meta)).unwrap()
}

#[test]
fn convert_preserves_names_shapes_dtypes_bytes() {
    let st = build_safetensors_bytes();
    let writer = writer_from_safetensors_bytes(&st).unwrap();
    let bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let inspect = file.inspect();
    assert_eq!(inspect.tensor_count, 2);
    // Global metadata includes our `source=safetensors` marker.
    assert!(
        inspect
            .metadata
            .iter()
            .any(|(k, v)| k == "source" && v == "safetensors")
    );

    let w = file.tensor_view("w").unwrap();
    assert_eq!(w.dtype(), LogicalDtype::F32);
    assert_eq!(w.shape(), &[3u64, 3]);
    let w_vec = w.to_vec::<f32>().unwrap();
    for (i, v) in w_vec.iter().enumerate().take(9) {
        assert!((v - i as f32).abs() < f32::EPSILON);
    }

    let b = file.tensor_view("b").unwrap();
    assert_eq!(b.shape(), &[3u64]);
    let b_vec = b.to_vec::<f32>().unwrap();
    assert!((b_vec[0] - 10.0).abs() < f32::EPSILON);
    assert!((b_vec[2] - 12.0).abs() < f32::EPSILON);
}

#[test]
fn convert_imports_safetensors_metadata() {
    let st = build_safetensors_with_metadata();
    let writer = writer_from_safetensors_bytes(&st).unwrap();
    let bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let meta = &file.inspect().metadata;

    // Standard marker is always present.
    assert!(
        meta.iter()
            .any(|(k, v)| k == "source" && v == "safetensors")
    );
    // Safetensors metadata imported under safetensors.* namespace.
    assert!(
        meta.iter()
            .any(|(k, v)| k == "safetensors.model_name" && v == "test-model"),
        "metadata: {meta:?}"
    );
    assert!(
        meta.iter()
            .any(|(k, v)| k == "safetensors.version" && v == "42"),
        "metadata: {meta:?}"
    );
}
