//! CastF16 encode/decode tests.

use std::io::Write;

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput, convert_f32_to_f16_bytes};
use rsmf_core::{LogicalDtype, RsmfFile, TargetTag};
use tempfile::NamedTempFile;

fn make_f32_tensor(name: &str, data: &[f32]) -> TensorInput {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorInput {
        shard_id: 0,

        name: name.into(),
        dtype: LogicalDtype::F32,
        shape: vec![data.len() as u64],

        metadata: vec![],
        canonical: VariantInput::canonical_raw(bytes),
        packed: vec![],
    }
}

#[test]
fn auto_f16_roundtrip_within_precision() {
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let tensor = make_f32_tensor("weight", &data);

    let writer = RsmfWriter::new().with_tensor_auto_f16(tensor, TargetTag::CpuGeneric);
    let bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();

    // Canonical view: raw f32, exact.
    let view = file.tensor_view("weight").unwrap();
    let canonical = view.decode_f32().unwrap();
    for (i, (&orig, &read)) in data.iter().zip(canonical.iter()).enumerate() {
        assert!(
            (orig - read).abs() < f32::EPSILON,
            "canonical mismatch at {i}"
        );
    }

    // The CpuGeneric packed variant (index 1) should be CastF16.
    let manifest = file.manifest();
    let t = &manifest.tensors[0];
    assert_eq!(t.packed_variants.len(), 1);
    let packed_idx = t.packed_variants[0];
    let packed_v = &manifest.variants[packed_idx as usize];
    assert_eq!(packed_v.encoding, rsmf_core::EncodingKind::CastF16);
    assert_eq!(packed_v.target, TargetTag::CpuGeneric);

    // Access packed variant via tensor_view_variant.
    let packed_view = file.tensor_view_variant("weight", packed_idx).unwrap();
    assert_eq!(packed_view.encoding, rsmf_core::EncodingKind::CastF16);
    let decoded = packed_view.decode_f32().unwrap();
    assert_eq!(decoded.len(), data.len());
    // f16 precision: values up to 8.0 are exact in f16; above that, tolerance increases.
    for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
        let tol = if orig.abs() <= 2048.0 { 0.01 } else { 1.0 };
        assert!(
            (orig - dec).abs() < tol,
            "f16 decode mismatch at {i}: orig={orig} decoded={dec}"
        );
    }
}

#[test]
fn convert_f32_to_f16_bytes_length() {
    let f32_bytes: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let f16_bytes = convert_f32_to_f16_bytes(&f32_bytes);
    assert_eq!(f16_bytes.len(), 4 * 2);
}

#[test]
fn decode_f32_on_raw_f32_works() {
    let data = vec![1.0f32, 2.0, 3.0];
    let tensor = make_f32_tensor("t", &data);
    let writer = RsmfWriter::new().with_tensor(tensor);
    let bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let view = file.tensor_view("t").unwrap();
    let decoded = view.decode_f32().unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn full_verify_passes_with_f16_variants() {
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let tensor = make_f32_tensor("w", &data);
    let writer = RsmfWriter::new().with_tensor_auto_f16(tensor, TargetTag::Wgpu);
    let bytes = writer.write_to_bytes().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    file.full_verify().unwrap();
}
