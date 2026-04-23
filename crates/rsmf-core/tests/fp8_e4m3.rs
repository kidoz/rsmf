//! Correctness tests for OCP FP8 E4M3.
//!
//! E4M3 is 1 sign + 4 exponent (bias 7) + 3 mantissa bits. It does **not**
//! follow IEEE-754: there is no infinity, and only a single NaN sentinel.
//! The OCP spec (`S.1111.111` → NaN) means encodings `0x7F` and `0xFF` must
//! decode to NaN. Every other E=15 encoding is a normal finite value up to
//! ±448.
//!
//! Assertions are computed by hand from the spec.

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile, StorageDtype, TargetTag};
use tempfile::tempdir;

fn decode_e4m3_bytes(bytes: &[u8]) -> Vec<f32> {
    let canonical_bytes: Vec<u8> = std::iter::repeat_n(0u8, bytes.len() * 4).collect();

    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "t".into(),
        dtype: LogicalDtype::F32,
        shape: vec![bytes.len() as u64],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(canonical_bytes),
        packed: vec![VariantInput::packed_fp8_e4m3(
            TargetTag::CpuGeneric,
            bytes.to_vec(),
        )],
    });
    let dir = tempdir().unwrap();
    let path = dir.path().join("e4m3.rsmf");
    writer.write_to_path(&path).unwrap();

    let file = RsmfFile::open(&path).unwrap();
    let tensor = file
        .manifest()
        .tensors
        .iter()
        .find(|t| t.name == "t")
        .unwrap();
    let packed_idx = tensor.packed_variants[0];
    let view = file.tensor_view_variant("t", packed_idx).unwrap();
    assert_eq!(view.storage_dtype, StorageDtype::Fp8E4M3);
    view.decode_f32().unwrap()
}

#[test]
fn decoder_handles_positive_and_negative_zero() {
    let out = decode_e4m3_bytes(&[0x00, 0x80]);
    assert_eq!(out[0], 0.0_f32);
    assert!(out[0].is_sign_positive());
    assert_eq!(out[1], 0.0_f32);
    assert!(out[1].is_sign_negative());
}

#[test]
fn decoder_handles_one_and_minus_one() {
    // 1.0 in E4M3: sign=0, exp=7 (bias-7), mant=0 → 0b0_0111_000 = 0x38.
    // −1.0: 0xB8.
    let out = decode_e4m3_bytes(&[0x38, 0xB8]);
    assert_eq!(out[0], 1.0_f32);
    assert_eq!(out[1], -1.0_f32);
}

#[test]
fn decoder_emits_nan_for_ocp_sentinel() {
    // OCP E4M3 NaN: S.1111.111 — encodings 0x7F (+NaN) and 0xFF (−NaN).
    // These are the only NaN patterns in the format; every other E=15
    // encoding is a finite normal value.
    let out = decode_e4m3_bytes(&[0x7F, 0xFF]);
    assert!(out[0].is_nan(), "0x7F must decode as NaN, got {}", out[0]);
    assert!(out[1].is_nan(), "0xFF must decode as NaN, got {}", out[1]);
}

#[test]
fn decoder_handles_max_finite() {
    // Max finite: S.1111.110 → exp=15, mant=6. Value = 2^(15-7) * (1 + 6/8)
    //                     = 256 * 1.75 = 448.
    // 0b0_1111_110 = 0x7E; −448 = 0xFE.
    let out = decode_e4m3_bytes(&[0x7E, 0xFE]);
    assert_eq!(out[0], 448.0_f32);
    assert_eq!(out[1], -448.0_f32);
}

#[test]
fn decoder_handles_near_max_exponent_non_nan_mantissas() {
    // Every E=15 encoding except mant=7 is a normal finite, **not** NaN.
    // Encoding 0x7D (exp=15, mant=5) = 2^8 * (1 + 5/8) = 256 * 1.625 = 416.
    let out = decode_e4m3_bytes(&[0x7D]);
    assert_eq!(out[0], 416.0_f32);
    assert!(!out[0].is_nan());
}

#[test]
fn decoder_handles_smallest_normal() {
    // Smallest positive normal: exp=1, mant=0 → 0b0_0001_000 = 0x08.
    // Value = 2^(1-7) * 1 = 2^-6 = 0.015625.
    let out = decode_e4m3_bytes(&[0x08]);
    assert_eq!(out[0], 2.0_f32.powi(-6));
}

#[test]
fn decoder_handles_smallest_subnormal() {
    // Smallest positive subnormal: exp=0, mant=1 → 0b0_0000_001 = 0x01.
    // Value = 2^-6 * (1/8) = 2^-9.
    let out = decode_e4m3_bytes(&[0x01]);
    assert_eq!(out[0], 2.0_f32.powi(-9));
}
