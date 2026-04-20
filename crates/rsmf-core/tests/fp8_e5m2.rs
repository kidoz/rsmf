//! Correctness tests for OCP FP8 E5M2.
//!
//! E5M2 is 1 sign + 5 exponent (bias 15) + 2 mantissa bits and behaves
//! like IEEE-754: normals, subnormals, ±0, ±∞, NaN. Because every bit
//! pattern maps to a known f32 value under the spec, every assertion
//! below is computed by hand from the spec rather than by cross-running
//! against a reference library.

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput, convert_f32_to_fp8_e5m2_bytes};
use rsmf_core::{LogicalDtype, RsmfFile, StorageDtype, TargetTag};
use tempfile::tempdir;

/// Pack a one-tensor file with a single E5M2 packed variant holding
/// `bytes`, then read the dequantised f32 out of the packed variant.
fn decode_e5m2_bytes(bytes: &[u8]) -> Vec<f32> {
    // The canonical tensor has to carry a matching f32 payload so the
    // reader sees a consistent shape. The packed variant is what we
    // actually want to test.
    let canonical_bytes: Vec<u8> = std::iter::repeat_n(0u8, bytes.len() * 4).collect();

    let writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 0,
        name: "t".into(),
        dtype: LogicalDtype::F32,
        shape: vec![bytes.len() as u64],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(canonical_bytes),
        packed: vec![VariantInput::packed_fp8_e5m2(
            TargetTag::CpuGeneric,
            bytes.to_vec(),
        )],
    });
    let dir = tempdir().unwrap();
    let path = dir.path().join("e5m2.rsmf");
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
    assert_eq!(view.storage_dtype, StorageDtype::Fp8E5M2);
    view.decode_f32().unwrap()
}

#[test]
fn decoder_handles_positive_and_negative_zero() {
    let out = decode_e5m2_bytes(&[0x00, 0x80]);
    assert_eq!(out[0], 0.0_f32);
    assert!(out[0].is_sign_positive());
    // −0 decodes as −0, preserving the sign bit.
    assert_eq!(out[1], 0.0_f32);
    assert!(out[1].is_sign_negative());
}

#[test]
fn decoder_handles_one_and_minus_one() {
    // 1.0 in E5M2: sign=0, exp=15 (bias-15), mant=0 → 0b0_01111_00 = 0x3C.
    // −1.0: 0xBC.
    let out = decode_e5m2_bytes(&[0x3C, 0xBC]);
    assert_eq!(out[0], 1.0_f32);
    assert_eq!(out[1], -1.0_f32);
}

#[test]
fn decoder_handles_infinities() {
    // ±∞: exp=31, mant=0.
    let out = decode_e5m2_bytes(&[0x7C, 0xFC]);
    assert!(out[0].is_infinite() && out[0].is_sign_positive());
    assert!(out[1].is_infinite() && out[1].is_sign_negative());
}

#[test]
fn decoder_handles_nan() {
    // NaN: exp=31, mant!=0.
    let out = decode_e5m2_bytes(&[0x7D, 0x7F]);
    assert!(out[0].is_nan());
    assert!(out[1].is_nan());
}

#[test]
fn decoder_handles_max_finite() {
    // Max finite: exp=30, mant=3 → 0b0_11110_11 = 0x7B.
    // Value = 2^(30-15) * (1 + 3/4) = 2^15 * 1.75 = 57344.
    let out = decode_e5m2_bytes(&[0x7B, 0xFB]);
    assert_eq!(out[0], 57344.0_f32);
    assert_eq!(out[1], -57344.0_f32);
}

#[test]
fn decoder_handles_smallest_normal() {
    // Smallest positive normal: exp=1, mant=0 → 0b0_00001_00 = 0x04.
    // Value = 2^(1-15) * 1 = 2^-14 ≈ 6.103515625e-5.
    let out = decode_e5m2_bytes(&[0x04]);
    assert_eq!(out[0], 2.0_f32.powi(-14));
}

#[test]
fn decoder_handles_subnormals() {
    // Smallest positive subnormal: exp=0, mant=1 → 0x01.
    // Value = 2^-14 * (1/4) = 2^-16 = 1.52587890625e-5.
    let smallest_sub = decode_e5m2_bytes(&[0x01]);
    assert_eq!(smallest_sub[0], 2.0_f32.powi(-16));

    // Largest positive subnormal: exp=0, mant=3 → 0x03.
    // Value = 2^-14 * (3/4).
    let largest_sub = decode_e5m2_bytes(&[0x03]);
    assert_eq!(largest_sub[0], 2.0_f32.powi(-14) * 0.75);
}

#[test]
fn decoder_handles_power_of_two_normals() {
    // 2.0: exp=16, mant=0 → 0b0_10000_00 = 0x40.
    // 4.0: exp=17, mant=0 → 0x44.
    // 0.5: exp=14, mant=0 → 0x38.
    let out = decode_e5m2_bytes(&[0x40, 0x44, 0x38]);
    assert_eq!(out[0], 2.0_f32);
    assert_eq!(out[1], 4.0_f32);
    assert_eq!(out[2], 0.5_f32);
}

// ---- encoder tests ----

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn encoder_produces_expected_bit_patterns_for_exact_values() {
    // 0.0, 1.0, -1.0, 2.0, 0.5 map to exact representable values.
    let out = convert_f32_to_fp8_e5m2_bytes(&f32_to_bytes(&[0.0, 1.0, -1.0, 2.0, 0.5]));
    assert_eq!(out, vec![0x00, 0x3C, 0xBC, 0x40, 0x38]);
}

#[test]
fn encoder_saturates_on_overflow() {
    // Values beyond max finite (57344) saturate to max finite, not ∞.
    let out = convert_f32_to_fp8_e5m2_bytes(&f32_to_bytes(&[65536.0, -1.0e30, 57344.0]));
    assert_eq!(out[0], 0x7B, "positive overflow should saturate");
    assert_eq!(out[1], 0xFB, "negative overflow should saturate");
    assert_eq!(out[2], 0x7B, "exactly max finite");
}

#[test]
fn encoder_preserves_inf_and_nan() {
    let out =
        convert_f32_to_fp8_e5m2_bytes(&f32_to_bytes(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN]));
    assert_eq!(out[0], 0x7C);
    assert_eq!(out[1], 0xFC);
    // NaN → canonical quiet NaN pattern (mantissa nonzero, exp=31).
    assert_eq!(out[2] & 0x7F, 0x7F);
}

#[test]
fn encoder_handles_subnormals() {
    // 2^-16 is the smallest positive subnormal representable exactly.
    let out = convert_f32_to_fp8_e5m2_bytes(&f32_to_bytes(&[2.0_f32.powi(-16)]));
    assert_eq!(out[0], 0x01);

    // Something below that rounds to zero (or to the smallest subnormal
    // under round-to-nearest; we accept either).
    let out = convert_f32_to_fp8_e5m2_bytes(&f32_to_bytes(&[1.0e-20]));
    assert!(out[0] == 0x00 || out[0] == 0x01);
}

#[test]
fn round_trip_preserves_representable_values() {
    // Pick a mix of values the format can represent exactly.
    let inputs: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 4.0, 0.25, 16.0, -16.0, 57344.0, -57344.0,
    ];
    let encoded = convert_f32_to_fp8_e5m2_bytes(&f32_to_bytes(&inputs));
    let decoded = decode_e5m2_bytes(&encoded);

    for (i, &want) in inputs.iter().enumerate() {
        assert_eq!(
            decoded[i], want,
            "round-trip diverged at index {i}: encoded={:02x} expected={want}",
            encoded[i]
        );
    }
}

#[test]
fn round_trip_is_close_for_non_exact_values() {
    // 0.1 is not exactly representable; round-trip should land at the
    // nearest E5M2 value, which is 2^-4 * (1 + 2/4) = 0.09375 (or a
    // nearby quantised value depending on rounding).
    let encoded = convert_f32_to_fp8_e5m2_bytes(&f32_to_bytes(&[0.1_f32]));
    let decoded = decode_e5m2_bytes(&encoded);

    // Relative error bound for a 2-bit mantissa format is roughly 25%
    // in the worst case (next-representable-step). Allow ≤ 30%.
    let err = (decoded[0] - 0.1_f32).abs() / 0.1_f32;
    assert!(
        err < 0.30,
        "0.1 round-trip error too large: decoded={} err={err}",
        decoded[0]
    );
}
