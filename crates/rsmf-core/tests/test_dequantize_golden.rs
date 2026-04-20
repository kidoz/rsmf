use rsmf_core::tensor::dequantize::{
    dequantize_q2_k, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k,
};

fn assert_f32_equality(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // Strict bit-for-bit equality for identical FMA
        // Allowing 1 ULP (Unit in the Last Place) for FMA differences
        let a_bits = a.to_bits();
        let e_bits = e.to_bits();

        let diff = if a_bits > e_bits {
            a_bits - e_bits
        } else {
            e_bits - a_bits
        };

        if diff > 1 {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            panic!(
                "Mismatch at index {}: actual {} (bits {:08x}), expected {} (bits {:08x}) - ULP diff {}",
                i, a, a_bits, e, e_bits, diff
            );
        }
    }
}

#[test]
fn test_q4_k_golden() {
    let raw = include_bytes!("../../../tmp/golden/golden_q4_k.raw");
    let f32_bytes = include_bytes!("../../../tmp/golden/golden_q4_k.f32");

    let expected_f32: Vec<f32> = f32_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let actual_f32 = dequantize_q4_k(raw).unwrap();
    assert_f32_equality(&actual_f32, &expected_f32);
}

#[test]
fn test_q5_k_golden() {
    let raw = include_bytes!("../../../tmp/golden/golden_q5_k.raw");
    let f32_bytes = include_bytes!("../../../tmp/golden/golden_q5_k.f32");

    let expected_f32: Vec<f32> = f32_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let actual_f32 = dequantize_q5_k(raw).unwrap();
    assert_f32_equality(&actual_f32, &expected_f32);
}

#[test]
fn test_q6_k_golden() {
    let raw = include_bytes!("../../../tmp/golden/golden_q6_k.raw");
    let f32_bytes = include_bytes!("../../../tmp/golden/golden_q6_k.f32");

    let expected_f32: Vec<f32> = f32_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let actual_f32 = dequantize_q6_k(raw).unwrap();
    assert_f32_equality(&actual_f32, &expected_f32);
}

#[test]
fn test_q2_k_golden() {
    let raw = include_bytes!("../../../tmp/golden/golden_q2_k.raw");
    let f32_bytes = include_bytes!("../../../tmp/golden/golden_q2_k.f32");

    let expected_f32: Vec<f32> = f32_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let actual_f32 = dequantize_q2_k(raw).unwrap();
    assert_f32_equality(&actual_f32, &expected_f32);
}
