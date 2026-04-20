//! K-Quants decoding and fp8 decoding implementations.

use crate::error::{Result, RsmfError};
use half::f16;

#[inline(always)]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize Q4_K block to f32.
pub fn dequantize_q4_k(bytes: &[u8]) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144;
    if bytes.len() % BLOCK_BYTES != 0 {
        return Err(RsmfError::structural(format!(
            "Q4_K length {} is not a multiple of {}",
            bytes.len(),
            BLOCK_BYTES
        )));
    }
    let nb = bytes.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);

    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let min = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
        let scales = &bytes[off + 4..off + 16];
        let qs = &bytes[off + 16..off + 144];

        let mut is = 0;
        let mut qs_off = 0;
        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = min * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = min * m2 as f32;

            for l in 0..32 {
                out.push(d1 * (qs[qs_off + l] & 0xF) as f32 - min1);
            }
            for l in 0..32 {
                out.push(d2 * (qs[qs_off + l] >> 4) as f32 - min2);
            }
            qs_off += 32;
            is += 2;
        }
    }
    Ok(out)
}

/// Dequantize Q5_K block to f32.
pub fn dequantize_q5_k(bytes: &[u8]) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 176;
    if bytes.len() % BLOCK_BYTES != 0 {
        return Err(RsmfError::structural(format!(
            "Q5_K length {} is not a multiple of {}",
            bytes.len(),
            BLOCK_BYTES
        )));
    }
    let nb = bytes.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);

    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let min = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
        let scales = &bytes[off + 4..off + 16];
        let qh = &bytes[off + 16..off + 48];
        let qs = &bytes[off + 48..off + 176];

        let mut is = 0;
        let mut qs_off = 0;
        let mut u1 = 1u8;
        let mut u2 = 2u8;

        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = min * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = min * m2 as f32;

            for l in 0..32 {
                let high_bit = if (qh[l] & u1) != 0 { 16.0 } else { 0.0 };
                out.push(d1 * ((qs[qs_off + l] & 0xF) as f32 + high_bit) - min1);
            }
            for l in 0..32 {
                let high_bit = if (qh[l] & u2) != 0 { 16.0 } else { 0.0 };
                out.push(d2 * ((qs[qs_off + l] >> 4) as f32 + high_bit) - min2);
            }
            qs_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

/// Dequantize Q6_K block to f32.
pub fn dequantize_q6_k(bytes: &[u8]) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 210;
    if bytes.len() % BLOCK_BYTES != 0 {
        return Err(RsmfError::structural(format!(
            "Q6_K length {} is not a multiple of {}",
            bytes.len(),
            BLOCK_BYTES
        )));
    }
    let nb = bytes.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);

    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let ql = &bytes[off..off + 128];
        let qh = &bytes[off + 128..off + 192];
        let sc = &bytes[off + 192..off + 208];
        let d = f16::from_le_bytes([bytes[off + 208], bytes[off + 209]]).to_f32();

        let mut ql_off = 0;
        let mut qh_off = 0;
        let mut sc_off = 0;

        for _ in (0..QK_K).step_by(128) {
            let mut y = [0.0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;

                y[l] = d * (sc[sc_off + is] as i8 as f32) * (q1 as f32);
                y[l + 32] = d * (sc[sc_off + is + 2] as i8 as f32) * (q2 as f32);
                y[l + 64] = d * (sc[sc_off + is + 4] as i8 as f32) * (q3 as f32);
                y[l + 96] = d * (sc[sc_off + is + 6] as i8 as f32) * (q4 as f32);
            }
            out.extend_from_slice(&y);
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    Ok(out)
}

/// Dequantize Q2_K block to f32.
pub fn dequantize_q2_k(bytes: &[u8]) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 84;
    if bytes.len() % BLOCK_BYTES != 0 {
        return Err(RsmfError::structural(format!(
            "Q2_K length {} is not a multiple of {}",
            bytes.len(),
            BLOCK_BYTES
        )));
    }
    let nb = bytes.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);

    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let scales = &bytes[off..off + 16];
        let q = &bytes[off + 16..off + 80];
        let d = f16::from_le_bytes([bytes[off + 80], bytes[off + 81]]).to_f32();
        let min = f16::from_le_bytes([bytes[off + 82], bytes[off + 83]]).to_f32();

        let mut is = 0;
        let mut q_off = 0;

        for _ in (0..QK_K).step_by(128) {
            let mut shift = 0;
            let mut y = [0.0f32; 128];
            for j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl1 = d * (sc & 0xF) as f32;
                let ml1 = min * (sc >> 4) as f32;
                for l in 0..16 {
                    y[j * 32 + l] = dl1 * ((q[q_off + l] >> shift) & 3) as i8 as f32 - ml1;
                }

                let sc = scales[is];
                is += 1;
                let dl2 = d * (sc & 0xF) as f32;
                let ml2 = min * (sc >> 4) as f32;
                for l in 0..16 {
                    y[j * 32 + 16 + l] = dl2 * ((q[q_off + l + 16] >> shift) & 3) as i8 as f32 - ml2;
                }

                shift += 2;
            }
            out.extend_from_slice(&y);
            q_off += 32;
        }
    }
    Ok(out)
}

/// Dequantize Q5_0 block to f32.
pub fn dequantize_q5_0(bytes: &[u8]) -> Result<Vec<f32>> {
    const QK5_0: usize = 32;
    const BLOCK_BYTES: usize = 22;
    if bytes.len() % BLOCK_BYTES != 0 {
        return Err(RsmfError::structural(format!(
            "Q5_0 length {} is not a multiple of {}",
            bytes.len(),
            BLOCK_BYTES
        )));
    }
    let nb = bytes.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK5_0);

    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qh_bytes: [u8; 4] = bytes[off + 2..off + 6].try_into().unwrap();
        let qh = u32::from_le_bytes(qh_bytes);
        let qs = &bytes[off + 6..off + 22];

        let mut y = [0.0f32; 32];
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

            let x0 = ((qs[j] & 0x0F) | xh_0) as i8 as f32 - 16.0;
            let x1 = ((qs[j] >> 4) | xh_1) as i8 as f32 - 16.0;

            y[j] = x0 * d;
            y[j + 16] = x1 * d;
        }
        out.extend_from_slice(&y);
    }
    Ok(out)
}
