//! `RsmfWriter` — builds RSMF files from in-memory tensor / graph / asset
//! inputs.
//!
//! See [`crate::reader::RsmfFile`] for the reading side and `docs/SPEC.md` for
//! the authoritative layout rules.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::checksum::{CHECKSUM_LEN, digest_128};
use crate::error::{Result, RsmfError};
use crate::manifest::{AssetDescriptor, GraphDescriptor, GraphKind, MANIFEST_VERSION, Manifest};
use crate::preamble::{FORMAT_MAJOR, FORMAT_MINOR, MAGIC, PREAMBLE_LEN, Preamble};
use crate::section::{SECTION_DESC_LEN, SectionDescriptor, SectionKind};
use crate::tensor::descriptor::TensorDescriptor;
use crate::tensor::dtype::{LogicalDtype, StorageDtype};
use crate::tensor::variant::{EncodingKind, LayoutTag, TargetTag, VariantDescriptor, VariantMeta};

/// Default alignment for the canonical arena in bytes.
pub const DEFAULT_CANONICAL_ALIGNMENT: u32 = 64;
/// Default alignment for a packed arena.
pub const DEFAULT_PACKED_ALIGNMENT: u32 = 64;
/// Default alignment for the graph section payload.
pub const DEFAULT_GRAPH_ALIGNMENT: u32 = 8;
/// Default alignment for the assets section payload.
pub const DEFAULT_ASSETS_ALIGNMENT: u32 = 8;

/// One variant's worth of input bytes plus descriptor metadata.
#[derive(Debug, Clone)]
pub struct VariantInput {
    /// Target backend tag.
    pub target: TargetTag,
    /// Encoding kind.
    pub encoding: EncodingKind,
    /// Storage dtype. `None` means inherit the owning tensor's logical dtype
    /// (required for canonical variants).
    pub storage_dtype: Option<StorageDtype>,
    /// Layout tag.
    pub layout: LayoutTag,
    /// Required payload alignment (power of two).
    pub alignment: u32,
    /// Raw bytes of the variant.
    pub bytes: Vec<u8>,
    /// Variant metadata.
    pub meta: VariantMeta,
}

impl VariantInput {
    /// Build a canonical raw variant (target=Canonical, encoding=Raw,
    /// layout=RowMajor, alignment=64, storage_dtype inherited).
    #[must_use]
    pub fn canonical_raw(bytes: Vec<u8>) -> Self {
        Self {
            target: TargetTag::Canonical,
            encoding: EncodingKind::Raw,
            storage_dtype: None,
            layout: LayoutTag::RowMajor,
            alignment: DEFAULT_CANONICAL_ALIGNMENT,
            bytes,
            meta: VariantMeta::default(),
        }
    }

    /// Build a packed f32→f16 variant. Caller must supply f16 bytes in
    /// row-major order.
    #[must_use]
    pub fn packed_cast_f16(target: TargetTag, bytes: Vec<u8>) -> Self {
        Self {
            target,
            encoding: EncodingKind::CastF16,
            storage_dtype: Some(StorageDtype::Logical(LogicalDtype::F16)),
            layout: LayoutTag::RowMajor,
            alignment: DEFAULT_PACKED_ALIGNMENT,
            bytes,
            meta: VariantMeta::default(),
        }
    }

    /// Build a packed f32→q8_0 block-quantized variant.
    #[must_use]
    pub fn packed_q8_0(target: TargetTag, bytes: Vec<u8>) -> Self {
        Self {
            target,
            encoding: EncodingKind::BlockQuantized,
            storage_dtype: Some(StorageDtype::Q8_0),
            layout: LayoutTag::Blocked,
            alignment: DEFAULT_PACKED_ALIGNMENT,
            bytes,
            meta: VariantMeta {
                block_shape: vec![32],
                group_size: 0,
                scale_dtype: Some(StorageDtype::Logical(LogicalDtype::F16)),
                zero_point_dtype: None,
                extra: vec![],
            },
        }
    }

    /// Build a packed f32→q4_0 block-quantized variant.
    #[must_use]
    pub fn packed_q4_0(target: TargetTag, bytes: Vec<u8>) -> Self {
        Self {
            target,
            encoding: EncodingKind::BlockQuantized,
            storage_dtype: Some(StorageDtype::Q4_0),
            layout: LayoutTag::Blocked,
            alignment: DEFAULT_PACKED_ALIGNMENT,
            bytes,
            meta: VariantMeta {
                block_shape: vec![32],
                group_size: 0,
                scale_dtype: Some(StorageDtype::Logical(LogicalDtype::F16)),
                zero_point_dtype: None,
                extra: vec![],
            },
        }
    }

    /// Build a packed f32→q3_k block-quantized variant.
    #[must_use]
    pub fn packed_q3_k(target: TargetTag, bytes: Vec<u8>) -> Self {
        Self {
            target,
            encoding: EncodingKind::BlockQuantized,
            storage_dtype: Some(StorageDtype::Q3K),
            layout: LayoutTag::Blocked,
            alignment: DEFAULT_PACKED_ALIGNMENT,
            bytes,
            meta: VariantMeta {
                block_shape: vec![256], // Super-block size
                group_size: 16,         // Sub-block scale group
                scale_dtype: Some(StorageDtype::Logical(LogicalDtype::F16)),
                zero_point_dtype: None,
                extra: vec![],
            },
        }
    }

    /// Build a packed f32→nf4 block-quantized variant.
    #[must_use]
    pub fn packed_nf4(target: TargetTag, bytes: Vec<u8>) -> Self {
        Self {
            target,
            encoding: EncodingKind::BlockQuantized,
            storage_dtype: Some(StorageDtype::NF4),
            layout: LayoutTag::Blocked,
            alignment: DEFAULT_PACKED_ALIGNMENT,
            bytes,
            meta: VariantMeta {
                block_shape: vec![32],
                group_size: 0,
                scale_dtype: Some(StorageDtype::Logical(LogicalDtype::F16)),
                zero_point_dtype: None,
                extra: vec![],
            },
        }
    }

    /// Build a packed f32→fp8_e5m2 variant. Caller must supply E5M2
    /// bytes already quantized (see [`convert_f32_to_fp8_e5m2_bytes`]).
    #[must_use]
    pub fn packed_fp8_e5m2(target: TargetTag, bytes: Vec<u8>) -> Self {
        Self {
            target,
            encoding: EncodingKind::Raw,
            storage_dtype: Some(StorageDtype::Fp8E5M2),
            layout: LayoutTag::RowMajor,
            alignment: DEFAULT_PACKED_ALIGNMENT,
            bytes,
            meta: VariantMeta::default(),
        }
    }

    /// Build a packed f32→fp8_e4m3 variant.
    #[must_use]
    pub fn packed_fp8_e4m3(target: TargetTag, bytes: Vec<u8>) -> Self {
        Self {
            target,
            encoding: EncodingKind::Raw,
            storage_dtype: Some(StorageDtype::Fp8E4M3),
            layout: LayoutTag::RowMajor,
            alignment: DEFAULT_PACKED_ALIGNMENT,
            bytes,
            meta: VariantMeta::default(),
        }
    }
}

/// One logical tensor plus its canonical bytes and optional packed variants.
#[derive(Debug, Clone)]
pub struct TensorInput {
    /// The shard ID this tensor belongs to.
    pub shard_id: u64,
    /// Tensor name (unique within the file).
    pub name: String,
    /// Logical dtype.
    pub dtype: LogicalDtype,
    /// Logical shape.
    pub shape: Vec<u64>,
    /// Metadata map for this tensor.
    pub metadata: Vec<(String, String)>,
    /// The canonical (target=Canonical, encoding=Raw) variant.
    pub canonical: VariantInput,
    /// Zero or more packed variants tagged by backend.
    pub packed: Vec<VariantInput>,
}

/// Input for the optional graph section.
#[derive(Debug, Clone)]
pub struct GraphInput {
    /// Graph kind.
    pub kind: GraphKind,
    /// Alignment.
    pub alignment: u32,
    /// Opaque graph bytes.
    pub bytes: Vec<u8>,
    /// Metadata.
    pub metadata: Vec<(String, String)>,
    /// If `Some(level)`, compress the graph section with zstd at the given
    /// level. Requires the `compression` feature on `rsmf-core`.
    pub compress: Option<i32>,
}

impl GraphInput {
    /// Build an ONNX graph input with default alignment.
    #[must_use]
    pub fn onnx(bytes: Vec<u8>) -> Self {
        Self {
            kind: GraphKind::Onnx,
            alignment: DEFAULT_GRAPH_ALIGNMENT,
            bytes,
            metadata: Vec::new(),
            compress: None,
        }
    }

    /// Build an ORT graph input with default alignment.
    #[must_use]
    pub fn ort(bytes: Vec<u8>) -> Self {
        Self {
            kind: GraphKind::Ort,
            alignment: DEFAULT_GRAPH_ALIGNMENT,
            bytes,
            metadata: Vec::new(),
            compress: None,
        }
    }

    /// Enable zstd compression on this graph section. Level typically 1–22
    /// (3 is a good default).
    #[must_use]
    pub fn with_compression(mut self, level: i32) -> Self {
        self.compress = Some(level);
        self
    }
}

/// Input for one named asset.
#[derive(Debug, Clone)]
pub struct AssetInput {
    /// Asset name (unique within the file).
    pub name: String,
    /// Alignment.
    pub alignment: u32,
    /// Asset bytes.
    pub bytes: Vec<u8>,
    /// Metadata.
    pub metadata: Vec<(String, String)>,
    /// If `Some(level)`, the assets section will be zstd-compressed.
    pub compress: Option<i32>,
}

impl AssetInput {
    /// Build an asset input with default alignment.
    #[must_use]
    pub fn new(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            alignment: DEFAULT_ASSETS_ALIGNMENT,
            bytes,
            metadata: Vec::new(),
            compress: None,
        }
    }

    /// Enable zstd compression on this asset's section.
    #[must_use]
    pub fn with_compression(mut self, level: i32) -> Self {
        self.compress = Some(level);
        self
    }
}

/// Convert an f32 tensor's canonical bytes to f16 storage bytes (lossy).
///
/// Used by [`RsmfWriter::with_tensor_auto_f16`] and available publicly for
/// custom pipelines.
#[must_use]
pub fn convert_f32_to_f16_bytes(f32_bytes: &[u8]) -> Vec<u8> {
    use half::f16;
    assert!(f32_bytes.len() % 4 == 0, "f32 bytes not multiple of 4");
    let count = f32_bytes.len() / 4;
    let mut out = Vec::with_capacity(count * 2);
    for i in 0..count {
        let off = i * 4;
        let val = f32::from_le_bytes([
            f32_bytes[off],
            f32_bytes[off + 1],
            f32_bytes[off + 2],
            f32_bytes[off + 3],
        ]);
        let h = f16::from_f32(val);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Convert an f32 tensor's canonical bytes to Q8_0 block-quantized bytes.
///
/// Each block of 32 f32 values is converted into a 2-byte f16 scale
/// followed by 32 i8 values.
#[must_use]
pub fn convert_f32_to_q8_0_bytes(f32_bytes: &[u8]) -> Vec<u8> {
    use half::f16;
    assert!(f32_bytes.len() % 4 == 0, "f32 bytes not multiple of 4");
    let count = f32_bytes.len() / 4;
    let block_size = 32;
    let num_blocks = count.div_ceil(block_size);
    let mut out = Vec::with_capacity(num_blocks * (2 + block_size));

    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(count);

        let mut max_abs: f32 = 0.0;
        for i in start..end {
            let off = i * 4;
            let val = f32::from_le_bytes([
                f32_bytes[off],
                f32_bytes[off + 1],
                f32_bytes[off + 2],
                f32_bytes[off + 3],
            ]);
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let scale_f16 = f16::from_f32(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        for i in start..start + block_size {
            if i < end {
                let off = i * 4;
                let val = f32::from_le_bytes([
                    f32_bytes[off],
                    f32_bytes[off + 1],
                    f32_bytes[off + 2],
                    f32_bytes[off + 3],
                ]);
                let q = (val / scale).round().clamp(-127.0, 127.0) as i8;
                out.push(q as u8);
            } else {
                out.push(0);
            }
        }
    }
    out
}

/// Convert an f32 tensor's canonical bytes to Q4_0 block-quantized bytes.
///
/// Each block of 32 f32 values is converted into a 2-byte f16 scale
/// followed by 16 bytes (32 elements) of 4-bit weights.
#[must_use]
pub fn convert_f32_to_q4_0_bytes(f32_bytes: &[u8]) -> Vec<u8> {
    use half::f16;
    assert!(f32_bytes.len() % 4 == 0, "f32 bytes not multiple of 4");
    let count = f32_bytes.len() / 4;
    let block_size = 32;
    let num_blocks = count.div_ceil(block_size);
    let mut out = Vec::with_capacity(num_blocks * (2 + 16));

    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(count);

        let mut max_abs: f32 = 0.0;
        for i in start..end {
            let off = i * 4;
            let val = f32::from_le_bytes([
                f32_bytes[off],
                f32_bytes[off + 1],
                f32_bytes[off + 2],
                f32_bytes[off + 3],
            ]);
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }

        let scale = if max_abs > 0.0 { max_abs / 8.0 } else { 1.0 };
        let scale_f16 = f16::from_f32(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        for i in 0..16 {
            let idx0 = start + i;
            let idx1 = start + i + 16;

            let q0 = if idx0 < end {
                let off = idx0 * 4;
                let val = f32::from_le_bytes([
                    f32_bytes[off],
                    f32_bytes[off + 1],
                    f32_bytes[off + 2],
                    f32_bytes[off + 3],
                ]);
                ((val / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8
            } else {
                8
            };

            let q1 = if idx1 < end {
                let off = idx1 * 4;
                let val = f32::from_le_bytes([
                    f32_bytes[off],
                    f32_bytes[off + 1],
                    f32_bytes[off + 2],
                    f32_bytes[off + 3],
                ]);
                ((val / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8
            } else {
                8
            };

            out.push(q0 | (q1 << 4));
        }
    }
    out
}

/// NF4 levels (from QLoRA paper).
const NF4_LEVELS: [f32; 16] = [
    -1.0000, -0.6944, -0.5122, -0.3739, -0.2561, -0.1496, -0.0497, 0.0000, 0.0624, 0.1236, 0.2200,
    0.3319, 0.4712, 0.6565, 0.7906, 1.0000,
];

/// Convert an f32 tensor's canonical bytes to NF4 block-quantized bytes.
///
/// Uses blocks of 32. Each block has a 2-byte f16 scale and 16 bytes of packed 4-bit indices.
#[must_use]
pub fn convert_f32_to_nf4_bytes(f32_bytes: &[u8]) -> Vec<u8> {
    use half::f16;
    assert!(f32_bytes.len() % 4 == 0, "f32 bytes not multiple of 4");
    let count = f32_bytes.len() / 4;
    let block_size = 32;
    let num_blocks = count.div_ceil(block_size);
    let mut out = Vec::with_capacity(num_blocks * 18);

    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(count);

        let mut max_abs: f32 = 0.0;
        for i in start..end {
            let val = f32::from_le_bytes([
                f32_bytes[i * 4],
                f32_bytes[i * 4 + 1],
                f32_bytes[i * 4 + 2],
                f32_bytes[i * 4 + 3],
            ]);
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }
        let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
        out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());

        for i in 0..16 {
            let idx0 = start + i;
            let idx1 = start + i + 16;

            let q0 = if idx0 < end {
                let val = f32::from_le_bytes([
                    f32_bytes[idx0 * 4],
                    f32_bytes[idx0 * 4 + 1],
                    f32_bytes[idx0 * 4 + 2],
                    f32_bytes[idx0 * 4 + 3],
                ]) / scale;
                NF4_LEVELS
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (val - *a)
                            .abs()
                            .partial_cmp(&(val - *b).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((7, &0.0))
                    .0 as u8
            } else {
                7
            }; // index of 0.0

            let q1 = if idx1 < end {
                let val = f32::from_le_bytes([
                    f32_bytes[idx1 * 4],
                    f32_bytes[idx1 * 4 + 1],
                    f32_bytes[idx1 * 4 + 2],
                    f32_bytes[idx1 * 4 + 3],
                ]) / scale;
                NF4_LEVELS
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (val - *a)
                            .abs()
                            .partial_cmp(&(val - *b).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((7, &0.0))
                    .0 as u8
            } else {
                7
            };

            out.push(q0 | (q1 << 4));
        }
    }
    out
}

/// Convert an f32 tensor's canonical bytes to FP8 E4M3 bytes.
#[must_use]
/// Convert an f32 tensor's canonical bytes to OCP FP8 E5M2 storage
/// bytes (1 sign, 5 exponent, 2 mantissa, bias 15, IEEE-like).
///
/// - Finite values outside the E5M2 range saturate to the max finite
///   magnitude (57344.0) — the "saturating-to-finite" overflow mode.
/// - NaN maps to the canonical E5M2 NaN (`0x7F`).
/// - ±∞ is preserved (`0x7C` / `0xFC`).
/// - Denormalised f32 inputs that fall below E5M2's minimum subnormal
///   (`2^-16`) round to zero with the sign preserved.
///
/// Rounding mode is round-to-nearest-even on the mantissa. The encoder
/// is symmetric with the decoder in `view.rs`, so round-trip through
/// the writer and `TensorView::decode_f32` reproduces the closest
/// representable E5M2 value to the original f32.
pub fn convert_f32_to_fp8_e5m2_bytes(f32_bytes: &[u8]) -> Vec<u8> {
    assert!(
        f32_bytes.len() % 4 == 0,
        "f32 bytes length not a multiple of 4"
    );
    let count = f32_bytes.len() / 4;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let val = f32::from_le_bytes([
            f32_bytes[i * 4],
            f32_bytes[i * 4 + 1],
            f32_bytes[i * 4 + 2],
            f32_bytes[i * 4 + 3],
        ]);
        out.push(encode_one_fp8_e5m2(val));
    }
    out
}

/// Encode a single `f32` into its E5M2 byte. See
/// [`convert_f32_to_fp8_e5m2_bytes`] for semantics.
fn encode_one_fp8_e5m2(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign_bit: u8 = if (bits >> 31) != 0 { 0x80 } else { 0 };

    if val.is_nan() {
        // Canonical quiet NaN with a non-zero mantissa; sign carried through.
        return sign_bit | 0x7F;
    }
    if val.is_infinite() {
        return sign_bit | 0x7C;
    }

    let abs_val = val.abs();
    if abs_val == 0.0 {
        return sign_bit;
    }

    // E5M2 max finite value is 2^15 * (1 + 3/4) = 57344.
    const MAX_FINITE: f32 = 57344.0;
    if abs_val >= MAX_FINITE * 2.0 {
        // Far past max finite → saturate to max finite.
        return sign_bit | 0x7B;
    }

    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x007F_FFFF;
    let e5m2_exp = f32_exp + 15;

    if e5m2_exp >= 31 {
        // Overflow in the normal range → saturate to max finite.
        return sign_bit | 0x7B;
    }

    if e5m2_exp <= 0 {
        // Subnormal or underflow. E5M2 subnormal: value = ±2^-14 * m/4,
        // m ∈ {1, 2, 3}. Convert via direct scaling + round-to-nearest.
        let scaled = abs_val * 2.0_f32.powi(14) * 4.0;
        let rounded = scaled.round();
        if rounded <= 0.0 {
            return sign_bit;
        }
        let m = rounded as u32;
        if m >= 4 {
            // Rounded up into the smallest normal: exp=1, mant=0.
            return sign_bit | 0x04;
        }
        return sign_bit | (m as u8);
    }

    // Normal range. Round-to-nearest-even on the two-bit mantissa.
    let shift = 23 - 2;
    let remainder = f32_mant & ((1 << shift) - 1);
    let half = 1u32 << (shift - 1);
    let mut m = f32_mant >> shift;
    // Round-to-nearest, ties-to-even.
    if remainder > half || (remainder == half && (m & 1) == 1) {
        m += 1;
    }
    let mut exp = e5m2_exp as u32;
    if m == 4 {
        // Mantissa overflowed → carry into the exponent.
        m = 0;
        exp += 1;
        if exp >= 31 {
            return sign_bit | 0x7B;
        }
    }
    sign_bit | ((exp as u8) << 2) | (m as u8 & 0x03)
}

/// Convert an f32 tensor's canonical bytes to OCP FP8 E4M3 storage
/// bytes (1 sign, 4 exponent, 3 mantissa, bias 7, no infinity).
/// Saturates overflow to the E4M3 max finite magnitude (448.0).
pub fn convert_f32_to_fp8_e4m3_bytes(f32_bytes: &[u8]) -> Vec<u8> {
    assert!(f32_bytes.len() % 4 == 0, "f32 bytes not multiple of 4");
    let count = f32_bytes.len() / 4;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let val = f32::from_le_bytes([
            f32_bytes[i * 4],
            f32_bytes[i * 4 + 1],
            f32_bytes[i * 4 + 2],
            f32_bytes[i * 4 + 3],
        ]);

        // Simple E4M3 conversion (Sign: 1, Exp: 4, Mant: 3, Bias: 7)
        let sign = if val.is_sign_negative() { 0x80 } else { 0 };
        let abs_val = val.abs();

        if abs_val == 0.0 {
            out.push(sign);
        } else if abs_val >= 448.0 {
            out.push(sign | 0x7E); // Max value
        } else {
            let bits = val.to_bits();
            let exp = ((bits >> 23) & 0xFF) as i32 - 127;
            let mant = bits & 0x7FFFFF;

            let fp8_exp = (exp + 7).clamp(0, 15) as u8;
            let fp8_mant = (mant >> (23 - 3)) as u8;
            out.push(sign | (fp8_exp << 3) | fp8_mant);
        }
    }
    out
}

/// Convert an f32 tensor's canonical bytes to Q3_K block-quantized bytes.
///
/// Uses super-blocks of 256 elements.
/// Layout: [hmask (32B)][qs (64B)][scales (12B)][super_scale (2B f16)] = 110 bytes per 256 elems.
#[must_use]
pub fn convert_f32_to_q3_k_bytes(f32_bytes: &[u8]) -> Vec<u8> {
    use half::f16;
    let count = f32_bytes.len() / 4;
    let block_size = 256;
    let num_blocks = count.div_ceil(block_size);
    let mut out = Vec::with_capacity(num_blocks * 110);

    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(count);

        let mut max_abs: f32 = 0.0;
        for i in start..end {
            let val = f32::from_le_bytes([
                f32_bytes[i * 4],
                f32_bytes[i * 4 + 1],
                f32_bytes[i * 4 + 2],
                f32_bytes[i * 4 + 3],
            ]);
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }
        let super_scale = if max_abs > 0.0 { max_abs / 4.0 } else { 1.0 };

        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];

        for i in 0..256 {
            let idx = start + i;
            let val = if idx < end {
                f32::from_le_bytes([
                    f32_bytes[idx * 4],
                    f32_bytes[idx * 4 + 1],
                    f32_bytes[idx * 4 + 2],
                    f32_bytes[idx * 4 + 3],
                ])
            } else {
                0.0
            };

            let q = ((val / super_scale).round().clamp(-4.0, 3.0) as i8 + 4) as u8;
            let low = q & 0x3;
            let qs_idx = i / 4;
            let qs_shift = (i % 4) * 2;
            qs[qs_idx] |= low << qs_shift;

            if (q & 0x4) != 0 {
                hmask[i / 8] |= 1 << (i % 8);
            }
        }

        out.extend_from_slice(&hmask);
        out.extend_from_slice(&qs);
        out.extend_from_slice(&[0u8; 12]);
        out.extend_from_slice(&f16::from_f32(super_scale).to_le_bytes());
    }
    out
}

/// Builds an RSMF file from inputs.
#[derive(Debug, Default, Clone)]
pub struct RsmfWriter {
    metadata: Vec<(String, String)>,
    tensors: Vec<TensorInput>,
    graphs: Vec<GraphInput>,
    assets: Vec<AssetInput>,
    canonical_section_alignment: u16,
    packed_section_alignment: u16,
    canonical_compress: Option<i32>,
    packed_compress: Option<i32>,
    dedup_canonical: bool,
    dedup_packed: bool,
}

impl RsmfWriter {
    /// Create a new, empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            graphs: Vec::new(),
            assets: Vec::new(),
            canonical_section_alignment: 64,
            packed_section_alignment: 64,
            canonical_compress: None,
            packed_compress: None,
            dedup_canonical: false,
            dedup_packed: false,
        }
    }

    /// Return a mutable reference to the internal tensors list.
    pub fn tensors(&mut self) -> &mut Vec<TensorInput> {
        &mut self.tensors
    }

    /// Append a global metadata key/value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// Append a tensor input.
    #[must_use]
    pub fn with_tensor(mut self, t: TensorInput) -> Self {
        self.tensors.push(t);
        self
    }

    /// Append a tensor input and automatically generate a `CastF16` packed
    /// variant for the given target tag.
    pub fn with_tensor_auto_f16(mut self, mut t: TensorInput, target: TargetTag) -> Self {
        if t.dtype == LogicalDtype::F32 {
            let f16_bytes = convert_f32_to_f16_bytes(&t.canonical.bytes);
            t.packed
                .push(VariantInput::packed_cast_f16(target, f16_bytes));
        }
        self.tensors.push(t);
        self
    }

    /// Append a tensor input and automatically generate a `Q8_0` variant.
    pub fn with_tensor_auto_q8_0(mut self, mut t: TensorInput, target: TargetTag) -> Self {
        if t.dtype == LogicalDtype::F32 {
            let q_bytes = convert_f32_to_q8_0_bytes(&t.canonical.bytes);
            t.packed.push(VariantInput::packed_q8_0(target, q_bytes));
        }
        self.tensors.push(t);
        self
    }

    /// Append a tensor input and automatically generate a `Q4_0` variant.
    pub fn with_tensor_auto_q4_0(mut self, mut t: TensorInput, target: TargetTag) -> Self {
        if t.dtype == LogicalDtype::F32 {
            let q_bytes = convert_f32_to_q4_0_bytes(&t.canonical.bytes);
            t.packed.push(VariantInput::packed_q4_0(target, q_bytes));
        }
        self.tensors.push(t);
        self
    }

    /// Append a tensor input and automatically generate a `Q3_K` variant.
    pub fn with_tensor_auto_q3_k(mut self, mut t: TensorInput, target: TargetTag) -> Self {
        if t.dtype == LogicalDtype::F32 {
            let q_bytes = convert_f32_to_q3_k_bytes(&t.canonical.bytes);
            t.packed.push(VariantInput::packed_q3_k(target, q_bytes));
        }
        self.tensors.push(t);
        self
    }

    /// Append a tensor input and automatically generate an `NF4` variant.
    pub fn with_tensor_auto_nf4(mut self, mut t: TensorInput, target: TargetTag) -> Self {
        if t.dtype == LogicalDtype::F32 {
            let q_bytes = convert_f32_to_nf4_bytes(&t.canonical.bytes);
            t.packed.push(VariantInput::packed_nf4(target, q_bytes));
        }
        self.tensors.push(t);
        self
    }

    /// Append a tensor input and automatically generate an `FP8_E4M3` variant.
    /// Append a tensor input and automatically generate an `Fp8E5M2`
    /// packed variant for the given target (F32 tensors only; others
    /// pass through unchanged).
    #[must_use]
    pub fn with_tensor_auto_fp8_e5m2(mut self, mut t: TensorInput, target: TargetTag) -> Self {
        if t.dtype == LogicalDtype::F32 {
            let q_bytes = convert_f32_to_fp8_e5m2_bytes(&t.canonical.bytes);
            t.packed
                .push(VariantInput::packed_fp8_e5m2(target, q_bytes));
        }
        self.tensors.push(t);
        self
    }

    /// Append a tensor input and automatically generate an `Fp8E4M3`
    /// packed variant for the given target (F32 tensors only; others
    /// pass through unchanged).
    #[must_use]
    pub fn with_tensor_auto_fp8_e4m3(mut self, mut t: TensorInput, target: TargetTag) -> Self {
        if t.dtype == LogicalDtype::F32 {
            let q_bytes = convert_f32_to_fp8_e4m3_bytes(&t.canonical.bytes);
            t.packed
                .push(VariantInput::packed_fp8_e4m3(target, q_bytes));
        }
        self.tensors.push(t);
        self
    }

    /// Set the graph input.
    #[must_use]
    pub fn with_graph(mut self, g: GraphInput) -> Self {
        self.graphs.push(g);
        self
    }

    /// Append an asset input.
    #[must_use]
    pub fn with_asset(mut self, a: AssetInput) -> Self {
        self.assets.push(a);
        self
    }

    /// Enable zstd compression on the canonical arena.
    #[must_use]
    pub fn with_canonical_compression(mut self, level: i32) -> Self {
        self.canonical_compress = Some(level);
        self
    }

    /// Enable zstd compression on the packed arena(s).
    #[must_use]
    pub fn with_packed_compression(mut self, level: i32) -> Self {
        self.packed_compress = Some(level);
        self
    }

    /// Enable content-addressable dedup for both the canonical and packed
    /// arenas. Variants whose raw bytes hash to the same BLAKE3-128 digest
    /// and whose existing offset satisfies the new variant's alignment
    /// share a single span in the arena — the common case is tied
    /// embeddings, repeated bias vectors, or rewritten shards carrying
    /// unchanged weights. Disabled by default: dedup is a semantic choice
    /// (it implies byte-level identity is meaningful to downstream tools),
    /// so callers opt in.
    #[must_use]
    pub fn with_dedup(mut self, enabled: bool) -> Self {
        self.dedup_canonical = enabled;
        self.dedup_packed = enabled;
        self
    }

    /// Enable dedup for the canonical arena only.
    #[must_use]
    pub fn with_canonical_dedup(mut self, enabled: bool) -> Self {
        self.dedup_canonical = enabled;
        self
    }

    /// Enable dedup for the packed arena(s) only.
    #[must_use]
    pub fn with_packed_dedup(mut self, enabled: bool) -> Self {
        self.dedup_packed = enabled;
        self
    }

    /// Encode the file to an in-memory byte vector.
    pub fn write_to_bytes(self) -> Result<Vec<u8>> {
        self.encode()
    }

    /// Encode the file and write it atomically to `path`.
    pub fn write_to_path(self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = self.encode()?;
        write_atomic(path, &bytes)
    }

    #[allow(clippy::too_many_lines)]
    fn encode(self) -> Result<Vec<u8>> {
        self.validate_inputs()?;

        // 1. Lay out canonical arena bytes.
        let mut canonical_arena_raw: Vec<u8> = Vec::new();
        let mut canonical_layout: Vec<CanonicalLayout> = Vec::with_capacity(self.tensors.len());
        let mut canonical_dedup: HashMap<[u8; CHECKSUM_LEN], (u64, u64)> = HashMap::new();

        for t in &self.tensors {
            let align = t.canonical.alignment as u64;
            let length = t.canonical.bytes.len() as u64;
            let checksum = digest_128(&t.canonical.bytes);
            let offset = append_variant_bytes(
                &mut canonical_arena_raw,
                self.dedup_canonical.then_some(&mut canonical_dedup),
                &t.canonical.bytes,
                align,
                checksum,
            );
            let storage_dtype = t
                .canonical
                .storage_dtype
                .unwrap_or(StorageDtype::Logical(t.dtype));
            canonical_layout.push(CanonicalLayout {
                offset,
                length,
                checksum,
                storage_dtype,
            });
        }
        let canonical_arena_bytes = maybe_compress(&canonical_arena_raw, self.canonical_compress)?;

        // 2. Lay out packed arena bytes, one per arena group.
        use crate::tensor::variant::ArenaGroup;
        use std::collections::BTreeMap;
        let mut arena_groups: BTreeMap<ArenaGroup, Vec<u8>> = BTreeMap::new();
        let mut packed_dedups: BTreeMap<ArenaGroup, HashMap<[u8; CHECKSUM_LEN], (u64, u64)>> =
            BTreeMap::new();
        let mut packed_layout: Vec<Vec<PackedLayout>> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let mut per_tensor = Vec::with_capacity(t.packed.len());
            for p in &t.packed {
                let group = p.target.arena_group();
                let arena = arena_groups.entry(group).or_default();
                let align = p.alignment as u64;
                let length = p.bytes.len() as u64;
                let checksum = digest_128(&p.bytes);
                let dedup = if self.dedup_packed {
                    Some(packed_dedups.entry(group).or_default())
                } else {
                    None
                };
                let offset = append_variant_bytes(arena, dedup, &p.bytes, align, checksum);
                let storage_dtype = p.storage_dtype.ok_or_else(|| {
                    RsmfError::structural(format!(
                        "packed variant for tensor {:?} target {:?} must set storage_dtype",
                        t.name,
                        p.target.name()
                    ))
                })?;
                per_tensor.push(PackedLayout {
                    offset,
                    length,
                    checksum,
                    storage_dtype,
                    arena_group: group,
                });
            }
            packed_layout.push(per_tensor);
        }
        let mut packed_arenas: Vec<(ArenaGroup, Vec<u8>)> = Vec::with_capacity(arena_groups.len());
        for (g, raw_bytes) in arena_groups {
            let bytes = maybe_compress(&raw_bytes, self.packed_compress)?;
            packed_arenas.push((g, bytes));
        }

        let packed_group_to_index: BTreeMap<ArenaGroup, u8> = packed_arenas
            .iter()
            .enumerate()
            .map(|(i, (g, _))| (*g, i as u8))
            .collect();

        // 3. Build the manifest.
        let graph_compress = self.graphs.iter().find_map(|g| g.compress);
        let mut graph_section_raw: Vec<u8> = Vec::new();
        let mut graph_descriptors: Vec<GraphDescriptor> = Vec::with_capacity(self.graphs.len());
        for g in &self.graphs {
            let align = g.alignment as u64;
            pad_to_alignment(&mut graph_section_raw, align);
            let offset = graph_section_raw.len() as u64;
            graph_section_raw.extend_from_slice(&g.bytes);
            let length = g.bytes.len() as u64;
            let checksum = digest_128(&g.bytes);
            graph_descriptors.push(GraphDescriptor {
                kind: g.kind,
                alignment: g.alignment,
                offset,
                length,
                checksum,
                metadata: g.metadata.clone(),
            });
        }
        let graph_section_bytes = if !self.graphs.is_empty() {
            Some(maybe_compress(&graph_section_raw, graph_compress)?)
        } else {
            None
        };

        let assets_compress = self.assets.iter().find_map(|a| a.compress);
        let mut asset_section_raw: Vec<u8> = Vec::new();
        let mut asset_descriptors: Vec<AssetDescriptor> = Vec::with_capacity(self.assets.len());
        for a in &self.assets {
            let align = a.alignment as u64;
            pad_to_alignment(&mut asset_section_raw, align);
            let offset = asset_section_raw.len() as u64;
            asset_section_raw.extend_from_slice(&a.bytes);
            let length = a.bytes.len() as u64;
            let checksum = digest_128(&a.bytes);
            asset_descriptors.push(AssetDescriptor {
                name: a.name.clone(),
                alignment: a.alignment,
                offset,
                length,
                checksum,
                metadata: a.metadata.clone(),
            });
        }
        let asset_section_bytes = if !self.assets.is_empty() {
            Some(maybe_compress(&asset_section_raw, assets_compress)?)
        } else {
            None
        };

        // 4. Build variants.
        let mut variants: Vec<VariantDescriptor> = Vec::new();
        let mut tensor_canonical_variant_idx: Vec<u32> = Vec::with_capacity(self.tensors.len());
        let mut tensor_packed_variant_idxs: Vec<Vec<u32>> = Vec::with_capacity(self.tensors.len());
        for (ti, t) in self.tensors.iter().enumerate() {
            let layout = &canonical_layout[ti];
            let descr = VariantDescriptor {
                target: TargetTag::Canonical,
                encoding: EncodingKind::Raw,
                storage_dtype: layout.storage_dtype,
                layout: LayoutTag::RowMajor,
                alignment: t.canonical.alignment,
                section_relative_offset: layout.offset,
                length: layout.length,
                checksum: layout.checksum,
                section_kind: SectionKind::CanonicalArena.to_raw() as u8,
                section_index: 0,
                meta: t.canonical.meta.clone(),
            };
            tensor_canonical_variant_idx.push(variants.len() as u32);
            variants.push(descr);
        }
        for (ti, t) in self.tensors.iter().enumerate() {
            let mut per_tensor_idxs = Vec::with_capacity(t.packed.len());
            for (pi, p) in t.packed.iter().enumerate() {
                let layout = &packed_layout[ti][pi];
                let section_index = packed_group_to_index
                    .get(&layout.arena_group)
                    .copied()
                    .unwrap_or(0);
                let descr = VariantDescriptor {
                    target: p.target,
                    encoding: p.encoding,
                    storage_dtype: layout.storage_dtype,
                    layout: p.layout,
                    alignment: p.alignment,
                    section_relative_offset: layout.offset,
                    length: layout.length,
                    checksum: layout.checksum,
                    section_kind: SectionKind::PackedArena.to_raw() as u8,
                    section_index,
                    meta: p.meta.clone(),
                };
                per_tensor_idxs.push(variants.len() as u32);
                variants.push(descr);
            }
            tensor_packed_variant_idxs.push(per_tensor_idxs);
        }

        // 5. Build tensor descriptors.
        let tensors_desc: Vec<TensorDescriptor> = self
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| TensorDescriptor {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                canonical_variant: tensor_canonical_variant_idx[i],
                packed_variants: tensor_packed_variant_idxs[i].clone(),
                shard_id: t.shard_id,
                metadata: t.metadata.clone(),
            })
            .collect();

        let manifest = Manifest {
            version: MANIFEST_VERSION,
            metadata: self.metadata.clone(),
            tensors: tensors_desc,
            variants,
            graphs: graph_descriptors,
            assets: asset_descriptors,
        };
        let manifest_bytes = manifest.encode()?;

        // 6. Compute section table.
        let mut section_count: u64 = 2; // manifest + canonical
        section_count += packed_arenas.len() as u64;
        if graph_section_bytes.is_some() {
            section_count += 1;
        }
        if asset_section_bytes.is_some() {
            section_count += 1;
        }
        let section_tbl_off = PREAMBLE_LEN;
        let section_tbl_len = section_count * SECTION_DESC_LEN;

        let mut cursor = section_tbl_off + section_tbl_len;

        // Manifest
        cursor = align_up_u64(cursor, 8);
        let manifest_off = cursor;
        let manifest_len = manifest_bytes.len() as u64;
        let manifest_checksum = digest_128(&manifest_bytes);
        cursor += manifest_len;

        // Canonical arena
        cursor = align_up_u64(cursor, u64::from(self.canonical_section_alignment));
        let canonical_off = cursor;
        let canonical_len = canonical_arena_bytes.len() as u64;
        let canonical_checksum = digest_128(&canonical_arena_bytes);
        cursor += canonical_len;

        // Packed arenas
        let mut packed_arena_layouts: Vec<(u64, u64, [u8; CHECKSUM_LEN])> =
            Vec::with_capacity(packed_arenas.len());
        for (_group, arena_bytes) in &packed_arenas {
            cursor = align_up_u64(cursor, u64::from(self.packed_section_alignment));
            let off = cursor;
            let len = arena_bytes.len() as u64;
            let checksum = digest_128(arena_bytes);
            packed_arena_layouts.push((off, len, checksum));
            cursor += len;
        }

        // Graph
        let mut graph_off = 0u64;
        let mut graph_len = 0u64;
        let mut graph_checksum = [0u8; CHECKSUM_LEN];
        if let Some(g_bytes) = &graph_section_bytes {
            cursor = align_up_u64(cursor, 8);
            graph_off = cursor;
            graph_len = g_bytes.len() as u64;
            graph_checksum = digest_128(g_bytes);
            cursor += graph_len;
        }

        // Assets
        let mut assets_off = 0u64;
        let mut assets_len = 0u64;
        let mut assets_checksum = [0u8; CHECKSUM_LEN];
        if let Some(a_bytes) = &asset_section_bytes {
            cursor = align_up_u64(cursor, 8);
            assets_off = cursor;
            assets_len = a_bytes.len() as u64;
            assets_checksum = digest_128(a_bytes);
            cursor += assets_len;
        }

        let total_len = cursor;

        // 7. Build section table entries.
        let mut section_table: Vec<SectionDescriptor> = Vec::with_capacity(section_count as usize);
        section_table.push(SectionDescriptor {
            kind: SectionKind::Manifest,
            align: 8,
            flags: 0,
            offset: manifest_off,
            length: manifest_len,
            checksum: manifest_checksum,
        });
        section_table.push(SectionDescriptor {
            kind: SectionKind::CanonicalArena,
            align: self.canonical_section_alignment,
            flags: compressed_section_flags(self.canonical_compress.is_some()),
            offset: canonical_off,
            length: canonical_len,
            checksum: canonical_checksum,
        });
        for (i, (_group, _arena_bytes)) in packed_arenas.iter().enumerate() {
            let (off, len, checksum) = packed_arena_layouts[i];
            section_table.push(SectionDescriptor {
                kind: SectionKind::PackedArena,
                align: self.packed_section_alignment,
                flags: compressed_section_flags(self.packed_compress.is_some()),
                offset: off,
                length: len,
                checksum,
            });
        }
        if graph_section_bytes.is_some() {
            section_table.push(SectionDescriptor {
                kind: SectionKind::Graph,
                align: 8,
                flags: compressed_section_flags(graph_compress.is_some()),
                offset: graph_off,
                length: graph_len,
                checksum: graph_checksum,
            });
        }
        if asset_section_bytes.is_some() {
            section_table.push(SectionDescriptor {
                kind: SectionKind::Assets,
                align: 8,
                flags: compressed_section_flags(assets_compress.is_some()),
                offset: assets_off,
                length: assets_len,
                checksum: assets_checksum,
            });
        }

        // 8. Preamble.
        let preamble = Preamble {
            magic: MAGIC,
            major: FORMAT_MAJOR,
            minor: FORMAT_MINOR,
            flags: 0,
            header_len: PREAMBLE_LEN,
            section_tbl_off,
            section_tbl_count: section_count,
            manifest_off,
            manifest_len,
            preamble_checksum: [0u8; 8],
        };
        let preamble_bytes = preamble.encode();

        // 9. Assemble.
        let mut out = Vec::with_capacity(total_len as usize);
        out.extend_from_slice(&preamble_bytes);
        for entry in &section_table {
            out.extend_from_slice(&entry.encode());
        }
        pad_to_file_offset(&mut out, manifest_off)?;
        out.extend_from_slice(&manifest_bytes);
        pad_to_file_offset(&mut out, canonical_off)?;
        out.extend_from_slice(&canonical_arena_bytes);
        for (i, (_group, arena_bytes)) in packed_arenas.iter().enumerate() {
            let (off, _len, _checksum) = packed_arena_layouts[i];
            pad_to_file_offset(&mut out, off)?;
            out.extend_from_slice(arena_bytes);
        }
        if let Some(g_bytes) = &graph_section_bytes {
            pad_to_file_offset(&mut out, graph_off)?;
            out.extend_from_slice(g_bytes);
        }
        if let Some(a_bytes) = &asset_section_bytes {
            pad_to_file_offset(&mut out, assets_off)?;
            out.extend_from_slice(a_bytes);
        }
        debug_assert_eq!(out.len() as u64, total_len);
        Ok(out)
    }

    fn validate_inputs(&self) -> Result<()> {
        let mut seen_names: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for t in &self.tensors {
            if !seen_names.insert(t.name.as_str()) {
                return Err(RsmfError::structural(format!(
                    "duplicate tensor name: {}",
                    t.name
                )));
            }
            if t.canonical.target != TargetTag::Canonical
                || t.canonical.encoding != EncodingKind::Raw
            {
                return Err(RsmfError::structural(format!(
                    "tensor {} canonical variant is invalid",
                    t.name
                )));
            }
            let expected_bytes = t
                .canonical_bytes_expected()
                .ok_or_else(|| RsmfError::structural("shape overflow"))?;
            if t.canonical.bytes.len() as u64 != expected_bytes {
                return Err(RsmfError::structural(format!(
                    "tensor {} byte length mismatch",
                    t.name
                )));
            }
            for p in &t.packed {
                if p.target == TargetTag::Canonical {
                    return Err(RsmfError::structural(format!(
                        "tensor {} packed variant cannot use target=Canonical",
                        t.name
                    )));
                }
                if p.storage_dtype.is_none() {
                    return Err(RsmfError::structural(format!(
                        "tensor {} packed variant missing storage_dtype",
                        t.name
                    )));
                }
            }
        }
        Ok(())
    }
}

impl TensorInput {
    fn canonical_bytes_expected(&self) -> Option<u64> {
        let mut acc: u64 = 1;
        for &d in &self.shape {
            acc = acc.checked_mul(d)?;
        }
        acc.checked_mul(self.dtype.size_bytes() as u64)
    }
}

struct CanonicalLayout {
    offset: u64,
    length: u64,
    checksum: [u8; CHECKSUM_LEN],
    storage_dtype: StorageDtype,
}

struct PackedLayout {
    offset: u64,
    length: u64,
    checksum: [u8; CHECKSUM_LEN],
    storage_dtype: StorageDtype,
    arena_group: crate::tensor::variant::ArenaGroup,
}

fn align_up_u64(value: u64, align: u64) -> u64 {
    if align <= 1 {
        return value;
    }
    (value + (align - 1)) & !(align - 1)
}

/// Section-table flags for an arena that may or may not be compressed.
/// When the batch writer compresses, it also bit-shuffles, so both flags
/// are set together; the reader honors `SECTION_FLAG_BIT_SHUFFLED` to
/// decide whether to un-shuffle after zstd-decode. Keeping the two bits
/// coupled here means compressed sections the batch writer emits are
/// byte-identical to the pre-flag behaviour.
fn compressed_section_flags(compressed: bool) -> u32 {
    if compressed {
        crate::section::SECTION_FLAG_COMPRESSED | crate::section::SECTION_FLAG_BIT_SHUFFLED
    } else {
        0
    }
}

fn maybe_compress(bytes: &[u8], level: Option<i32>) -> Result<Vec<u8>> {
    let Some(_level) = level else {
        return Ok(bytes.to_vec());
    };
    #[cfg(feature = "compression")]
    {
        let shuffled = crate::bit_shuffle::shuffle(bytes, 4);
        zstd::encode_all(std::io::Cursor::new(&shuffled), _level)
            .map_err(|e| RsmfError::structural(format!("zstd compression failed: {e}")))
    }
    #[cfg(not(feature = "compression"))]
    {
        Err(RsmfError::unsupported(
            "compression requested but the `compression` feature is not enabled".to_string(),
        ))
    }
}

fn pad_to_alignment(bytes: &mut Vec<u8>, align: u64) {
    let target = align_up_u64(bytes.len() as u64, align) as usize;
    if bytes.len() < target {
        bytes.resize(target, 0);
    }
}

/// Append `bytes` to `arena` at the next aligned offset, optionally
/// reusing an existing offset when a prior identical span is present.
///
/// When `dedup` is `Some`, the caller has opted into content-addressable
/// storage: if the BLAKE3-128 digest has been seen before AND the earlier
/// offset already satisfies `align`, the earlier offset is returned and
/// `arena` is left unchanged. Otherwise the bytes are appended as usual
/// and the new (offset, length) pair is memoised.
fn append_variant_bytes(
    arena: &mut Vec<u8>,
    dedup: Option<&mut HashMap<[u8; CHECKSUM_LEN], (u64, u64)>>,
    bytes: &[u8],
    align: u64,
    checksum: [u8; CHECKSUM_LEN],
) -> u64 {
    let length = bytes.len() as u64;
    if let Some(map) = dedup {
        if let Some(&(existing_offset, existing_len)) = map.get(&checksum)
            && existing_len == length
            && (align <= 1 || existing_offset % align == 0)
        {
            return existing_offset;
        }
        pad_to_alignment(arena, align);
        let offset = arena.len() as u64;
        arena.extend_from_slice(bytes);
        map.insert(checksum, (offset, length));
        offset
    } else {
        pad_to_alignment(arena, align);
        let offset = arena.len() as u64;
        arena.extend_from_slice(bytes);
        offset
    }
}

fn pad_to_file_offset(bytes: &mut Vec<u8>, target: u64) -> Result<()> {
    let target = target as usize;
    if (bytes.len()) > target {
        return Err(RsmfError::structural(format!(
            "writer cursor {} exceeds target offset {}",
            bytes.len(),
            target
        )));
    }
    if bytes.len() < target {
        bytes.resize(target, 0);
    }
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let path_buf: PathBuf = path.to_path_buf();
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    let mut tmp = tempfile_in(&parent).map_err(|e| RsmfError::IoWithPath {
        path: path_buf.clone(),
        source: e,
    })?;
    tmp.file
        .write_all(bytes)
        .map_err(|e| RsmfError::IoWithPath {
            path: path_buf.clone(),
            source: e,
        })?;
    tmp.file.flush().map_err(|e| RsmfError::IoWithPath {
        path: path_buf.clone(),
        source: e,
    })?;
    let tmp_path = tmp.path.clone();
    drop(tmp.file);
    fs::rename(&tmp_path, &path_buf).map_err(|e| RsmfError::IoWithPath {
        path: path_buf,
        source: e,
    })?;
    Ok(())
}

struct Tmp {
    file: fs::File,
    path: PathBuf,
}

fn tempfile_in(dir: &Path) -> std::io::Result<Tmp> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let pid = std::process::id();
    let path = dir.join(format!(".rsmf-tmp-{pid}-{nanos}"));
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    Ok(Tmp { file, path })
}
