//! Logical and storage dtypes.

use crate::error::{Result, RsmfError};

/// All logical element types supported by RSMF v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum LogicalDtype {
    /// 8-bit unsigned integer.
    U8 = 1,
    /// 8-bit signed integer.
    I8 = 2,
    /// 16-bit unsigned integer.
    U16 = 3,
    /// 16-bit signed integer.
    I16 = 4,
    /// 32-bit unsigned integer.
    U32 = 5,
    /// 32-bit signed integer.
    I32 = 6,
    /// 64-bit unsigned integer.
    U64 = 7,
    /// 64-bit signed integer.
    I64 = 8,
    /// IEEE-754 half-precision float.
    F16 = 9,
    /// Brain floating point (16-bit truncated f32).
    BF16 = 10,
    /// 32-bit IEEE float.
    F32 = 11,
    /// 64-bit IEEE float.
    F64 = 12,
    /// Boolean stored as 1 byte (0 = false).
    Bool = 13,
}

impl LogicalDtype {
    /// Decode from the on-disk discriminant.
    pub fn from_raw(raw: u16) -> Result<Self> {
        Ok(match raw {
            1 => Self::U8,
            2 => Self::I8,
            3 => Self::U16,
            4 => Self::I16,
            5 => Self::U32,
            6 => Self::I32,
            7 => Self::U64,
            8 => Self::I64,
            9 => Self::F16,
            10 => Self::BF16,
            11 => Self::F32,
            12 => Self::F64,
            13 => Self::Bool,
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown logical dtype {other}"
                )));
            }
        })
    }

    /// Return the element size in bytes.
    #[must_use]
    pub fn size_bytes(self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Return the natural alignment (in bytes) required for this dtype.
    #[must_use]
    pub fn alignment(self) -> usize {
        self.size_bytes()
    }

    /// Human-readable name for diagnostics.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Bool => "bool",
        }
    }
}

/// On-disk storage dtypes.
///
/// For RSMF v1, this includes all [`LogicalDtype`] plus specialized
/// block-quantized formats.
///
/// Discriminants 1-13 are reserved for [`LogicalDtype`]. Discriminants
/// 100+ are block / specialty formats; new values are appended at
/// increasing indices. Old readers reject unknown discriminants with a
/// structural error (non-silent), so new entries are format-compatible
/// per the v1 evolution rules.
///
/// Not all reserved discriminants have a full decoder yet — unimplemented
/// ones surface as `RsmfError::Unsupported` from
/// [`crate::TensorView::decode_f32`]. Writers that emit them are
/// documenting the format's forward direction; decoders can land later
/// without a format-level change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum StorageDtype {
    /// Matches the logical dtype (1-13).
    Logical(LogicalDtype),
    /// 4-bit quantization (Q4_0). Stored as blocks of [f16 scale][16 bytes = 32 elements].
    Q4_0 = 100,
    /// 8-bit block quantization (Q8_0). Stored as blocks of [f16 scale][32 bytes = 32 elements].
    Q8_0 = 101,
    /// 3-bit K-quant (Q3K). Stored as super-blocks of 256 elements.
    Q3K = 102,
    /// NormalFloat 4-bit (NF4). Optimal for normally distributed weights.
    /// Stored as blocks of [f16 scale][16 bytes = 32 elements].
    NF4 = 103,
    /// 8-bit float (E4M3). Standard OCP FP8 format.
    Fp8E4M3 = 104,
    /// 4-bit K-quant (Q4K). llama.cpp super-block-with-sub-blocks
    /// format — 4 bits per weight at ~4.5 bpw including scales.
    /// Discriminant reserved; decoder not yet implemented.
    Q4K = 105,
    /// 5-bit linear block quant (Q5_0). llama.cpp legacy variant.
    /// Discriminant reserved; decoder not yet implemented.
    Q5_0 = 106,
    /// 5-bit K-quant (Q5K). ~5.5 bpw.
    /// Discriminant reserved; decoder not yet implemented.
    Q5K = 107,
    /// 6-bit K-quant (Q6K). ~6.5 bpw, near-lossless.
    /// Discriminant reserved; decoder not yet implemented.
    Q6K = 108,
    /// 2-bit K-quant (Q2K). ~2.6 bpw, extreme compression.
    /// Discriminant reserved; decoder not yet implemented.
    Q2K = 109,
    /// 8-bit float (E5M2). The other standard OCP FP8 format; used
    /// alongside Fp8E4M3 on H100/Blackwell inference for the layers
    /// that need larger dynamic range.
    /// Discriminant reserved; decoder not yet implemented.
    Fp8E5M2 = 110,
}

impl StorageDtype {
    /// Decode from the on-disk discriminant.
    pub fn from_raw(raw: u16) -> Result<Self> {
        if (1..=13).contains(&raw) {
            return Ok(Self::Logical(LogicalDtype::from_raw(raw)?));
        }
        Ok(match raw {
            100 => Self::Q4_0,
            101 => Self::Q8_0,
            102 => Self::Q3K,
            103 => Self::NF4,
            104 => Self::Fp8E4M3,
            105 => Self::Q4K,
            106 => Self::Q5_0,
            107 => Self::Q5K,
            108 => Self::Q6K,
            109 => Self::Q2K,
            110 => Self::Fp8E5M2,
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown storage dtype {other}"
                )));
            }
        })
    }

    /// Return the raw discriminant for encoding.
    #[must_use]
    pub fn to_raw(self) -> u16 {
        match self {
            Self::Logical(dt) => dt as u16,
            Self::Q4_0 => 100,
            Self::Q8_0 => 101,
            Self::Q3K => 102,
            Self::NF4 => 103,
            Self::Fp8E4M3 => 104,
            Self::Q4K => 105,
            Self::Q5_0 => 106,
            Self::Q5K => 107,
            Self::Q6K => 108,
            Self::Q2K => 109,
            Self::Fp8E5M2 => 110,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Logical(dt) => dt.name(),
            Self::Q4_0 => "q4_0",
            Self::Q8_0 => "q8_0",
            Self::Q3K => "q3_k",
            Self::NF4 => "nf4",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Q4K => "q4_k",
            Self::Q5_0 => "q5_0",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Q2K => "q2_k",
            Self::Fp8E5M2 => "fp8_e5m2",
        }
    }

    /// Return the element size in bytes (for non-blocked layouts).
    /// For blocked layouts, returns `None`.
    #[must_use]
    pub fn size_bytes(self) -> Option<usize> {
        match self {
            Self::Logical(dt) => Some(dt.size_bytes()),
            Self::Fp8E4M3 | Self::Fp8E5M2 => Some(1),
            Self::Q4_0
            | Self::Q8_0
            | Self::Q3K
            | Self::NF4
            | Self::Q4K
            | Self::Q5_0
            | Self::Q5K
            | Self::Q6K
            | Self::Q2K => None, // Blocked
        }
    }
}

/// Sentinel value indicating an unset optional dtype field. Used for
/// [`crate::tensor::variant::VariantMeta`] `scale_dtype` / `zero_point_dtype`.
pub const DTYPE_NONE: u16 = 0xFFFF;
