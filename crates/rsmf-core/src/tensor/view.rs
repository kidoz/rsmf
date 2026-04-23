//! Typed view over a tensor's canonical bytes.
//!
//! On the CPU, views are *mmap-backed* — `bytes` is a borrowed slice of the
//! file mapping and no extra allocation happens. Callers that need owned
//! numeric data can call [`TensorView::to_vec`]; when bytes are already aligned
//! for the target type, [`TensorView::as_slice`] returns a zero-copy reference.

use bytemuck::Pod;
use half::f16;
use wide::f32x8;

#[cfg(feature = "tracing")]
use tracing::info_span;

use crate::error::{Result, RsmfError};
use crate::tensor::descriptor::TensorDescriptor;
use crate::tensor::dtype::{LogicalDtype, StorageDtype};
use crate::tensor::variant::EncodingKind;

/// NF4 levels (from QLoRA paper).
const NF4_LEVELS: [f32; 16] = [
    -1.0000, -0.6944, -0.5122, -0.3739, -0.2561, -0.1496, -0.0497, 0.0000, 0.0624, 0.1236, 0.2200,
    0.3319, 0.4712, 0.6565, 0.7906, 1.0000,
];

/// A borrowed, mmap-backed view over a tensor variant's bytes.
#[derive(Debug)]
pub struct TensorView<'a> {
    /// Tensor descriptor this view was derived from.
    pub descriptor: &'a TensorDescriptor,
    /// Raw bytes of the variant, mmap-backed on CPU.
    pub bytes: &'a [u8],
    /// Encoding kind of the variant (Raw, CastF16, …).
    pub encoding: EncodingKind,
    /// Physical layout of the variant bytes (row-major vs blocked).
    /// Consumers that want zero-copy access to `bytes` must check this
    /// before re-interpreting the payload as a C-contiguous array.
    pub layout: crate::tensor::variant::LayoutTag,
    /// Variant metadata (block shapes, quant params).
    pub meta: &'a crate::tensor::variant::VariantMeta,
    /// Storage dtype.
    pub storage_dtype: StorageDtype,
}

impl<'a> TensorView<'a> {
    /// Logical element dtype.
    #[must_use]
    pub fn dtype(&self) -> LogicalDtype {
        self.descriptor.dtype
    }

    /// Logical shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.descriptor.shape
    }

    /// Borrowed byte slice (mmap-backed on CPU).
    #[must_use]
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Try to borrow the bytes as a typed slice.
    pub fn as_slice<T: Pod>(&self) -> Result<&'a [T]> {
        if std::mem::size_of::<T>() != self.descriptor.dtype.size_bytes() {
            return Err(RsmfError::unsupported(format!(
                "dtype {:?} is {}B but requested element is {}B",
                self.descriptor.dtype,
                self.descriptor.dtype.size_bytes(),
                std::mem::size_of::<T>()
            )));
        }
        bytemuck::try_cast_slice::<u8, T>(self.bytes).map_err(|e| {
            RsmfError::unsupported(format!(
                "tensor bytes not aligned for {}B element: {e:?}; use to_vec()",
                std::mem::size_of::<T>()
            ))
        })
    }

    /// Decode the variant's bytes into `Vec<f32>`, handling encoding
    /// transparently. Optimized with SIMD where possible.
    pub fn decode_f32(&self) -> Result<Vec<f32>> {
        // OPTIMIZE: Falling back to raw core::arch::x86_64 (AVX-512) or ARM NEON/SVE 
        // intrinsics for heavy dequantization loops (e.g., INT8 to F32) can yield 
        // significant speedups over generic f32x8 wide vectors. This aligns with 
        // the CpuAvx512 and CpuNeon tags already present in the Manifest.
        #[cfg(feature = "tracing")]
        let _span = info_span!(
            "TensorView::decode_f32",
            name = %self.descriptor.name,
            encoding = ?self.encoding,
            storage = ?self.storage_dtype
        )
        .entered();

        match self.encoding {
            EncodingKind::Raw => {
                if self.storage_dtype == StorageDtype::Fp8E4M3 {
                    // OCP FP8 E4M3: 1 sign, 4 exp, 3 mantissa, bias 7.
                    // No infinity. NaN is the S.1111.111 sentinel (0x7F / 0xFF).
                    // Max finite = S.1111.110 = ±448.
                    let mut out = Vec::with_capacity(self.bytes.len());
                    for &b in self.bytes {
                        if b & 0x7F == 0x7F {
                            out.push(f32::NAN);
                            continue;
                        }
                        let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
                        let exp = ((b >> 3) & 0x0F) as i32;
                        let mant = (b & 0x07) as f32;

                        if exp == 0 {
                            out.push(sign * 2.0f32.powi(-6) * (mant / 8.0));
                        } else {
                            out.push(sign * 2.0f32.powi(exp - 7) * (1.0 + mant / 8.0));
                        }
                    }
                    return Ok(out);
                }

                if self.storage_dtype == StorageDtype::Fp8E5M2 {
                    // OCP FP8 E5M2: 1 sign, 5 exp, 2 mantissa, bias 15.
                    // IEEE-754-shaped: subnormals at exp==0, inf at
                    // exp==31 mant==0, NaN at exp==31 mant!=0.
                    // Max finite = 2^(30-15) * (1 + 3/4) = 57344.
                    let mut out = Vec::with_capacity(self.bytes.len());
                    for &b in self.bytes {
                        let sign_bit = b & 0x80 != 0;
                        let exp = ((b >> 2) & 0x1F) as i32;
                        let mant = (b & 0x03) as f32;
                        let sign_mul = if sign_bit { -1.0_f32 } else { 1.0 };
                        let v = if exp == 0 {
                            if mant == 0.0 {
                                // ±0
                                sign_mul * 0.0
                            } else {
                                // Subnormal: sign * 2^-14 * (m/4)
                                sign_mul * 2.0_f32.powi(-14) * (mant / 4.0)
                            }
                        } else if exp == 31 {
                            if mant == 0.0 {
                                // ±infinity
                                if sign_bit {
                                    f32::NEG_INFINITY
                                } else {
                                    f32::INFINITY
                                }
                            } else {
                                // NaN — sign of NaN isn't meaningful
                                f32::NAN
                            }
                        } else {
                            // Normal: sign * 2^(exp-15) * (1 + m/4)
                            sign_mul * 2.0_f32.powi(exp - 15) * (1.0 + mant / 4.0)
                        };
                        out.push(v);
                    }
                    return Ok(out);
                }

                if self.descriptor.dtype != LogicalDtype::F32 {
                    return Err(RsmfError::unsupported(format!(
                        "decode_f32 on Raw variant requires logical dtype F32, got {:?}",
                        self.descriptor.dtype
                    )));
                }
                self.to_vec::<f32>()
            }
            EncodingKind::CastF16 => {
                if self.bytes.len() % 2 != 0 {
                    return Err(RsmfError::structural("CastF16 length not even".to_string()));
                }
                let count = self.bytes.len() / 2;
                let mut out = Vec::with_capacity(count);
                // Scalar fallback for f16 for now
                for i in 0..count {
                    let h = f16::from_le_bytes([self.bytes[i * 2], self.bytes[i * 2 + 1]]);
                    out.push(h.to_f32());
                }
                Ok(out)
            }
            EncodingKind::BlockQuantized => {
                if self.descriptor.dtype != LogicalDtype::F32 {
                    return Err(RsmfError::unsupported(format!(
                        "decode_f32 requires logical F32, got {:?}",
                        self.descriptor.dtype
                    )));
                }

                match self.storage_dtype {
                    StorageDtype::Q8_0 | StorageDtype::Logical(LogicalDtype::I8) => {
                        let block_size =
                            self.meta.block_shape.first().copied().unwrap_or(32) as usize;
                        let block_bytes = 2 + block_size;
                        let num_blocks = self.bytes.len() / block_bytes;
                        let mut out = Vec::with_capacity(num_blocks * block_size);

                        for b in 0..num_blocks {
                            let off = b * block_bytes;
                            let scale =
                                f16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]).to_f32();
                            let v_scale = f32x8::splat(scale);

                            let i8_data = &self.bytes[off + 2..off + 2 + block_size];

                            let mut i = 0;
                            while i + 8 <= block_size {
                                let mut floats = [0.0f32; 8];
                                for j in 0..8 {
                                    floats[j] = i8_data[i + j] as i8 as f32;
                                }
                                let v_floats = f32x8::from(floats);
                                let dequant = v_floats * v_scale;
                                out.extend_from_slice(&dequant.to_array());
                                i += 8;
                            }
                            while i < block_size {
                                out.push(i8_data[i] as i8 as f32 * scale);
                                i += 1;
                            }
                        }
                        Ok(out)
                    }
                    StorageDtype::Q4_0 => {
                        let block_size = 32;
                        let block_bytes = 18; // 2 scale + 16 weights
                        let num_blocks = self.bytes.len() / block_bytes;
                        let mut out = Vec::with_capacity(num_blocks * block_size);

                        for b in 0..num_blocks {
                            let off = b * block_bytes;
                            let scale =
                                f16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]).to_f32();

                            let q_data = &self.bytes[off + 2..off + 18];
                            let mut block_out = [0.0f32; 32];
                            for (i, &p) in q_data.iter().enumerate().take(16) {
                                block_out[i] = ((p & 0x0F) as i8 - 8) as f32 * scale;
                                block_out[i + 16] = ((p >> 4) as i8 - 8) as f32 * scale;
                            }
                            out.extend_from_slice(&block_out);
                        }
                        Ok(out)
                    }
                    StorageDtype::Q3K => {
                        let block_size = 256;
                        let block_bytes = 110;
                        let num_blocks = self.bytes.len() / block_bytes;
                        let mut out = Vec::with_capacity(num_blocks * block_size);

                        for b in 0..num_blocks {
                            let off = b * block_bytes;
                            let super_scale =
                                f16::from_le_bytes([self.bytes[off + 108], self.bytes[off + 109]])
                                    .to_f32();

                            let hmask = &self.bytes[off..off + 32];
                            let qs = &self.bytes[off + 32..off + 96];

                            for i in 0..256 {
                                let low = (qs[i / 4] >> ((i % 4) * 2)) & 0x3;
                                let high = if (hmask[i / 8] & (1 << (i % 8))) != 0 {
                                    4
                                } else {
                                    0
                                };
                                let q = (low | high) as i8 - 4;
                                out.push(q as f32 * super_scale);
                            }
                        }
                        Ok(out)
                    }
                    StorageDtype::NF4 => {
                        let block_size = 32;
                        let block_bytes = 18;
                        let num_blocks = self.bytes.len() / block_bytes;
                        let mut out = Vec::with_capacity(num_blocks * block_size);

                        for b in 0..num_blocks {
                            let off = b * block_bytes;
                            let scale =
                                f16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]).to_f32();
                            let q_data = &self.bytes[off + 2..off + 18];

                            for &p in q_data.iter().take(16) {
                                out.push(NF4_LEVELS[(p & 0x0F) as usize] * scale);
                                out.push(NF4_LEVELS[(p >> 4) as usize] * scale);
                            }
                        }
                        Ok(out)
                    }
                    StorageDtype::Q4K => crate::tensor::dequantize::dequantize_q4_k(self.bytes),
                    StorageDtype::Q5K => crate::tensor::dequantize::dequantize_q5_k(self.bytes),
                    StorageDtype::Q6K => crate::tensor::dequantize::dequantize_q6_k(self.bytes),
                    StorageDtype::Q2K => crate::tensor::dequantize::dequantize_q2_k(self.bytes),
                    StorageDtype::Q5_0 => crate::tensor::dequantize::dequantize_q5_0(self.bytes),
                    _ => Err(RsmfError::unsupported(
                        "Storage dtype not supported".to_string(),
                    )),
                }
            }
        }
    }

    /// Copy the tensor bytes into an owned `Vec<T>`.
    pub fn to_vec<T: Pod>(&self) -> Result<Vec<T>> {
        if std::mem::size_of::<T>() != self.descriptor.dtype.size_bytes() {
            return Err(RsmfError::unsupported(format!(
                "dtype {:?} is {}B but requested element is {}B",
                self.descriptor.dtype,
                self.descriptor.dtype.size_bytes(),
                std::mem::size_of::<T>()
            )));
        }
        if self.bytes.len() % std::mem::size_of::<T>() != 0 {
            return Err(RsmfError::structural(format!(
                "tensor byte length {} is not a multiple of {}B element size",
                self.bytes.len(),
                std::mem::size_of::<T>()
            )));
        }
        bytemuck::try_cast_slice::<u8, T>(self.bytes)
            .map(|s| s.to_vec())
            .map_err(|e| RsmfError::unsupported(format!("alignment: {e:?}")))
    }
}
