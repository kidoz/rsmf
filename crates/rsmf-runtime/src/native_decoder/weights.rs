use super::*;
use half::f16;
use rsmf_core::{EncodingKind, StorageDtype};

/// Owned quantized native decoder matrix used by direct CPU kernels.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeDecoderQuantizedMatrix {
    /// Raw row-major I8 matrix with implicit scale 1.0.
    RawI8 {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Row-major signed values.
        values: Vec<i8>,
    },
    /// RSMF/GGUF-style Q8_0 blocks stored as `[f16 scale][i8; block_size]`.
    Q8_0 {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Quantization block size, usually 32.
        block_size: usize,
        /// Owned block bytes.
        bytes: Vec<u8>,
    },
    /// RSMF/GGUF-style Q4_0 blocks stored as `[f16 scale][u4; 32]`.
    Q4_0 {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Owned block bytes.
        bytes: Vec<u8>,
    },
    /// RSMF/GGUF-style Q3_K super-blocks.
    Q3K {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Owned block bytes.
        bytes: Vec<u8>,
    },
    /// RSMF/GGUF-style Q4_K super-blocks.
    Q4K {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Owned block bytes.
        bytes: Vec<u8>,
    },
    /// RSMF/GGUF-style Q5_K super-blocks.
    Q5K {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Owned block bytes.
        bytes: Vec<u8>,
    },
    /// RSMF/GGUF-style Q6_K super-blocks.
    Q6K {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Owned block bytes.
        bytes: Vec<u8>,
    },
}

impl NativeDecoderQuantizedMatrix {
    /// Resident bytes held by this quantized matrix.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::RawI8 { values, .. } => values.len(),
            Self::Q8_0 { bytes, .. }
            | Self::Q4_0 { bytes, .. }
            | Self::Q3K { bytes, .. }
            | Self::Q4K { bytes, .. }
            | Self::Q5K { bytes, .. }
            | Self::Q6K { bytes, .. } => bytes.len(),
        }
    }

    /// Direct matrix-vector multiply into f32 without materializing the whole
    /// matrix as f32 first.
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>> {
        match self {
            Self::RawI8 { rows, cols, values } => {
                validate_cpu_vector_len("quantized_matvec_i8", "input", input.len(), *cols)?;
                validate_cpu_matrix_len(
                    "quantized_matvec_i8",
                    "weight",
                    values.len(),
                    *rows,
                    *cols,
                )?;
                let mut output = vec![0.0f32; *rows];
                for (row, output_value) in output.iter_mut().enumerate().take(*rows) {
                    let mut sum = 0.0f32;
                    let row_start = row * *cols;
                    for (col, input_value) in input.iter().enumerate().take(*cols) {
                        sum += *input_value * values[row_start + col] as f32;
                    }
                    *output_value = sum;
                }
                Ok(output)
            }
            Self::Q8_0 {
                rows,
                cols,
                block_size,
                bytes,
            } => {
                validate_cpu_vector_len("quantized_matvec_q8_0", "input", input.len(), *cols)?;
                validate_cpu_positive("quantized_matvec_q8_0", "block_size", *block_size)?;
                let element_count =
                    cpu_element_count("quantized_matvec_q8_0", "weight", *rows, *cols)?;
                let block_bytes = block_size.checked_add(2).ok_or_else(|| {
                    native_decoder_cpu_shape_error(
                        "quantized_matvec_q8_0",
                        "block byte count overflow",
                    )
                })?;
                let expected_blocks = element_count.div_ceil(*block_size);
                validate_cpu_vector_len(
                    "quantized_matvec_q8_0",
                    "bytes",
                    bytes.len(),
                    expected_blocks.checked_mul(block_bytes).ok_or_else(|| {
                        native_decoder_cpu_shape_error(
                            "quantized_matvec_q8_0",
                            "byte count overflow",
                        )
                    })?,
                )?;
                let mut output = vec![0.0f32; *rows];
                for (row, output_value) in output.iter_mut().enumerate().take(*rows) {
                    let mut sum = 0.0f32;
                    for (col, input_value) in input.iter().enumerate().take(*cols) {
                        let index = row * *cols + col;
                        let block_index = index / *block_size;
                        let block_offset = index % *block_size;
                        let byte_offset = block_index * block_bytes;
                        let scale =
                            f16::from_le_bytes([bytes[byte_offset], bytes[byte_offset + 1]])
                                .to_f32();
                        let value = bytes[byte_offset + 2 + block_offset] as i8 as f32 * scale;
                        sum += *input_value * value;
                    }
                    *output_value = sum;
                }
                Ok(output)
            }
            Self::Q4_0 { rows, cols, bytes } => {
                validate_cpu_vector_len("quantized_matvec_q4_0", "input", input.len(), *cols)?;
                let element_count =
                    cpu_element_count("quantized_matvec_q4_0", "weight", *rows, *cols)?;
                let block_size = 32usize;
                let block_bytes = 18usize;
                let expected_blocks = element_count.div_ceil(block_size);
                validate_cpu_vector_len(
                    "quantized_matvec_q4_0",
                    "bytes",
                    bytes.len(),
                    expected_blocks.checked_mul(block_bytes).ok_or_else(|| {
                        native_decoder_cpu_shape_error(
                            "quantized_matvec_q4_0",
                            "byte count overflow",
                        )
                    })?,
                )?;
                let mut output = vec![0.0f32; *rows];
                for (row, output_value) in output.iter_mut().enumerate().take(*rows) {
                    let mut sum = 0.0f32;
                    for (col, input_value) in input.iter().enumerate().take(*cols) {
                        let index = row * *cols + col;
                        let block_index = index / block_size;
                        let block_offset = index % block_size;
                        let byte_offset = block_index * block_bytes;
                        let scale =
                            f16::from_le_bytes([bytes[byte_offset], bytes[byte_offset + 1]])
                                .to_f32();
                        let packed = if block_offset < 16 {
                            bytes[byte_offset + 2 + block_offset]
                        } else {
                            bytes[byte_offset + 2 + block_offset - 16] >> 4
                        };
                        let quantized = if block_offset < 16 {
                            packed & 0x0f
                        } else {
                            packed
                        };
                        let value = (quantized as i8 - 8) as f32 * scale;
                        sum += *input_value * value;
                    }
                    *output_value = sum;
                }
                Ok(output)
            }
            Self::Q3K { rows, cols, bytes } => quantized_k_matvec(
                input,
                bytes,
                qk_matvec_spec(
                    "quantized_matvec_q3_k",
                    *rows,
                    *cols,
                    Q3_K_BLOCK_BYTES,
                    q3_k_value,
                ),
            ),
            Self::Q4K { rows, cols, bytes } => quantized_k_matvec(
                input,
                bytes,
                qk_matvec_spec(
                    "quantized_matvec_q4_k",
                    *rows,
                    *cols,
                    Q4_K_BLOCK_BYTES,
                    q4_k_value,
                ),
            ),
            Self::Q5K { rows, cols, bytes } => quantized_k_matvec(
                input,
                bytes,
                qk_matvec_spec(
                    "quantized_matvec_q5_k",
                    *rows,
                    *cols,
                    Q5_K_BLOCK_BYTES,
                    q5_k_value,
                ),
            ),
            Self::Q6K { rows, cols, bytes } => quantized_k_matvec(
                input,
                bytes,
                qk_matvec_spec(
                    "quantized_matvec_q6_k",
                    *rows,
                    *cols,
                    Q6_K_BLOCK_BYTES,
                    q6_k_value,
                ),
            ),
        }
    }
}

#[derive(Clone, Copy)]
struct QkMatvecSpec {
    op: &'static str,
    rows: usize,
    cols: usize,
    block_elements: usize,
    block_bytes: usize,
    decode: fn(&[u8], usize) -> f32,
}

fn qk_matvec_spec(
    op: &'static str,
    rows: usize,
    cols: usize,
    block_bytes: usize,
    decode: fn(&[u8], usize) -> f32,
) -> QkMatvecSpec {
    QkMatvecSpec {
        op,
        rows,
        cols,
        block_elements: QK_SUPER_BLOCK_ELEMENTS,
        block_bytes,
        decode,
    }
}

fn quantized_k_matvec(input: &[f32], bytes: &[u8], spec: QkMatvecSpec) -> Result<Vec<f32>> {
    validate_cpu_vector_len(spec.op, "input", input.len(), spec.cols)?;
    let element_count = cpu_element_count(spec.op, "weight", spec.rows, spec.cols)?;
    let expected_blocks = element_count.div_ceil(spec.block_elements);
    validate_cpu_vector_len(
        spec.op,
        "bytes",
        bytes.len(),
        expected_blocks
            .checked_mul(spec.block_bytes)
            .ok_or_else(|| native_decoder_cpu_shape_error(spec.op, "byte count overflow"))?,
    )?;
    let mut output = vec![0.0f32; spec.rows];
    for (row, output_value) in output.iter_mut().enumerate().take(spec.rows) {
        let mut sum = 0.0f32;
        for (col, input_value) in input.iter().enumerate().take(spec.cols) {
            let index = row * spec.cols + col;
            let block_index = index / spec.block_elements;
            let block_offset = index % spec.block_elements;
            let byte_offset = block_index * spec.block_bytes;
            sum += *input_value
                * (spec.decode)(
                    &bytes[byte_offset..byte_offset + spec.block_bytes],
                    block_offset,
                );
        }
        *output_value = sum;
    }
    Ok(output)
}

fn q3_k_value(block: &[u8], offset: usize) -> f32 {
    let super_scale = f16::from_le_bytes([block[108], block[109]]).to_f32();
    let hmask = &block[..32];
    let qs = &block[32..96];
    let low = (qs[offset / 4] >> ((offset % 4) * 2)) & 0x03;
    let high = if (hmask[offset / 8] & (1 << (offset % 8))) != 0 {
        4
    } else {
        0
    };
    ((low | high) as i8 - 4) as f32 * super_scale
}

fn q4_k_value(block: &[u8], offset: usize) -> f32 {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let min = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];
    let group = offset / 64;
    let within = offset % 64;
    let qs_off = group * 32;
    if within < 32 {
        let (scale, min_scale) = get_scale_min_k4(group * 2, scales);
        d * scale as f32 * (qs[qs_off + within] & 0x0f) as f32 - min * min_scale as f32
    } else {
        let (scale, min_scale) = get_scale_min_k4(group * 2 + 1, scales);
        d * scale as f32 * (qs[qs_off + within - 32] >> 4) as f32 - min * min_scale as f32
    }
}

fn q5_k_value(block: &[u8], offset: usize) -> f32 {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let min = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];
    let group = offset / 64;
    let within = offset % 64;
    let qs_off = group * 32;
    let u1 = 1u8 << (group * 2);
    let u2 = 2u8 << (group * 2);
    if within < 32 {
        let (scale, min_scale) = get_scale_min_k4(group * 2, scales);
        let high_bit = if (qh[within] & u1) != 0 { 16.0 } else { 0.0 };
        d * scale as f32 * ((qs[qs_off + within] & 0x0f) as f32 + high_bit) - min * min_scale as f32
    } else {
        let lane = within - 32;
        let (scale, min_scale) = get_scale_min_k4(group * 2 + 1, scales);
        let high_bit = if (qh[lane] & u2) != 0 { 16.0 } else { 0.0 };
        d * scale as f32 * ((qs[qs_off + lane] >> 4) as f32 + high_bit) - min * min_scale as f32
    }
}

fn q6_k_value(block: &[u8], offset: usize) -> f32 {
    let ql = &block[..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
    let group = offset / 128;
    let within = offset % 128;
    let ql_off = group * 64;
    let qh_off = group * 32;
    let sc_off = group * 8;
    let (q, scale_index) = if within < 32 {
        let lane = within;
        (
            ((ql[ql_off + lane] & 0x0f) | ((qh[qh_off + lane] & 0x03) << 4)) as i8 - 32,
            lane / 16,
        )
    } else if within < 64 {
        let lane = within - 32;
        (
            ((ql[ql_off + lane + 32] & 0x0f) | (((qh[qh_off + lane] >> 2) & 0x03) << 4)) as i8 - 32,
            lane / 16 + 2,
        )
    } else if within < 96 {
        let lane = within - 64;
        (
            ((ql[ql_off + lane] >> 4) | (((qh[qh_off + lane] >> 4) & 0x03) << 4)) as i8 - 32,
            lane / 16 + 4,
        )
    } else {
        let lane = within - 96;
        (
            ((ql[ql_off + lane + 32] >> 4) | (((qh[qh_off + lane] >> 6) & 0x03) << 4)) as i8 - 32,
            lane / 16 + 6,
        )
    };
    d * sc[sc_off + scale_index] as i8 as f32 * q as f32
}

fn get_scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        let d = scales[index] & 63;
        let m = scales[index + 4] & 63;
        (d, m)
    } else {
        let d = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        let m = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
        (d, m)
    }
}

const QK_SUPER_BLOCK_ELEMENTS: usize = 256;
const Q3_K_BLOCK_BYTES: usize = 110;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;

/// Owned LLaMA-style layer weights decoded for native decoder execution.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderLayerWeights {
    /// Input RMSNorm weight, shape `[hidden_size]`.
    pub input_layernorm: Vec<f32>,
    /// Post-attention RMSNorm weight, shape `[hidden_size]`.
    pub post_attention_layernorm: Vec<f32>,
    /// Query projection weight, row-major shape `[hidden_size, hidden_size]`.
    pub q_proj: Vec<f32>,
    /// Optional selected quantized query projection matrix.
    pub q_proj_quantized: Option<NativeDecoderQuantizedMatrix>,
    /// Key projection weight, row-major shape `[kv_width, hidden_size]`.
    pub k_proj: Vec<f32>,
    /// Optional selected quantized key projection matrix.
    pub k_proj_quantized: Option<NativeDecoderQuantizedMatrix>,
    /// Value projection weight, row-major shape `[kv_width, hidden_size]`.
    pub v_proj: Vec<f32>,
    /// Optional selected quantized value projection matrix.
    pub v_proj_quantized: Option<NativeDecoderQuantizedMatrix>,
    /// Attention output projection weight, row-major shape `[hidden_size, hidden_size]`.
    pub o_proj: Vec<f32>,
    /// Optional selected quantized attention output projection matrix.
    pub o_proj_quantized: Option<NativeDecoderQuantizedMatrix>,
    /// SwiGLU gate projection weight, row-major shape `[intermediate_size, hidden_size]`.
    pub gate_proj: Vec<f32>,
    /// Optional selected quantized gate projection matrix.
    pub gate_proj_quantized: Option<NativeDecoderQuantizedMatrix>,
    /// SwiGLU up projection weight, row-major shape `[intermediate_size, hidden_size]`.
    pub up_proj: Vec<f32>,
    /// Optional selected quantized up projection matrix.
    pub up_proj_quantized: Option<NativeDecoderQuantizedMatrix>,
    /// SwiGLU down projection weight, row-major shape `[hidden_size, intermediate_size]`.
    pub down_proj: Vec<f32>,
    /// Optional selected quantized down projection matrix.
    pub down_proj_quantized: Option<NativeDecoderQuantizedMatrix>,
}

impl NativeDecoderLayerWeights {
    pub(crate) fn as_cpu(&self) -> NativeDecoderCpuLayerWeights<'_> {
        NativeDecoderCpuLayerWeights {
            input_layernorm: &self.input_layernorm,
            post_attention_layernorm: &self.post_attention_layernorm,
            q_proj: &self.q_proj,
            q_proj_quantized: self.q_proj_quantized.as_ref(),
            k_proj: &self.k_proj,
            k_proj_quantized: self.k_proj_quantized.as_ref(),
            v_proj: &self.v_proj,
            v_proj_quantized: self.v_proj_quantized.as_ref(),
            o_proj: &self.o_proj,
            o_proj_quantized: self.o_proj_quantized.as_ref(),
            gate_proj: &self.gate_proj,
            gate_proj_quantized: self.gate_proj_quantized.as_ref(),
            up_proj: &self.up_proj,
            up_proj_quantized: self.up_proj_quantized.as_ref(),
            down_proj: &self.down_proj,
            down_proj_quantized: self.down_proj_quantized.as_ref(),
        }
    }

    /// Resident decoded weight bytes held by this layer.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        f32_slice_bytes(&self.input_layernorm)
            .saturating_add(f32_slice_bytes(&self.post_attention_layernorm))
            .saturating_add(f32_slice_bytes(&self.q_proj))
            .saturating_add(f32_slice_bytes(&self.k_proj))
            .saturating_add(f32_slice_bytes(&self.v_proj))
            .saturating_add(f32_slice_bytes(&self.o_proj))
            .saturating_add(f32_slice_bytes(&self.gate_proj))
            .saturating_add(f32_slice_bytes(&self.up_proj))
            .saturating_add(f32_slice_bytes(&self.down_proj))
            .saturating_add(quantized_matrix_bytes(&self.q_proj_quantized))
            .saturating_add(quantized_matrix_bytes(&self.k_proj_quantized))
            .saturating_add(quantized_matrix_bytes(&self.v_proj_quantized))
            .saturating_add(quantized_matrix_bytes(&self.o_proj_quantized))
            .saturating_add(quantized_matrix_bytes(&self.gate_proj_quantized))
            .saturating_add(quantized_matrix_bytes(&self.up_proj_quantized))
            .saturating_add(quantized_matrix_bytes(&self.down_proj_quantized))
    }
}

/// Owned native decoder weights decoded from an RSMF file.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeDecoderWeights {
    /// Parsed decoder configuration used to validate these weights.
    pub config: NativeDecoderConfig,
    /// Token embedding matrix, row-major shape `[vocab_size, hidden_size]`.
    pub token_embedding: Vec<f32>,
    /// Final RMSNorm weight, shape `[hidden_size]`.
    pub final_norm: Vec<f32>,
    /// Optional LM head matrix, row-major shape `[vocab_size, hidden_size]`.
    /// When absent, token embeddings are used as tied output embeddings.
    pub lm_head: Option<Vec<f32>>,
    /// Optional selected quantized LM-head matrix used by direct quantized
    /// kernels when the selected RSMF variant supports it.
    pub lm_head_quantized: Option<NativeDecoderQuantizedMatrix>,
    /// Per-layer decoded weights.
    pub layers: Vec<NativeDecoderLayerWeights>,
}

impl NativeDecoderWeights {
    /// Resident decoded weight bytes held by this native decoder weight set.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let base = f32_slice_bytes(&self.token_embedding)
            .saturating_add(f32_slice_bytes(&self.final_norm))
            .saturating_add(
                self.lm_head
                    .as_ref()
                    .map_or(0, |weights| f32_slice_bytes(weights)),
            )
            .saturating_add(
                self.lm_head_quantized
                    .as_ref()
                    .map_or(0, NativeDecoderQuantizedMatrix::resident_bytes),
            );
        self.layers.iter().fold(base, |total, layer| {
            total.saturating_add(layer.resident_bytes())
        })
    }
}

pub(crate) fn f32_slice_bytes(values: &[f32]) -> usize {
    values.len().saturating_mul(std::mem::size_of::<f32>())
}

fn quantized_matrix_bytes(value: &Option<NativeDecoderQuantizedMatrix>) -> usize {
    value
        .as_ref()
        .map_or(0, NativeDecoderQuantizedMatrix::resident_bytes)
}

pub(crate) fn load_native_decoder_weights(
    file: &RsmfFile,
    config: NativeDecoderConfig,
    options: &NativeDecoderWeightOptions,
) -> Result<NativeDecoderWeights> {
    if config.family != NativeDecoderFamily::Llama {
        return Err(RuntimeError::UnsupportedNativeDecoder {
            family: format!("{:?}", config.family),
        });
    }
    let hidden = usize_to_u64(config.hidden_size, "hidden_size")?;
    let intermediate = usize_to_u64(config.intermediate_size, "intermediate_size")?;
    let vocab = usize_to_u64(config.vocab_size, "vocab_size")?;
    let kv_width = usize_to_u64(config.key_value_width(), "kv projection width")?;
    let token_embedding = load_native_decoder_tensor_f32(
        file,
        options,
        "model.embed_tokens.weight",
        &[vocab, hidden],
    )?;
    let final_norm = load_native_decoder_tensor_f32(file, options, "model.norm.weight", &[hidden])?;
    let (lm_head, lm_head_quantized) = if config.tie_word_embeddings {
        (None, None)
    } else {
        let loaded = load_native_decoder_tensor_f32_with_quantized(
            file,
            options,
            "lm_head.weight",
            &[vocab, hidden],
        )?;
        (Some(loaded.values), loaded.quantized)
    };
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let q_proj = load_native_decoder_tensor_f32_with_quantized(
            file,
            options,
            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
            &[hidden, hidden],
        )?;
        let k_proj = load_native_decoder_tensor_f32_with_quantized(
            file,
            options,
            &format!("model.layers.{layer}.self_attn.k_proj.weight"),
            &[kv_width, hidden],
        )?;
        let v_proj = load_native_decoder_tensor_f32_with_quantized(
            file,
            options,
            &format!("model.layers.{layer}.self_attn.v_proj.weight"),
            &[kv_width, hidden],
        )?;
        let o_proj = load_native_decoder_tensor_f32_with_quantized(
            file,
            options,
            &format!("model.layers.{layer}.self_attn.o_proj.weight"),
            &[hidden, hidden],
        )?;
        let gate_proj = load_native_decoder_tensor_f32_with_quantized(
            file,
            options,
            &format!("model.layers.{layer}.mlp.gate_proj.weight"),
            &[intermediate, hidden],
        )?;
        let up_proj = load_native_decoder_tensor_f32_with_quantized(
            file,
            options,
            &format!("model.layers.{layer}.mlp.up_proj.weight"),
            &[intermediate, hidden],
        )?;
        let down_proj = load_native_decoder_tensor_f32_with_quantized(
            file,
            options,
            &format!("model.layers.{layer}.mlp.down_proj.weight"),
            &[hidden, intermediate],
        )?;
        layers.push(NativeDecoderLayerWeights {
            input_layernorm: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.input_layernorm.weight"),
                &[hidden],
            )?,
            post_attention_layernorm: load_native_decoder_tensor_f32(
                file,
                options,
                &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            q_proj: q_proj.values,
            q_proj_quantized: q_proj.quantized,
            k_proj: k_proj.values,
            k_proj_quantized: k_proj.quantized,
            v_proj: v_proj.values,
            v_proj_quantized: v_proj.quantized,
            o_proj: o_proj.values,
            o_proj_quantized: o_proj.quantized,
            gate_proj: gate_proj.values,
            gate_proj_quantized: gate_proj.quantized,
            up_proj: up_proj.values,
            up_proj_quantized: up_proj.quantized,
            down_proj: down_proj.values,
            down_proj_quantized: down_proj.quantized,
        });
    }
    Ok(NativeDecoderWeights {
        config,
        token_embedding,
        final_norm,
        lm_head,
        lm_head_quantized,
        layers,
    })
}

struct NativeDecoderLoadedTensor {
    values: Vec<f32>,
    quantized: Option<NativeDecoderQuantizedMatrix>,
}

fn load_native_decoder_tensor_f32_with_quantized(
    file: &RsmfFile,
    options: &NativeDecoderWeightOptions,
    tensor_name: &str,
    expected_shape: &[u64],
) -> Result<NativeDecoderLoadedTensor> {
    let view = native_decoder_weight_view(file, options, tensor_name)?;
    let values =
        decode_native_decoder_weight_view_f32(&view, options, tensor_name, expected_shape)?;
    let quantized = NativeDecoderQuantizedMatrix::from_weight_view(view, options, expected_shape)?;
    Ok(NativeDecoderLoadedTensor { values, quantized })
}

pub(crate) fn load_native_decoder_tensor_f32(
    file: &RsmfFile,
    options: &NativeDecoderWeightOptions,
    tensor_name: &str,
    expected_shape: &[u64],
) -> Result<Vec<f32>> {
    let view = native_decoder_weight_view(file, options, tensor_name)?;
    decode_native_decoder_weight_view_f32(&view, options, tensor_name, expected_shape)
}

fn decode_native_decoder_weight_view_f32(
    view: &TensorView<'_>,
    options: &NativeDecoderWeightOptions,
    tensor_name: &str,
    expected_shape: &[u64],
) -> Result<Vec<f32>> {
    if view.shape() != expected_shape {
        return Err(RuntimeError::NativeDecoderTensorShapeMismatch {
            tensor_name: tensor_name.to_string(),
            expected_shape: format_shape(expected_shape),
            actual_shape: format_shape(view.shape()),
        });
    }
    validate_native_decoder_weight_dtype(view.descriptor)?;
    if matches!(
        view.descriptor.dtype,
        LogicalDtype::I8 | LogicalDtype::I16 | LogicalDtype::I32
    ) && !options.allow_lossy_quantized
    {
        return Err(RuntimeError::NativeDecoderTensorDtypeUnsupported {
            tensor_name: tensor_name.to_string(),
            dtype: format!("{:?}", view.descriptor.dtype),
        });
    }
    if view.encoding == rsmf_core::EncodingKind::BlockQuantized && !options.allow_lossy_quantized {
        return Err(RuntimeError::NativeDecoderTensorDtypeUnsupported {
            tensor_name: tensor_name.to_string(),
            dtype: format!("{:?}", view.storage_dtype),
        });
    }
    if view.layout != LayoutTag::RowMajor
        && view.encoding != rsmf_core::EncodingKind::BlockQuantized
    {
        return Err(RuntimeError::NativeDecoderTensorUnsupported {
            tensor_name: tensor_name.to_string(),
            reason: format!(
                "only row-major weights are supported, got {:?}",
                view.layout
            ),
        });
    }
    let mut values =
        view.decode_f32()
            .map_err(|error| RuntimeError::NativeDecoderTensorUnsupported {
                tensor_name: tensor_name.to_string(),
                reason: error.to_string(),
            })?;
    let expected_len = shape_element_count(expected_shape)?;
    if view.encoding == EncodingKind::BlockQuantized && values.len() > expected_len {
        values.truncate(expected_len);
    }
    if values.len() != expected_len {
        return Err(RuntimeError::NativeDecoderTensorUnsupported {
            tensor_name: tensor_name.to_string(),
            reason: format!("decoded {} elements, expected {expected_len}", values.len()),
        });
    }
    Ok(values)
}

impl NativeDecoderQuantizedMatrix {
    fn from_weight_view(
        view: TensorView<'_>,
        options: &NativeDecoderWeightOptions,
        expected_shape: &[u64],
    ) -> Result<Option<Self>> {
        if !options.allow_lossy_quantized {
            return Ok(None);
        }
        let [rows_u64, cols_u64] = expected_shape else {
            return Ok(None);
        };
        let rows = usize::try_from(*rows_u64)
            .map_err(|_| RuntimeError::Shape("quantized matrix rows exceed usize".to_string()))?;
        let cols = usize::try_from(*cols_u64)
            .map_err(|_| RuntimeError::Shape("quantized matrix cols exceed usize".to_string()))?;
        if view.encoding == EncodingKind::Raw && view.descriptor.dtype == LogicalDtype::I8 {
            if view.layout != LayoutTag::RowMajor {
                return Err(RuntimeError::NativeDecoderTensorUnsupported {
                    tensor_name: view.descriptor.name.clone(),
                    reason: format!(
                        "raw I8 quantized matrix requires row-major layout, got {:?}",
                        view.layout
                    ),
                });
            }
            let expected_len = rows.checked_mul(cols).ok_or_else(|| {
                RuntimeError::Shape("quantized matrix element count overflow".to_string())
            })?;
            validate_cpu_vector_len(
                "quantized_matrix_i8",
                "bytes",
                view.bytes.len(),
                expected_len,
            )?;
            return Ok(Some(Self::RawI8 {
                rows,
                cols,
                values: view.bytes.iter().map(|value| *value as i8).collect(),
            }));
        }
        if view.encoding == EncodingKind::BlockQuantized
            && matches!(
                view.storage_dtype,
                StorageDtype::Q8_0 | StorageDtype::Logical(LogicalDtype::I8)
            )
        {
            let block_size = view.meta.block_shape.first().copied().unwrap_or(32) as usize;
            validate_cpu_positive("quantized_matrix_q8_0", "block_size", block_size)?;
            return Ok(Some(Self::Q8_0 {
                rows,
                cols,
                block_size,
                bytes: view.bytes.to_vec(),
            }));
        }
        if view.encoding == EncodingKind::BlockQuantized && view.storage_dtype == StorageDtype::Q4_0
        {
            return Ok(Some(Self::Q4_0 {
                rows,
                cols,
                bytes: view.bytes.to_vec(),
            }));
        }
        if view.encoding == EncodingKind::BlockQuantized {
            let bytes = view.bytes.to_vec();
            return Ok(match view.storage_dtype {
                StorageDtype::Q3K => Some(Self::Q3K { rows, cols, bytes }),
                StorageDtype::Q4K => Some(Self::Q4K { rows, cols, bytes }),
                StorageDtype::Q5K => Some(Self::Q5K { rows, cols, bytes }),
                StorageDtype::Q6K => Some(Self::Q6K { rows, cols, bytes }),
                _ => None,
            });
        }
        Ok(None)
    }
}

pub(crate) fn native_decoder_weight_view<'a>(
    file: &'a RsmfFile,
    options: &NativeDecoderWeightOptions,
    tensor_name: &str,
) -> Result<TensorView<'a>> {
    let result = if let Some(variant_idx) = options.tensor_variants.get(tensor_name) {
        file.tensor_view_variant(tensor_name, *variant_idx)
    } else {
        file.tensor_view(tensor_name)
    };
    result.map_err(|error| match error {
        RsmfError::NotFound { what } if what == format!("tensor {tensor_name}") => {
            RuntimeError::NativeDecoderTensorMissing {
                tensor_name: tensor_name.to_string(),
            }
        }
        RsmfError::NotFound { what } => RuntimeError::NativeDecoderTensorUnsupported {
            tensor_name: tensor_name.to_string(),
            reason: what,
        },
        other => RuntimeError::Core(other),
    })
}

pub(crate) fn shape_element_count(shape: &[u64]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim).map_err(|_| {
            RuntimeError::Shape(format!(
                "native decoder dimension {dim} cannot convert to usize"
            ))
        })?;
        count
            .checked_mul(dim)
            .ok_or_else(|| RuntimeError::Shape("native decoder element count overflow".to_string()))
    })
}
