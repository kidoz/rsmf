use super::*;

pub(crate) fn resolve_native_decoder_backend(
    backend: NativeDecoderBackend,
) -> Result<NativeDecoderBackend> {
    match backend {
        NativeDecoderBackend::Auto | NativeDecoderBackend::CpuReference => Ok(NativeDecoderBackend::CpuReference),
        NativeDecoderBackend::CpuThreaded => Ok(NativeDecoderBackend::CpuThreaded),
        NativeDecoderBackend::AppleCpuAccelerate | NativeDecoderBackend::Accelerated => {
            if apple_accelerate_available() {
                Ok(NativeDecoderBackend::AppleCpuAccelerate)
            } else {
                Ok(NativeDecoderBackend::CpuReference)
            }
        }
        NativeDecoderBackend::MetalWgpuLmHead => {
            if native_decoder_wgpu_linear_available() {
                Ok(NativeDecoderBackend::MetalWgpuLmHead)
            } else {
                Err(native_decoder_wgpu_unavailable_error())
            }
        }
        NativeDecoderBackend::MetalWgpuFullDecoder => {
            Err(RuntimeError::NativeDecoderBackendUnavailable {
                backend: "metal_wgpu_full_decoder".to_string(),
                reason: "Metal/WGPU full decoder kernels are not implemented yet".to_string(),
            })
        }
        NativeDecoderBackend::OrtCoreMl => Err(RuntimeError::NativeDecoderBackendUnavailable {
            backend: "ort_core_ml".to_string(),
            reason: "ORT CoreML execution provider applies to graph payloads, not the native decoder path yet".to_string(),
        }),
    }
}

pub(crate) fn native_decoder_wgpu_linear_available() -> bool {
    native_decoder_wgpu_executor().is_ok()
}

pub(crate) fn apple_accelerate_available() -> bool {
    cfg!(all(target_os = "macos", feature = "apple-accelerate"))
}

pub(crate) fn native_decoder_backend_linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
    backend: NativeDecoderBackend,
    performance: &NativeDecoderPerformanceOptions,
) -> Result<Vec<f32>> {
    match backend {
        NativeDecoderBackend::AppleCpuAccelerate => {
            native_decoder_apple_accelerate_linear(input, rows, in_features, weight, out_features)
        }
        NativeDecoderBackend::CpuThreaded if rows > 1 => native_decoder_cpu_linear_threaded(
            input,
            rows,
            in_features,
            weight,
            out_features,
            performance.cpu_threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
            }),
        ),
        _ => native_decoder_cpu_linear(input, rows, in_features, weight, out_features),
    }
}

#[cfg(feature = "wgpu")]
pub(crate) fn native_decoder_wgpu_linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    native_decoder_wgpu_executor()?
        .linear(input, rows, in_features, weight, out_features)
        .map_err(|error| RuntimeError::NativeDecoderBackendUnavailable {
            backend: "metal_wgpu_lm_head".to_string(),
            reason: error.to_string(),
        })
}

#[cfg(not(feature = "wgpu"))]
pub(crate) fn native_decoder_wgpu_linear(
    _input: &[f32],
    _rows: usize,
    _in_features: usize,
    _weight: &[f32],
    _out_features: usize,
) -> Result<Vec<f32>> {
    Err(native_decoder_wgpu_unavailable_error())
}

#[cfg(feature = "wgpu")]
fn native_decoder_wgpu_executor() -> Result<&'static rsmf_wgpu::WgpuLinearExecutor> {
    static EXECUTOR: std::sync::OnceLock<
        std::result::Result<rsmf_wgpu::WgpuLinearExecutor, String>,
    > = std::sync::OnceLock::new();
    match EXECUTOR.get_or_init(|| rsmf_wgpu::WgpuLinearExecutor::new().map_err(|e| e.to_string())) {
        Ok(executor) => Ok(executor),
        Err(reason) => Err(RuntimeError::NativeDecoderBackendUnavailable {
            backend: "metal_wgpu_lm_head".to_string(),
            reason: reason.clone(),
        }),
    }
}

#[cfg(not(feature = "wgpu"))]
fn native_decoder_wgpu_executor() -> Result<()> {
    Err(native_decoder_wgpu_unavailable_error())
}

fn native_decoder_wgpu_unavailable_error() -> RuntimeError {
    RuntimeError::NativeDecoderBackendUnavailable {
        backend: "metal_wgpu_lm_head".to_string(),
        reason: native_decoder_wgpu_unavailable_reason(),
    }
}

#[cfg(feature = "wgpu")]
fn native_decoder_wgpu_unavailable_reason() -> String {
    "WGPU device initialization failed or no compatible adapter is available".to_string()
}

#[cfg(not(feature = "wgpu"))]
fn native_decoder_wgpu_unavailable_reason() -> String {
    "rsmf-runtime was built without the wgpu feature".to_string()
}

#[cfg(all(target_os = "macos", feature = "apple-accelerate"))]
pub(crate) fn native_decoder_apple_accelerate_linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    validate_cpu_matrix_len(
        "linear_apple_accelerate",
        "input",
        input.len(),
        rows,
        in_features,
    )?;
    validate_cpu_matrix_len(
        "linear_apple_accelerate",
        "weight",
        weight.len(),
        out_features,
        in_features,
    )?;
    let output_len = cpu_element_count("linear_apple_accelerate", "output", rows, out_features)?;
    let mut output = vec![0.0f32; output_len];
    let rows_i32 = i32::try_from(rows).map_err(|_| {
        native_decoder_cpu_shape_error("linear_apple_accelerate", "rows exceed i32")
    })?;
    let in_i32 = i32::try_from(in_features).map_err(|_| {
        native_decoder_cpu_shape_error("linear_apple_accelerate", "in_features exceed i32")
    })?;
    let out_i32 = i32::try_from(out_features).map_err(|_| {
        native_decoder_cpu_shape_error("linear_apple_accelerate", "out_features exceed i32")
    })?;
    if rows == 1 {
        // SAFETY: All pointers are derived from validated Rust slices. Matrix
        // dimensions and strides are checked above and converted to the CBLAS
        // `i32` ABI before the call. Output is uniquely borrowed and sized for
        // `out_features` elements.
        unsafe {
            apple_accelerate::cblas_sgemv(
                apple_accelerate::CBLAS_ROW_MAJOR,
                apple_accelerate::CBLAS_NO_TRANS,
                out_i32,
                in_i32,
                1.0,
                weight.as_ptr(),
                in_i32,
                input.as_ptr(),
                1,
                0.0,
                output.as_mut_ptr(),
                1,
            );
        }
    } else {
        // SAFETY: All pointers are derived from validated Rust slices. A is
        // row-major `[rows, in_features]`, B is row-major
        // `[out_features, in_features]` and is passed transposed, and C is
        // row-major `[rows, out_features]` with non-overlapping output storage.
        unsafe {
            apple_accelerate::cblas_sgemm(
                apple_accelerate::CBLAS_ROW_MAJOR,
                apple_accelerate::CBLAS_NO_TRANS,
                apple_accelerate::CBLAS_TRANS,
                rows_i32,
                out_i32,
                in_i32,
                1.0,
                input.as_ptr(),
                in_i32,
                weight.as_ptr(),
                in_i32,
                0.0,
                output.as_mut_ptr(),
                out_i32,
            );
        }
    }
    Ok(output)
}

#[cfg(not(all(target_os = "macos", feature = "apple-accelerate")))]
pub(crate) fn native_decoder_apple_accelerate_linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    native_decoder_cpu_linear(input, rows, in_features, weight, out_features)
}

#[cfg(all(target_os = "macos", feature = "apple-accelerate"))]
mod apple_accelerate {
    pub const CBLAS_ROW_MAJOR: i32 = 101;
    pub const CBLAS_NO_TRANS: i32 = 111;
    pub const CBLAS_TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemv(
            layout: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            x: *const f32,
            inc_x: i32,
            beta: f32,
            y: *mut f32,
            inc_y: i32,
        );

        pub fn cblas_sgemm(
            layout: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}
