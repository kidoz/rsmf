//! # rsmf-cuda
//!
//! Synchronous host→device upload helper for RSMF tensors on CUDA.
//!
//! This crate is intentionally minimal: it performs a blocking
//! `htod_sync_copy` of canonical tensor bytes into a newly-allocated
//! `CudaSlice<u8>`. It is **not** a zero-copy path, does not chunk the
//! transfer, and does not materialise CUDA-specific packed layouts
//! ([`rsmf_core::TargetTag::Cuda`]) — those remain future work. Callers
//! that need chunked, staging-buffer uploads on portable GPU hardware
//! should use `rsmf-wgpu` instead.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod device;
pub mod upload;

pub use device::{DeviceHandle, detect_capabilities, request_device};
pub use upload::{UploadError, upload_canonical_tensor_async};

/// Common backend error type.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Device not found or unavailable.
    #[error("GPU device not found")]
    DeviceNotFound,
    /// Initialization failed.
    #[error("GPU initialization failed: {0}")]
    InitFailed(String),
}
