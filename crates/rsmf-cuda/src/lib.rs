//! # rsmf-cuda
//!
//! CUDA materialization path for RSMF tensors.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod device;
pub mod upload;

pub use device::{DeviceHandle, detect_capabilities, request_device};
pub use upload::{UploadError, upload_canonical_tensor};

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
