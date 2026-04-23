//! # rsmf-metal
//!
//! Synchronous host→GPU upload helper for RSMF tensors on Apple Metal.
//!
//! This crate allocates a single `MTLBuffer` with `StorageModeShared`
//! and copies canonical tensor bytes into it. It is **not** a zero-copy
//! path, does not chunk the transfer, and does not materialise
//! Metal-specific packed layouts ([`rsmf_core::TargetTag::Metal`]) —
//! those remain future work. Callers that need chunked, staging-buffer
//! uploads on portable GPU hardware should use `rsmf-wgpu` instead.

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
