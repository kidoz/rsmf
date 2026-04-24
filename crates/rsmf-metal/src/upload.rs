//! Metal materialization of a canonical tensor's bytes.

#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, MTLResourceOptions};

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
/// Placeholder buffer type used on non-Apple targets.
pub struct Buffer;

use crate::device::DeviceHandle;

/// Reasons `upload_canonical_tensor` can fail.
#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    /// Empty tensor upload.
    #[error("cannot upload an empty tensor")]
    Empty,
    /// Metal is unavailable on this compilation target.
    #[error("metal uploads are only supported on Apple targets")]
    UnsupportedPlatform,
}

/// Upload a byte slice into a newly allocated Metal device buffer.
///
/// For systems with Unified Memory (e.g. Apple Silicon), this uses
/// `StorageModeShared` to minimise copy overhead while keeping the buffer
/// accessible to the GPU.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn upload_canonical_tensor(handle: &DeviceHandle, bytes: &[u8]) -> Result<Buffer, UploadError> {
    if bytes.is_empty() {
        return Err(UploadError::Empty);
    }

    let has_unified = handle.device.has_unified_memory();
    let options = if has_unified {
        MTLResourceOptions::StorageModeShared
    } else {
        MTLResourceOptions::StorageModeManaged
    };

    // Create a new buffer and initialize it with the slice data.
    let buffer =
        handle
            .device
            .new_buffer_with_data(bytes.as_ptr() as *const _, bytes.len() as u64, options);

    if !has_unified {
        buffer.did_modify_range(metal::NSRange::new(0, bytes.len() as u64));
    }

    Ok(buffer)
}

/// Upload a byte slice into a newly allocated Metal device buffer.
///
/// Returns [`UploadError::UnsupportedPlatform`] on non-Apple targets because
/// Metal APIs are unavailable there.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn upload_canonical_tensor(
    _handle: &DeviceHandle,
    bytes: &[u8],
) -> Result<Buffer, UploadError> {
    if bytes.is_empty() {
        return Err(UploadError::Empty);
    }

    Err(UploadError::UnsupportedPlatform)
}
