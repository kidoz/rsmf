//! CUDA materialization of a canonical tensor's bytes.

use cudarc::driver::{CudaSlice, DriverError};

use crate::device::DeviceHandle;

/// Reasons `upload_canonical_tensor` can fail.
#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    /// The CUDA driver returned an error.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    /// Empty tensor upload.
    #[error("cannot upload an empty tensor")]
    Empty,
}

/// Upload a byte slice into a newly allocated CUDA device buffer.
///
/// # Safety and Synchronization
/// This function uses [`cudarc::driver::CudaDevice::htod_sync_copy`], which
/// performs a **synchronous** (blocking) host-to-device copy. The calling
/// thread will block until the transfer is complete.
///
/// The returned [`CudaSlice`] is immediately usable for subsequent CUDA
/// operations on the same device. Memory is automatically freed when the
/// `CudaSlice` is dropped, provided no other references exist.
pub fn upload_canonical_tensor(
    handle: &DeviceHandle,
    bytes: &[u8],
) -> Result<CudaSlice<u8>, UploadError> {
    if bytes.is_empty() {
        return Err(UploadError::Empty);
    }
    // `htod_sync_copy` allocates a device buffer and synchronizes a Host-to-Device copy.
    let slice = handle.device.htod_sync_copy(bytes)?;
    Ok(slice)
}
