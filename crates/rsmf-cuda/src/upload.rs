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

/// Upload a byte slice into a newly allocated CUDA device buffer asynchronously.
///
/// # Safety and Synchronization
/// This function uses [`cudarc::driver::CudaDevice::htod_copy`], which
/// performs an **asynchronous** host-to-device copy, allowing overlapping
/// with computation. It consumes a `Vec` to keep the host memory alive
/// until the transfer completes.
pub fn upload_canonical_tensor_async(
    handle: &DeviceHandle,
    bytes: &[u8],
) -> Result<CudaSlice<u8>, UploadError> {
    if bytes.is_empty() {
        return Err(UploadError::Empty);
    }

    // Copy to an owned Vec so it can be safely referenced during async transfer
    let host_vec = bytes.to_vec();

    // htod_copy handles allocating the CudaSlice and starting the async copy
    let slice = handle.device.htod_copy(host_vec)?;

    Ok(slice)
}
