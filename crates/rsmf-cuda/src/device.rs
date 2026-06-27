//! CUDA device detection and management.

use std::sync::Arc;

/// A handle to an active CUDA device.
pub struct DeviceHandle {
    /// The underlying cudarc device handle.
    pub device: Arc<cudarc::driver::CudaDevice>,
}

/// Detect if CUDA capabilities are available on this system.
///
/// Returns `true` if at least one CUDA-capable device is detected.
///
/// `cudarc` loads the CUDA driver library lazily and *panics* when neither
/// `libcuda` nor `nvcuda` can be found, rather than returning an error. On a
/// machine without an installed CUDA driver — including hardware-free CI
/// runners — that panic would abort the caller. Detection must degrade
/// gracefully, so the driver call is wrapped in `catch_unwind` and any failure
/// to load the library is reported as "no capabilities".
pub fn detect_capabilities() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::result::device::get_count().unwrap_or(0) > 0)
        .unwrap_or(false)
}

/// Request a connection to a CUDA device.
///
/// Returns a handle if the device can be initialised, or an error otherwise.
pub fn request_device(ordinal: usize) -> Result<DeviceHandle, cudarc::driver::DriverError> {
    let device = cudarc::driver::CudaDevice::new(ordinal)?;
    Ok(DeviceHandle { device })
}
