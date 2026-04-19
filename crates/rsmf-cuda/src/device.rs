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
pub fn detect_capabilities() -> bool {
    cudarc::driver::result::device::get_count().unwrap_or(0) > 0
}

/// Request a connection to a CUDA device.
///
/// Returns a handle if the device can be initialised, or an error otherwise.
pub fn request_device(ordinal: usize) -> Result<DeviceHandle, cudarc::driver::DriverError> {
    let device = cudarc::driver::CudaDevice::new(ordinal)?;
    Ok(DeviceHandle { device })
}
