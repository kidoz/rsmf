//! Metal device detection and management.

#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{CommandQueue, Device};

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
/// Placeholder device type used on non-Apple targets so the crate remains
/// buildable as part of the cross-platform workspace.
pub struct Device;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
/// Placeholder command queue type used on non-Apple targets.
pub struct CommandQueue;

/// A handle to an active Metal device and a default command queue.
pub struct DeviceHandle {
    /// The underlying Metal device.
    pub device: Device,
    /// A command queue for submitting work.
    pub queue: CommandQueue,
}

/// Detect if Metal capabilities are available on this system.
///
/// Returns `true` if the system registry has at least one Metal device.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn detect_capabilities() -> bool {
    !Device::all().is_empty()
}

/// Detect if Metal capabilities are available on this system.
///
/// Returns `false` on non-Apple targets because Metal is unavailable.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn detect_capabilities() -> bool {
    false
}

/// Request a connection to the system's default Metal device.
///
/// Returns a handle if successful, or `None` if Metal is unavailable.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn request_device() -> Option<DeviceHandle> {
    let device = Device::system_default()?;
    let queue = device.new_command_queue();
    Some(DeviceHandle { device, queue })
}

/// Request a connection to the system's default Metal device.
///
/// Returns `None` on non-Apple targets because Metal is unavailable.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn request_device() -> Option<DeviceHandle> {
    None
}
