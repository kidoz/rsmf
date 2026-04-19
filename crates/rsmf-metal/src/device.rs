//! Metal device detection and management.

use metal::{CommandQueue, Device};

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
pub fn detect_capabilities() -> bool {
    !Device::all().is_empty()
}

/// Request a connection to the system's default Metal device.
///
/// Returns a handle if successful, or `None` if Metal is unavailable.
pub fn request_device() -> Option<DeviceHandle> {
    let device = Device::system_default()?;
    let queue = device.new_command_queue();
    Some(DeviceHandle { device, queue })
}
