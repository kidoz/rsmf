//! WGPU device capability detection and initialisation.
//!
//! None of the functions here panic if a GPU is missing — they return `Option`
//! so benchmarks and tools can skip gracefully.

use pollster::FutureExt;

/// Capabilities of the default-preferred WGPU adapter.
#[derive(Debug)]
pub struct WgpuCapabilities {
    /// The adapter itself. Owned so callers can request a device from it.
    pub adapter: wgpu::Adapter,
    /// Human-readable adapter info.
    pub info: wgpu::AdapterInfo,
    /// Adapter limits.
    pub limits: wgpu::Limits,
    /// Adapter features.
    pub features: wgpu::Features,
}

/// Logical handle to a device + queue pair.
#[derive(Debug)]
pub struct DeviceHandle {
    /// WGPU device.
    pub device: wgpu::Device,
    /// WGPU queue.
    pub queue: wgpu::Queue,
    /// Reported adapter info (copied from [`WgpuCapabilities`]).
    pub info: wgpu::AdapterInfo,
    /// Reported limits for the adapter.
    pub limits: wgpu::Limits,
}

/// Detect WGPU capabilities. Returns `None` if no adapter is available
/// (e.g. on CI runners without a GPU).
#[must_use]
pub fn detect_capabilities() -> Option<WgpuCapabilities> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .block_on()?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let features = adapter.features();
    Some(WgpuCapabilities {
        adapter,
        info,
        limits,
        features,
    })
}

/// Request a device + queue pair from the given capabilities. Returns `None`
/// on failure.
#[must_use]
pub fn request_device(caps: &WgpuCapabilities) -> Option<DeviceHandle> {
    let (device, queue) = caps
        .adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rsmf-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: caps.limits.clone(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .block_on()
        .ok()?;
    Some(DeviceHandle {
        device,
        queue,
        info: caps.info.clone(),
        limits: caps.limits.clone(),
    })
}
