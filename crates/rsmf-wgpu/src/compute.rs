//! Small WGPU compute kernels used by higher-level runtimes.

use std::borrow::Cow;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::{detect_capabilities, request_device};

const LINEAR_SHADER: &str = r#"
struct Dims {
    rows: u32,
    out_features: u32,
    in_features: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let out_col = gid.y;
    if (row >= dims.rows || out_col >= dims.out_features) {
        return;
    }

    var acc = 0.0;
    var in_col = 0u;
    loop {
        if (in_col >= dims.in_features) {
            break;
        }
        acc = acc + input[row * dims.in_features + in_col] *
            weight[out_col * dims.in_features + in_col];
        in_col = in_col + 1u;
    }
    output[row * dims.out_features + out_col] = acc;
}
"#;

/// Errors returned by WGPU compute helpers.
#[derive(Debug, thiserror::Error)]
pub enum WgpuComputeError {
    /// No WGPU adapter/device could be initialized.
    #[error("WGPU device is unavailable")]
    DeviceUnavailable,
    /// Input, weight, or output dimensions are invalid.
    #[error("invalid WGPU compute shape: {0}")]
    Shape(String),
    /// GPU output readback failed.
    #[error("WGPU output readback failed: {0}")]
    Readback(String),
}

/// Reusable WGPU f32 linear projection executor.
#[derive(Debug)]
pub struct WgpuLinearExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
}

impl WgpuLinearExecutor {
    /// Create an executor on the default-preferred WGPU adapter.
    pub fn new() -> Result<Self, WgpuComputeError> {
        let caps = detect_capabilities().ok_or(WgpuComputeError::DeviceUnavailable)?;
        let adapter_name = caps.info.name.clone();
        let handle = request_device(&caps).ok_or(WgpuComputeError::DeviceUnavailable)?;
        let shader = handle
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rsmf-wgpu linear shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LINEAR_SHADER)),
            });
        let bind_group_layout =
            handle
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rsmf-wgpu linear bgl"),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, false),
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let pipeline_layout =
            handle
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rsmf-wgpu linear pipeline layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = handle
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rsmf-wgpu linear pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Ok(Self {
            device: handle.device,
            queue: handle.queue,
            bind_group_layout,
            pipeline,
            adapter_name,
        })
    }

    /// Human-readable WGPU adapter name.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Execute row-major f32 linear projection.
    ///
    /// `input` has shape `[rows, in_features]`, `weight` has shape
    /// `[out_features, in_features]`, and the result has shape
    /// `[rows, out_features]`.
    pub fn linear(
        &self,
        input: &[f32],
        rows: usize,
        in_features: usize,
        weight: &[f32],
        out_features: usize,
    ) -> Result<Vec<f32>, WgpuComputeError> {
        validate_matrix("input", input.len(), rows, in_features)?;
        validate_matrix("weight", weight.len(), out_features, in_features)?;
        let output_len = element_count("output", rows, out_features)?;
        if output_len == 0 {
            return Ok(Vec::new());
        }
        let output_bytes = byte_len(output_len)?;

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsmf-wgpu linear input"),
                contents: &f32s_to_le_bytes(input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let weight_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsmf-wgpu linear weight"),
                contents: &f32s_to_le_bytes(weight),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsmf-wgpu linear output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsmf-wgpu linear dims"),
                contents: &dims_bytes(rows, out_features, in_features)?,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsmf-wgpu linear staging"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rsmf-wgpu linear bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                binding(0, &input_buffer),
                binding(1, &weight_buffer),
                binding(2, &output_buffer),
                binding(3, &dims_buffer),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rsmf-wgpu linear encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsmf-wgpu linear pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                div_ceil(rows as u32, 8),
                div_ceil(out_features as u32, 8),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_bytes);
        self.queue.submit([encoder.finish()]);

        let slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ignored = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|error| WgpuComputeError::Readback(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| WgpuComputeError::Readback(error.to_string()))?
            .map_err(|error| WgpuComputeError::Readback(error.to_string()))?;

        let mapped = slice.get_mapped_range();
        let output = le_bytes_to_f32s(&mapped)?;
        drop(mapped);
        staging_buffer.unmap();
        Ok(output)
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn binding<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn validate_matrix(
    name: &'static str,
    len: usize,
    rows: usize,
    cols: usize,
) -> Result<(), WgpuComputeError> {
    let expected = element_count(name, rows, cols)?;
    if len != expected {
        return Err(WgpuComputeError::Shape(format!(
            "{name} has {len} values, expected {expected}"
        )));
    }
    Ok(())
}

fn element_count(name: &'static str, rows: usize, cols: usize) -> Result<usize, WgpuComputeError> {
    rows.checked_mul(cols)
        .ok_or_else(|| WgpuComputeError::Shape(format!("{name} element count overflow")))
}

fn byte_len(values: usize) -> Result<u64, WgpuComputeError> {
    values
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| WgpuComputeError::Shape("buffer byte length overflow".to_string()))
}

fn dims_bytes(
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<u8>, WgpuComputeError> {
    let dims = [
        u32::try_from(rows)
            .map_err(|_| WgpuComputeError::Shape("row count exceeds u32::MAX".to_string()))?,
        u32::try_from(out_features)
            .map_err(|_| WgpuComputeError::Shape("out_features exceeds u32::MAX".to_string()))?,
        u32::try_from(in_features)
            .map_err(|_| WgpuComputeError::Shape("in_features exceeds u32::MAX".to_string()))?,
        0,
    ];
    let mut out = Vec::with_capacity(16);
    for value in dims {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

fn f32s_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn le_bytes_to_f32s(bytes: &[u8]) -> Result<Vec<f32>, WgpuComputeError> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(WgpuComputeError::Shape(format!(
            "output byte length {} is not divisible by 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}
