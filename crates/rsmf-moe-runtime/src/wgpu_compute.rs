//! Small WGPU compute executor for the MoE runtime.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

use pollster::FutureExt;
use rsmf_core::{DeviceKind, PlacementManifest};
use wgpu::util::DeviceExt;

use crate::TransferRunReport;
use crate::transfer::{TransferEvent, TransferExecutor};
use crate::{MoeRuntimeError, Result};

const SHADER: &str = r#"
struct Dims {
    tokens: u32,
    rows: u32,
    cols: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> matrix: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    let row = gid.y;
    if (token >= dims.tokens || row >= dims.rows) {
        return;
    }

    var acc = 0.0;
    var col = 0u;
    loop {
        if (col >= dims.cols) {
            break;
        }
        acc = acc + input[token * dims.cols + col] * matrix[row * dims.cols + col];
        col = col + 1u;
    }
    output[token * dims.rows + row] = acc;
}
"#;

/// WGPU executor used by `MoeRuntime` when the `wgpu` feature is enabled.
#[derive(Debug)]
pub(crate) struct WgpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    matrix_cache: Mutex<BTreeMap<MatrixCacheKey, wgpu::Buffer>>,
}

/// Output of one WGPU matrix multiplication.
#[derive(Debug)]
pub(crate) struct WgpuMatmulOutput {
    /// Row-major result values.
    pub(crate) values: Vec<f32>,
    /// True when the matrix buffer was already resident on the adapter.
    pub(crate) cache_hit: bool,
    /// Transfer report for the resident matrix buffer lookup/upload.
    pub(crate) transfer: TransferRunReport,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MatrixCacheKey {
    tensor_name: String,
    rows: usize,
    cols: usize,
}

/// Pool of physical WGPU adapter executors mapped to logical placement devices.
#[derive(Debug)]
pub(crate) struct WgpuExecutorPool {
    executors: Vec<WgpuExecutor>,
    device_assignments: BTreeMap<u32, usize>,
    requested_devices: usize,
    available_adapters: usize,
}

impl WgpuExecutorPool {
    pub(crate) fn new(placement: &PlacementManifest) -> Option<Self> {
        let wgpu_device_ids = placement
            .devices
            .iter()
            .filter(|device| device.kind == DeviceKind::Wgpu)
            .map(|device| device.id)
            .collect::<Vec<_>>();
        if wgpu_device_ids.is_empty() {
            return None;
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        if adapters.is_empty() {
            return None;
        }

        let requested_devices = wgpu_device_ids.len();
        let available_adapters = adapters.len();
        let executor_count = requested_devices.min(available_adapters);
        let mut executors = Vec::with_capacity(executor_count);
        for adapter in adapters.into_iter().take(executor_count) {
            if let Some(executor) = WgpuExecutor::from_adapter(adapter) {
                executors.push(executor);
            }
        }
        if executors.is_empty() {
            return None;
        }

        let mut device_assignments = BTreeMap::new();
        for (idx, device_id) in wgpu_device_ids.into_iter().enumerate() {
            device_assignments.insert(device_id, idx % executors.len());
        }

        Some(Self {
            executors,
            device_assignments,
            requested_devices,
            available_adapters,
        })
    }

    pub(crate) fn executor_slot_for_device(&self, device_id: u32) -> Option<usize> {
        self.device_assignments
            .get(&device_id)
            .copied()
            .filter(|executor_idx| *executor_idx < self.executors.len())
    }

    pub(crate) fn executor_by_slot(&self, executor_idx: usize) -> Option<&WgpuExecutor> {
        self.executors.get(executor_idx)
    }

    pub(crate) fn requested_devices(&self) -> usize {
        self.requested_devices
    }

    pub(crate) fn available_adapters(&self) -> usize {
        self.available_adapters
    }

    pub(crate) fn active_adapters(&self) -> usize {
        self.executors.len()
    }

    pub(crate) fn adapter_names(&self) -> Vec<String> {
        self.executors
            .iter()
            .map(|executor| executor.adapter_name().to_string())
            .collect()
    }
}

impl WgpuExecutor {
    fn from_adapter(adapter: wgpu::Adapter) -> Option<Self> {
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rsmf-moe-runtime"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .block_on()
            .ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsmf-moe matmul shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rsmf-moe matmul bgl"),
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rsmf-moe matmul pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsmf-moe matmul pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            adapter_name: info.name,
            matrix_cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub(crate) fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub(crate) fn matmul_cached(
        &self,
        tensor_name: &str,
        matrix: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
        tokens: usize,
    ) -> Result<WgpuMatmulOutput> {
        if tokens == 0 {
            return Ok(WgpuMatmulOutput {
                values: Vec::new(),
                cache_hit: true,
                transfer: TransferExecutor::wgpu().execute(TransferEvent::host_to_device(
                    0,
                    Duration::ZERO,
                    true,
                ))?,
            });
        }
        let expected_matrix = rows.checked_mul(cols).ok_or_else(|| {
            MoeRuntimeError::Shape("WGPU matrix element count overflow".to_string())
        })?;
        if matrix.len() != expected_matrix {
            return Err(MoeRuntimeError::Shape(format!(
                "WGPU matrix has {} values, expected {expected_matrix}",
                matrix.len()
            )));
        }
        let expected_input = tokens.checked_mul(cols).ok_or_else(|| {
            MoeRuntimeError::Shape("WGPU input element count overflow".to_string())
        })?;
        if input.len() != expected_input {
            return Err(MoeRuntimeError::Shape(format!(
                "WGPU input has {} values, expected {expected_input}",
                input.len()
            )));
        }

        let output_len = tokens.checked_mul(rows).ok_or_else(|| {
            MoeRuntimeError::Shape("WGPU output element count overflow".to_string())
        })?;
        let output_bytes = byte_len(output_len)?;
        let cached_matrix = self.cached_matrix_buffer(tensor_name, matrix, rows, cols)?;

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsmf-moe input"),
                contents: &f32s_to_le_bytes(input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsmf-moe output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsmf-moe dims"),
                contents: &dims_bytes(tokens, rows, cols)?,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsmf-moe staging"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rsmf-moe matmul bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                binding(0, &input_buffer),
                binding(1, &cached_matrix.buffer),
                binding(2, &output_buffer),
                binding(3, &dims_buffer),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rsmf-moe matmul encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsmf-moe matmul pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(div_ceil(tokens as u32, 8), div_ceil(rows as u32, 8), 1);
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
            .map_err(|e| MoeRuntimeError::Unsupported(format!("WGPU device poll failed: {e}")))?;
        receiver
            .recv()
            .map_err(|e| MoeRuntimeError::Unsupported(format!("WGPU map callback failed: {e}")))?
            .map_err(|e| MoeRuntimeError::Unsupported(format!("WGPU output map failed: {e}")))?;

        let mapped = slice.get_mapped_range();
        let out = le_bytes_to_f32s(&mapped)?;
        drop(mapped);
        staging_buffer.unmap();
        Ok(WgpuMatmulOutput {
            values: out,
            cache_hit: cached_matrix.cache_hit,
            transfer: cached_matrix.transfer,
        })
    }

    fn cached_matrix_buffer(
        &self,
        tensor_name: &str,
        matrix: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<CachedMatrixBuffer> {
        let key = MatrixCacheKey {
            tensor_name: tensor_name.to_string(),
            rows,
            cols,
        };
        if let Some(buffer) = self
            .matrix_cache
            .lock()
            .map_err(|_| MoeRuntimeError::Unsupported("WGPU matrix cache poisoned".to_string()))?
            .get(&key)
            .cloned()
        {
            return Ok(CachedMatrixBuffer {
                buffer,
                cache_hit: true,
                transfer: TransferExecutor::wgpu().execute(TransferEvent::host_to_device(
                    0,
                    Duration::ZERO,
                    true,
                ))?,
            });
        }

        let transfer_start = Instant::now();
        let matrix_bytes = f32s_to_le_bytes(matrix);
        let transfer_bytes = matrix_bytes.len();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsmf-moe resident matrix"),
                contents: &matrix_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let transfer_time = transfer_start.elapsed();
        self.matrix_cache
            .lock()
            .map_err(|_| MoeRuntimeError::Unsupported("WGPU matrix cache poisoned".to_string()))?
            .insert(key, buffer.clone());
        Ok(CachedMatrixBuffer {
            buffer,
            cache_hit: false,
            transfer: TransferExecutor::wgpu().execute(TransferEvent::host_to_device(
                transfer_bytes,
                transfer_time,
                false,
            ))?,
        })
    }
}

#[derive(Debug)]
struct CachedMatrixBuffer {
    buffer: wgpu::Buffer,
    cache_hit: bool,
    transfer: TransferRunReport,
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

fn byte_len(values: usize) -> Result<u64> {
    values
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| MoeRuntimeError::Shape("WGPU buffer byte length overflow".to_string()))
}

fn dims_bytes(tokens: usize, rows: usize, cols: usize) -> Result<Vec<u8>> {
    let dims = [
        u32::try_from(tokens)
            .map_err(|_| MoeRuntimeError::Shape("WGPU token count exceeds u32::MAX".to_string()))?,
        u32::try_from(rows)
            .map_err(|_| MoeRuntimeError::Shape("WGPU row count exceeds u32::MAX".to_string()))?,
        u32::try_from(cols).map_err(|_| {
            MoeRuntimeError::Shape("WGPU column count exceeds u32::MAX".to_string())
        })?,
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

fn le_bytes_to_f32s(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(MoeRuntimeError::Shape(format!(
            "WGPU output byte length {} is not divisible by 4",
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
