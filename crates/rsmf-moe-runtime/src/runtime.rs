//! Runtime planner and F32 MoE layer execution.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rsmf_core::{
    DeviceKind, MoeRole, PlacementManifest, PrefetchIndex, RsmfFile, TensorDescriptor,
};

use crate::{
    DeviceBatch, MoeRunOutput, MoeRunReport, MoeRuntimeError, Result, RuntimeBackend,
    batch_by_destination,
};

/// Hidden activation used between expert projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertActivation {
    /// `hidden = up(x)`.
    Identity,
    /// `hidden = silu(gate(x)) * up(x)`. Requires a `moe.role=gate` tensor for
    /// every expert.
    SiluGated,
}

/// Runtime construction options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeRuntimeOptions {
    /// Prefer WGPU when the crate was built with the `wgpu` feature.
    pub prefer_wgpu: bool,
    /// Expert activation convention.
    pub activation: ExpertActivation,
}

impl Default for MoeRuntimeOptions {
    fn default() -> Self {
        Self {
            prefer_wgpu: false,
            activation: ExpertActivation::Identity,
        }
    }
}

/// Minimal MoE runtime over an opened RSMF file.
///
/// The file must already have any external shard files attached with
/// [`RsmfFile::open_with_shards`] or [`RsmfFile::with_shard_path`].
#[derive(Debug)]
pub struct MoeRuntime {
    file: RsmfFile,
    placement: PlacementManifest,
    prefetch: PrefetchIndex,
    backend: RuntimeBackend,
    options: MoeRuntimeOptions,
}

impl MoeRuntime {
    /// Build a runtime from an opened RSMF file.
    ///
    /// Returns an error if the file has no placement manifest, because the PoC
    /// uses placement records to map expert shards to logical devices.
    pub fn new(file: RsmfFile, options: MoeRuntimeOptions) -> Result<Self> {
        let placement = file
            .placement_manifest()?
            .ok_or_else(|| MoeRuntimeError::Missing("PlacementManifest".to_string()))?;
        let prefetch = file.prefetch_hints()?;
        let backend = select_backend(&placement, options.prefer_wgpu);
        Ok(Self {
            file,
            placement,
            prefetch,
            backend,
            options,
        })
    }

    /// Borrow the underlying RSMF file.
    #[must_use]
    pub fn file(&self) -> &RsmfFile {
        &self.file
    }

    /// Borrow the placement manifest consumed by the runtime.
    #[must_use]
    pub fn placement(&self) -> &PlacementManifest {
        &self.placement
    }

    /// Return the backend status selected at construction time.
    #[must_use]
    pub fn backend(&self) -> &RuntimeBackend {
        &self.backend
    }

    /// Run one MoE layer using host-side top-1 gating and placement-aware
    /// expert batches.
    ///
    /// `input` is row-major `[token_count, input_width]`. The router tensor for
    /// `layer` must carry `moe.role=router` and shape
    /// `[n_experts, input_width]`. Expert tensors use `moe.expert` plus
    /// `moe.role=up` / `down` and optional `gate`.
    pub fn run_layer_top1(
        &self,
        layer: u32,
        input: &[f32],
        input_width: usize,
    ) -> Result<MoeRunOutput> {
        let loaded = self.load_layer(layer, input_width)?;
        let token_count = checked_token_count(input, input_width)?;

        let gating_start = Instant::now();
        let assignments = route_top1(&loaded.router, input, token_count)?;
        let gating_time = gating_start.elapsed();

        let dispatch_start = Instant::now();
        let routing_batches = batch_by_destination(&assignments);
        let device_batches = self.device_batches(&loaded, &routing_batches)?;
        let dispatch_time = dispatch_start.elapsed();

        let mut output = vec![0.0; token_count * loaded.output_width];
        let compute_start = Instant::now();
        for batch in &device_batches {
            let expert = loaded
                .experts
                .get(&batch.expert_id)
                .ok_or_else(|| MoeRuntimeError::Missing(format!("expert {}", batch.expert_id)))?;
            for &token_idx in &batch.token_indices {
                let input_row = &input[token_idx * input_width..(token_idx + 1) * input_width];
                let row = eval_expert(input_row, expert, self.options.activation)?;
                let out_start = token_idx * loaded.output_width;
                output[out_start..out_start + loaded.output_width].copy_from_slice(&row);
            }
        }
        let compute_time = compute_start.elapsed();

        let combine_start = Instant::now();
        // Output rows are scattered into their final positions during compute;
        // keep this phase explicit so reports have the same shape as a future
        // device-backed combine path.
        let combine_time = combine_start.elapsed();

        Ok(MoeRunOutput {
            output,
            output_width: loaded.output_width,
            report: MoeRunReport {
                backend: self.backend.clone(),
                token_count,
                input_width,
                output_width: loaded.output_width,
                device_batches,
                gating_time,
                dispatch_time,
                compute_time,
                combine_time,
            },
        })
    }

    /// Run the same layer as [`Self::run_layer_top1`] without placement-based
    /// batching.
    ///
    /// This is the single-device correctness reference for the PoC.
    pub fn run_layer_reference_top1(
        &self,
        layer: u32,
        input: &[f32],
        input_width: usize,
    ) -> Result<Vec<f32>> {
        let loaded = self.load_layer(layer, input_width)?;
        let token_count = checked_token_count(input, input_width)?;
        let assignments = route_top1(&loaded.router, input, token_count)?;
        let mut output = vec![0.0; token_count * loaded.output_width];
        for (token_idx, expert_id) in assignments.iter().copied().enumerate() {
            let expert = loaded
                .experts
                .get(&expert_id)
                .ok_or_else(|| MoeRuntimeError::Missing(format!("expert {expert_id}")))?;
            let input_row = &input[token_idx * input_width..(token_idx + 1) * input_width];
            let row = eval_expert(input_row, expert, self.options.activation)?;
            let out_start = token_idx * loaded.output_width;
            output[out_start..out_start + loaded.output_width].copy_from_slice(&row);
        }
        Ok(output)
    }

    fn load_layer(&self, layer: u32, input_width: usize) -> Result<LoadedLayer> {
        let moe = self.file.moe_experts()?;
        let router_entry = moe
            .entries
            .iter()
            .find(|entry| entry.layer == layer && entry.role == MoeRole::Router)
            .ok_or_else(|| MoeRuntimeError::Missing(format!("layer {layer} router tensor")))?;
        let router = load_matrix(&self.file, &router_entry.tensor_name)?;
        if router.cols != input_width {
            return Err(MoeRuntimeError::Shape(format!(
                "router {} has input width {}, got {input_width}",
                router_entry.tensor_name, router.cols
            )));
        }

        let mut builders: HashMap<u32, ExpertBuilder> = HashMap::new();
        for entry in moe
            .entries
            .iter()
            .filter(|entry| entry.layer == layer && entry.expert_id.is_some())
        {
            let expert_id = entry.expert_id.unwrap_or(0);
            let builder = builders.entry(expert_id).or_insert_with(|| ExpertBuilder {
                up_name: None,
                down_name: None,
                gate_name: None,
            });
            match &entry.role {
                MoeRole::Up => builder.up_name = Some(entry.tensor_name.clone()),
                MoeRole::Down => builder.down_name = Some(entry.tensor_name.clone()),
                MoeRole::Gate => builder.gate_name = Some(entry.tensor_name.clone()),
                _ => {}
            }
        }

        let mut experts = HashMap::with_capacity(builders.len());
        let mut output_width = None;
        for (expert_id, builder) in builders {
            let up_name = builder.up_name.ok_or_else(|| {
                MoeRuntimeError::Missing(format!("layer {layer} expert {expert_id} up tensor"))
            })?;
            let down_name = builder.down_name.ok_or_else(|| {
                MoeRuntimeError::Missing(format!("layer {layer} expert {expert_id} down tensor"))
            })?;
            let up = load_matrix(&self.file, &up_name)?;
            let down = load_matrix(&self.file, &down_name)?;
            if up.cols != input_width {
                return Err(MoeRuntimeError::Shape(format!(
                    "{up_name} has input width {}, expected {input_width}",
                    up.cols
                )));
            }
            if down.cols != up.rows {
                return Err(MoeRuntimeError::Shape(format!(
                    "{down_name} input width {} does not match {up_name} hidden width {}",
                    down.cols, up.rows
                )));
            }
            match output_width {
                Some(width) if width != down.rows => {
                    return Err(MoeRuntimeError::Shape(format!(
                        "{down_name} output width {} differs from earlier expert output width {width}",
                        down.rows
                    )));
                }
                _ => output_width = Some(down.rows),
            }
            let gate = if let Some(gate_name) = builder.gate_name {
                let gate = load_matrix(&self.file, &gate_name)?;
                if gate.rows != up.rows || gate.cols != up.cols {
                    return Err(MoeRuntimeError::Shape(format!(
                        "{gate_name} shape [{}x{}] does not match {up_name} shape [{}x{}]",
                        gate.rows, gate.cols, up.rows, up.cols
                    )));
                }
                Some(gate)
            } else if self.options.activation == ExpertActivation::SiluGated {
                return Err(MoeRuntimeError::Missing(format!(
                    "layer {layer} expert {expert_id} gate tensor"
                )));
            } else {
                None
            };
            let shard_id = shared_expert_shard(&self.file, &[&up_name, &down_name])?;
            let placement = self.placement_for_shard(shard_id)?;
            let prefetch_groups =
                self.prefetch_groups_for_tensors([up_name.as_str(), down_name.as_str()]);
            experts.insert(
                expert_id,
                ExpertWeights {
                    up,
                    down,
                    gate,
                    shard_id,
                    device_id: placement.primary_device,
                    prefetch_groups,
                },
            );
        }

        let output_width = output_width
            .ok_or_else(|| MoeRuntimeError::Missing(format!("layer {layer} expert tensors")))?;
        Ok(LoadedLayer {
            router,
            experts,
            output_width,
        })
    }

    fn placement_for_shard(&self, shard_id: u64) -> Result<&rsmf_core::PlacementRecord> {
        let placement = self
            .placement
            .placements
            .iter()
            .find(|placement| placement.shard_id == shard_id)
            .ok_or_else(|| {
                MoeRuntimeError::Missing(format!("placement record for expert shard_id {shard_id}"))
            })?;
        if !self
            .placement
            .devices
            .iter()
            .any(|device| device.id == placement.primary_device)
        {
            return Err(MoeRuntimeError::Missing(format!(
                "placement primary_device {}",
                placement.primary_device
            )));
        }
        Ok(placement)
    }

    fn prefetch_groups_for_tensors<'a>(
        &self,
        tensor_names: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let wanted: HashSet<&str> = tensor_names.into_iter().collect();
        let mut groups = Vec::new();
        for entry in &self.prefetch.entries {
            if wanted.contains(entry.tensor_name.as_str()) && !groups.contains(&entry.group) {
                groups.push(entry.group.clone());
            }
        }
        groups
    }

    fn device_batches(
        &self,
        loaded: &LoadedLayer,
        batches: &[crate::RoutingBatch],
    ) -> Result<Vec<DeviceBatch>> {
        let mut out = Vec::with_capacity(batches.len());
        for batch in batches {
            let expert = loaded
                .experts
                .get(&batch.expert_id)
                .ok_or_else(|| MoeRuntimeError::Missing(format!("expert {}", batch.expert_id)))?;
            out.push(DeviceBatch {
                expert_id: batch.expert_id,
                shard_id: expert.shard_id,
                device_id: expert.device_id,
                token_indices: batch.token_indices.clone(),
                prefetch_groups: expert.prefetch_groups.clone(),
            });
        }
        Ok(out)
    }
}

#[derive(Debug)]
struct LoadedLayer {
    router: Matrix,
    experts: HashMap<u32, ExpertWeights>,
    output_width: usize,
}

#[derive(Debug)]
struct ExpertBuilder {
    up_name: Option<String>,
    down_name: Option<String>,
    gate_name: Option<String>,
}

#[derive(Debug)]
struct ExpertWeights {
    up: Matrix,
    down: Matrix,
    gate: Option<Matrix>,
    shard_id: u64,
    device_id: u32,
    prefetch_groups: Vec<String>,
}

#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

fn checked_token_count(input: &[f32], input_width: usize) -> Result<usize> {
    if input_width == 0 {
        return Err(MoeRuntimeError::Shape(
            "input_width must be greater than zero".to_string(),
        ));
    }
    if input.len() % input_width != 0 {
        return Err(MoeRuntimeError::Shape(format!(
            "input length {} is not divisible by input_width {input_width}",
            input.len()
        )));
    }
    Ok(input.len() / input_width)
}

fn route_top1(router: &Matrix, input: &[f32], token_count: usize) -> Result<Vec<u32>> {
    let mut assignments = Vec::with_capacity(token_count);
    for token_idx in 0..token_count {
        let token = &input[token_idx * router.cols..(token_idx + 1) * router.cols];
        let mut best_expert = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for expert_id in 0..router.rows {
            let row = router.row(expert_id);
            let score = dot(row, token)?;
            if score > best_score {
                best_score = score;
                best_expert = expert_id;
            }
        }
        assignments.push(best_expert as u32);
    }
    Ok(assignments)
}

fn eval_expert(
    input: &[f32],
    expert: &ExpertWeights,
    activation: ExpertActivation,
) -> Result<Vec<f32>> {
    let up = mat_vec(&expert.up, input)?;
    let hidden = match activation {
        ExpertActivation::Identity => up,
        ExpertActivation::SiluGated => {
            let gate = expert
                .gate
                .as_ref()
                .ok_or_else(|| MoeRuntimeError::Missing("gate tensor".to_string()))?;
            let gate_values = mat_vec(gate, input)?;
            up.into_iter()
                .zip(gate_values)
                .map(|(up, gate)| up * silu(gate))
                .collect()
        }
    };
    mat_vec(&expert.down, &hidden)
}

fn mat_vec(matrix: &Matrix, input: &[f32]) -> Result<Vec<f32>> {
    if matrix.cols != input.len() {
        return Err(MoeRuntimeError::Shape(format!(
            "mat_vec input width {} does not match matrix cols {}",
            input.len(),
            matrix.cols
        )));
    }
    let mut out = vec![0.0; matrix.rows];
    for (row_idx, out_value) in out.iter_mut().enumerate() {
        *out_value = dot(matrix.row(row_idx), input)?;
    }
    Ok(out)
}

fn dot(lhs: &[f32], rhs: &[f32]) -> Result<f32> {
    if lhs.len() != rhs.len() {
        return Err(MoeRuntimeError::Shape(format!(
            "dot length mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        )));
    }
    Ok(lhs.iter().zip(rhs).map(|(a, b)| a * b).sum())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Matrix {
    fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }
}

fn load_matrix(file: &RsmfFile, tensor_name: &str) -> Result<Matrix> {
    let view = file.tensor_view(tensor_name)?;
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(MoeRuntimeError::Shape(format!(
            "{tensor_name} must be rank-2, got rank {}",
            shape.len()
        )));
    }
    let rows = usize::try_from(shape[0]).map_err(|_| {
        MoeRuntimeError::Shape(format!("{tensor_name} row count exceeds usize::MAX"))
    })?;
    let cols = usize::try_from(shape[1]).map_err(|_| {
        MoeRuntimeError::Shape(format!("{tensor_name} column count exceeds usize::MAX"))
    })?;
    let data = view.decode_f32()?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| MoeRuntimeError::Shape(format!("{tensor_name} element count overflow")))?;
    if data.len() != expected {
        return Err(MoeRuntimeError::Shape(format!(
            "{tensor_name} decoded {} values, expected {expected}",
            data.len()
        )));
    }
    Ok(Matrix { rows, cols, data })
}

fn shared_expert_shard(file: &RsmfFile, tensor_names: &[&str]) -> Result<u64> {
    let mut shard_id = None;
    for tensor_name in tensor_names {
        let tensor = tensor_descriptor(file, tensor_name)?;
        match shard_id {
            Some(existing) if existing != tensor.shard_id => {
                return Err(MoeRuntimeError::Shape(format!(
                    "expert tensors span multiple shards: {existing} and {}",
                    tensor.shard_id
                )));
            }
            _ => shard_id = Some(tensor.shard_id),
        }
    }
    shard_id.ok_or_else(|| MoeRuntimeError::Missing("expert shard id".to_string()))
}

fn tensor_descriptor<'a>(file: &'a RsmfFile, tensor_name: &str) -> Result<&'a TensorDescriptor> {
    file.manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == tensor_name)
        .ok_or_else(|| MoeRuntimeError::Missing(format!("tensor {tensor_name}")))
}

fn select_backend(placement: &PlacementManifest, prefer_wgpu: bool) -> RuntimeBackend {
    if !prefer_wgpu {
        return RuntimeBackend::CpuFallback {
            reason: "prefer_wgpu=false".to_string(),
        };
    }

    let requested_devices = placement
        .devices
        .iter()
        .filter(|device| device.kind == DeviceKind::Wgpu)
        .count();

    #[cfg(feature = "wgpu")]
    {
        let available_adapters = usize::from(rsmf_wgpu::detect_capabilities().is_some());
        if available_adapters == 0 {
            RuntimeBackend::CpuFallback {
                reason: format!(
                    "requested {requested_devices} WGPU devices but no adapter is available"
                ),
            }
        } else if requested_devices > available_adapters {
            RuntimeBackend::CpuFallback {
                reason: format!(
                    "requested {requested_devices} WGPU placement devices but only {available_adapters} adapter was detected"
                ),
            }
        } else {
            RuntimeBackend::WgpuProbe {
                requested_devices,
                available_adapters,
            }
        }
    }

    #[cfg(not(feature = "wgpu"))]
    {
        RuntimeBackend::CpuFallback {
            reason: format!(
                "requested {requested_devices} WGPU placement devices but rsmf-moe-runtime was built without the wgpu feature"
            ),
        }
    }
}
