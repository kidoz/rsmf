//! Runtime planner and F32 MoE layer execution.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::time::{Duration, Instant};

use rsmf_core::{
    DeviceKind, MemoryTier, MoeRole, PlacementManifest, PrefetchIndex, RsmfFile, TensorDescriptor,
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
    /// Enforce non-zero placement device capacity values against the decoded
    /// resident expert tensors planned for that device.
    pub enforce_device_capacity: bool,
    /// Runtime resource limits checked before large allocations.
    pub limits: RuntimeLimits,
}

impl Default for MoeRuntimeOptions {
    fn default() -> Self {
        Self {
            prefer_wgpu: false,
            activation: ExpertActivation::Identity,
            enforce_device_capacity: true,
            limits: RuntimeLimits::default(),
        }
    }
}

/// Resource limits for one MoE layer invocation.
///
/// These guardrails reject malformed or unexpectedly large inputs before the
/// runtime allocates output buffers or decoded matrix storage. Set a field to
/// `None` to disable that specific guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeLimits {
    /// Maximum number of token rows accepted by one run.
    pub max_tokens: Option<usize>,
    /// Maximum number of placement devices accepted by one prepared layer.
    pub max_devices: Option<usize>,
    /// Maximum number of routed experts accepted by one prepared layer.
    pub max_experts: Option<usize>,
    /// Maximum decoded elements accepted for one rank-2 tensor.
    pub max_tensor_elements: Option<usize>,
    /// Maximum decoded resident expert bytes accepted for one placement
    /// device. This is independent from `PlacementManifest` device capacity.
    pub max_resident_bytes_per_device: Option<usize>,
    /// Maximum output elements allocated for one run.
    pub max_output_elements: Option<usize>,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_tokens: Some(1_048_576),
            max_devices: Some(1024),
            max_experts: Some(65_536),
            max_tensor_elements: Some(1_073_741_824),
            max_resident_bytes_per_device: None,
            max_output_elements: Some(1_073_741_824),
        }
    }
}

impl RuntimeLimits {
    fn check_tokens(&self, token_count: usize) -> Result<()> {
        if let Some(limit) = self.max_tokens
            && token_count > limit
        {
            return Err(MoeRuntimeError::Shape(format!(
                "token count {token_count} exceeds runtime limit {limit}"
            )));
        }
        Ok(())
    }

    fn check_devices(&self, device_count: usize) -> Result<()> {
        if let Some(limit) = self.max_devices
            && device_count > limit
        {
            return Err(MoeRuntimeError::Shape(format!(
                "prepared layer uses {device_count} placement devices, exceeding runtime limit {limit}"
            )));
        }
        Ok(())
    }

    fn check_experts(&self, expert_count: usize) -> Result<()> {
        if let Some(limit) = self.max_experts
            && expert_count > limit
        {
            return Err(MoeRuntimeError::Shape(format!(
                "prepared layer has {expert_count} experts, exceeding runtime limit {limit}"
            )));
        }
        Ok(())
    }

    fn check_tensor_elements(&self, tensor_name: &str, elements: usize) -> Result<()> {
        if let Some(limit) = self.max_tensor_elements
            && elements > limit
        {
            return Err(MoeRuntimeError::Shape(format!(
                "{tensor_name} has {elements} elements, exceeding runtime limit {limit}"
            )));
        }
        Ok(())
    }

    fn check_resident_device_bytes(&self, device_id: u32, bytes: usize) -> Result<()> {
        if let Some(limit) = self.max_resident_bytes_per_device
            && bytes > limit
        {
            return Err(MoeRuntimeError::Shape(format!(
                "placement device {device_id} has {bytes} resident expert bytes, exceeding runtime limit {limit}"
            )));
        }
        Ok(())
    }

    fn check_output_elements(&self, elements: usize) -> Result<()> {
        if let Some(limit) = self.max_output_elements
            && elements > limit
        {
            return Err(MoeRuntimeError::Shape(format!(
                "output has {elements} elements, exceeding runtime limit {limit}"
            )));
        }
        Ok(())
    }
}

/// One expert in a prepared MoE layer plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedExpert {
    /// Expert id selected by router output rows.
    pub expert_id: u32,
    /// Tensor shard id shared by this expert's resident tensors.
    pub shard_id: u64,
    /// Primary placement device id that owns this expert shard.
    pub device_id: u32,
    /// Up projection tensor name.
    pub up_tensor: String,
    /// Down projection tensor name.
    pub down_tensor: String,
    /// Optional gate projection tensor name.
    pub gate_tensor: Option<String>,
    /// Decoded f32 bytes resident in this prepared plan for the expert.
    pub resident_bytes: usize,
    /// Prefetch groups associated with this expert's tensor variants.
    pub prefetch_groups: Vec<String>,
}

/// Placement device in a prepared MoE layer plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedDevice {
    /// Placement device id.
    pub device_id: u32,
    /// Placement backend kind.
    pub kind: DeviceKind,
    /// Placement memory tier.
    pub tier: MemoryTier,
    /// Capacity from the placement manifest. `0` means unknown.
    pub capacity_bytes: u64,
    /// Decoded f32 expert bytes resident in this prepared plan for the device.
    pub resident_bytes: usize,
    /// Expert ids assigned to this device.
    pub expert_ids: Vec<u32>,
    /// Shard ids assigned to this device.
    pub shard_ids: Vec<u64>,
    /// Prefetch groups used by experts assigned to this device.
    pub prefetch_groups: Vec<String>,
}

/// Prepared layer placement and residency summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeLayerPlanReport {
    /// MoE layer index.
    pub layer: u32,
    /// Input width accepted by this plan.
    pub input_width: usize,
    /// Output width produced by each token.
    pub output_width: usize,
    /// Number of routed experts in the layer.
    pub expert_count: usize,
    /// Total decoded f32 expert bytes resident in this plan.
    pub resident_bytes: usize,
    /// Planned experts.
    pub experts: Vec<PlannedExpert>,
    /// Planned placement devices.
    pub devices: Vec<PlannedDevice>,
    /// True when experts are placed on more than one primary device.
    pub multi_device: bool,
    /// Human-readable tensor-parallelism status for this plan.
    pub tensor_parallelism: TensorParallelismStatus,
}

/// Tensor-parallel execution status for a prepared MoE layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorParallelismStatus {
    /// The prepared plan does not require tensor parallelism because each expert
    /// is owned by one shard/device.
    NotRequired,
    /// Tensor-parallel collectives are not implemented by this runtime yet.
    Unavailable {
        /// Human-readable reason.
        reason: String,
    },
}

/// Resident, validated MoE layer plan.
#[derive(Debug)]
pub struct MoeLayerPlan {
    layer: u32,
    input_width: usize,
    loaded: LoadedLayer,
    report: MoeLayerPlanReport,
}

impl MoeLayerPlan {
    /// MoE layer index.
    #[must_use]
    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Input width accepted by this prepared plan.
    #[must_use]
    pub fn input_width(&self) -> usize {
        self.input_width
    }

    /// Borrow the immutable plan report.
    #[must_use]
    pub fn report(&self) -> &MoeLayerPlanReport {
        &self.report
    }
}

/// Per-device execution metrics captured during a routed run.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceRunReport {
    /// Placement device id.
    pub device_id: u32,
    /// Experts executed on this device during the run.
    pub expert_ids: Vec<u32>,
    /// Number of expert batches executed on this device.
    pub batch_count: usize,
    /// Number of token rows processed by this device.
    pub token_count: usize,
    /// Compute time spent while executing this device's batches.
    pub compute_time: Duration,
}

/// Reference comparison and timing for a checked routed run.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeReferenceComparison {
    /// Maximum absolute difference between routed and reference outputs.
    pub max_abs_diff: f32,
    /// Allowed absolute tolerance.
    pub tolerance: f32,
    /// True when all output elements are within `tolerance`.
    pub passed: bool,
    /// Wall time for routed execution.
    pub routed_wall_time: Duration,
    /// Wall time for single-device reference execution.
    pub reference_wall_time: Duration,
    /// `reference_wall_time / routed_wall_time`.
    pub speedup: f64,
}

/// Output of a routed run checked against the single-device reference path.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeCheckedRunOutput {
    /// Placement-aware routed output.
    pub routed: crate::MoeRunOutput,
    /// Single-device reference output.
    pub reference_output: Vec<f32>,
    /// Numerical and timing comparison.
    pub comparison: MoeReferenceComparison,
}

/// Placement-aware MoE runtime over an opened RSMF file.
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
    #[cfg(feature = "wgpu")]
    wgpu: Option<crate::wgpu_compute::WgpuExecutor>,
}

impl MoeRuntime {
    /// Build a runtime from an opened RSMF file.
    ///
    /// Returns an error if the file has no placement manifest, because the
    /// runtime uses placement records to map expert shards to logical devices.
    pub fn new(file: RsmfFile, options: MoeRuntimeOptions) -> Result<Self> {
        let placement = file
            .placement_manifest()?
            .ok_or_else(|| MoeRuntimeError::Missing("PlacementManifest".to_string()))?;
        let prefetch = file.prefetch_hints()?;
        #[cfg(feature = "wgpu")]
        let (backend, wgpu) = select_backend(&placement, options.prefer_wgpu);
        #[cfg(not(feature = "wgpu"))]
        let backend = select_backend(&placement, options.prefer_wgpu);
        Ok(Self {
            file,
            placement,
            prefetch,
            backend,
            options,
            #[cfg(feature = "wgpu")]
            wgpu,
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

    /// Decode and validate one MoE layer into a resident execution plan.
    ///
    /// The returned plan owns decoded f32 router and expert weights, validates
    /// placement capacity/limits, and can be reused across many token batches
    /// with [`Self::run_prepared_layer_top1`].
    pub fn prepare_layer(&self, layer: u32, input_width: usize) -> Result<MoeLayerPlan> {
        let loaded = self.load_layer(layer, input_width)?;
        let report = self.plan_report(layer, input_width, &loaded)?;
        self.options.limits.check_experts(report.expert_count)?;
        self.options.limits.check_devices(report.devices.len())?;
        for device in &report.devices {
            self.options
                .limits
                .check_resident_device_bytes(device.device_id, device.resident_bytes)?;
            if self.options.enforce_device_capacity
                && device.capacity_bytes != 0
                && device.resident_bytes as u128 > u128::from(device.capacity_bytes)
            {
                return Err(MoeRuntimeError::Shape(format!(
                    "placement device {} has {} resident expert bytes, exceeding manifest capacity {}",
                    device.device_id, device.resident_bytes, device.capacity_bytes
                )));
            }
        }
        Ok(MoeLayerPlan {
            layer,
            input_width,
            loaded,
            report,
        })
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
        let plan = self.prepare_layer(layer, input_width)?;
        self.run_prepared_layer_top1(&plan, input)
    }

    /// Run one MoE layer using a prepared resident plan.
    ///
    /// `input` is row-major `[token_count, plan.input_width()]`.
    pub fn run_prepared_layer_top1(
        &self,
        plan: &MoeLayerPlan,
        input: &[f32],
    ) -> Result<MoeRunOutput> {
        let token_count = checked_token_count(input, plan.input_width)?;
        self.options.limits.check_tokens(token_count)?;

        let gating_start = Instant::now();
        let assignments = route_top1(&plan.loaded.router, input, token_count)?;
        let gating_time = gating_start.elapsed();

        let dispatch_start = Instant::now();
        let routing_batches = batch_by_destination(&assignments);
        let device_batches = self.device_batches(&plan.loaded, &routing_batches)?;
        let dispatch_time = dispatch_start.elapsed();

        let compute_start = Instant::now();
        let (output, device_runs) = self.compute_batches(
            &plan.loaded,
            input,
            plan.input_width,
            token_count,
            &device_batches,
        )?;
        let compute_time = compute_start.elapsed();

        let combine_start = Instant::now();
        // Output rows are scattered into their final positions during compute;
        // keep this phase explicit so reports have the same shape as a future
        // device-backed combine path.
        let combine_time = combine_start.elapsed();

        Ok(MoeRunOutput {
            output,
            output_width: plan.loaded.output_width,
            report: MoeRunReport {
                backend: self.backend.clone(),
                plan: plan.report.clone(),
                token_count,
                input_width: plan.input_width,
                output_width: plan.loaded.output_width,
                device_batches,
                device_runs,
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
    /// This is the single-device correctness reference for the routed runtime.
    pub fn run_layer_reference_top1(
        &self,
        layer: u32,
        input: &[f32],
        input_width: usize,
    ) -> Result<Vec<f32>> {
        let plan = self.prepare_layer(layer, input_width)?;
        self.run_prepared_layer_reference_top1(&plan, input)
    }

    /// Run the single-device reference path with a prepared resident plan.
    pub fn run_prepared_layer_reference_top1(
        &self,
        plan: &MoeLayerPlan,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let token_count = checked_token_count(input, plan.input_width)?;
        self.options.limits.check_tokens(token_count)?;
        let assignments = route_top1(&plan.loaded.router, input, token_count)?;
        let total_output_len = output_len(token_count, plan.loaded.output_width)?;
        self.options
            .limits
            .check_output_elements(total_output_len)?;
        let mut output = vec![0.0; total_output_len];
        for (token_idx, expert_id) in assignments.iter().copied().enumerate() {
            let expert = plan
                .loaded
                .experts
                .get(&expert_id)
                .ok_or_else(|| MoeRuntimeError::Missing(format!("expert {expert_id}")))?;
            let input_row =
                &input[token_idx * plan.input_width..(token_idx + 1) * plan.input_width];
            let row = eval_expert(input_row, expert, self.options.activation)?;
            let out_start = token_idx * plan.loaded.output_width;
            output[out_start..out_start + plan.loaded.output_width].copy_from_slice(&row);
        }
        Ok(output)
    }

    /// Run placement-aware top-1 execution and compare it to the single-device
    /// reference output with the provided absolute tolerance.
    pub fn run_layer_top1_checked(
        &self,
        layer: u32,
        input: &[f32],
        input_width: usize,
        tolerance: f32,
    ) -> Result<MoeCheckedRunOutput> {
        let plan = self.prepare_layer(layer, input_width)?;
        self.run_prepared_layer_top1_checked(&plan, input, tolerance)
    }

    /// Run a prepared layer and compare it to the single-device reference path.
    pub fn run_prepared_layer_top1_checked(
        &self,
        plan: &MoeLayerPlan,
        input: &[f32],
        tolerance: f32,
    ) -> Result<MoeCheckedRunOutput> {
        if tolerance.is_sign_negative() || !tolerance.is_finite() {
            return Err(MoeRuntimeError::Shape(format!(
                "reference tolerance must be non-negative and finite, got {tolerance}"
            )));
        }
        let routed_start = Instant::now();
        let routed = self.run_prepared_layer_top1(plan, input)?;
        let routed_wall_time = routed_start.elapsed();
        let reference_start = Instant::now();
        let reference_output = self.run_prepared_layer_reference_top1(plan, input)?;
        let reference_wall_time = reference_start.elapsed();
        let max_abs_diff = max_abs_diff(&routed.output, &reference_output)?;
        Ok(MoeCheckedRunOutput {
            comparison: MoeReferenceComparison {
                max_abs_diff,
                tolerance,
                passed: max_abs_diff <= tolerance,
                routed_wall_time,
                reference_wall_time,
                speedup: speedup(reference_wall_time, routed_wall_time),
            },
            routed,
            reference_output,
        })
    }

    fn load_layer(&self, layer: u32, input_width: usize) -> Result<LoadedLayer> {
        let moe = self.file.moe_experts()?;
        if let Some(top_k) = moe.top_k
            && top_k != 1
        {
            return Err(MoeRuntimeError::Unsupported(format!(
                "run_layer_top1 requires moe.top_k=1, got {top_k}"
            )));
        }

        let mut router_entries = moe
            .entries
            .iter()
            .filter(|entry| entry.layer == layer && entry.role == MoeRole::Router);
        let router_entry = router_entries
            .next()
            .ok_or_else(|| MoeRuntimeError::Missing(format!("layer {layer} router tensor")))?;
        if let Some(duplicate) = router_entries.next() {
            return Err(MoeRuntimeError::Shape(format!(
                "layer {layer} has multiple router tensors: {} and {}",
                router_entry.tensor_name, duplicate.tensor_name
            )));
        }
        if let Some(expert_id) = router_entry.expert_id {
            return Err(MoeRuntimeError::Shape(format!(
                "router tensor {} must not set moe.expert={expert_id}",
                router_entry.tensor_name
            )));
        }
        let router = self.load_matrix(&router_entry.tensor_name)?;
        if router.cols != input_width {
            return Err(MoeRuntimeError::Shape(format!(
                "router {} has input width {}, got {input_width}",
                router_entry.tensor_name, router.cols
            )));
        }
        if router.rows == 0 {
            return Err(MoeRuntimeError::Shape(format!(
                "router {} must have at least one expert row",
                router_entry.tensor_name
            )));
        }
        let router_rows = u32::try_from(router.rows).map_err(|_| {
            MoeRuntimeError::Shape(format!(
                "router {} row count exceeds u32::MAX",
                router_entry.tensor_name
            ))
        })?;
        if let Some(n_experts) = moe.n_experts
            && router_rows != n_experts
        {
            return Err(MoeRuntimeError::Shape(format!(
                "router {} has {router_rows} expert rows, but moe.n_experts={n_experts}",
                router_entry.tensor_name
            )));
        }

        let mut builders: BTreeMap<u32, ExpertBuilder> = BTreeMap::new();
        for entry in moe
            .entries
            .iter()
            .filter(|entry| entry.layer == layer && entry.expert_id.is_some())
        {
            if entry.shared {
                return Err(MoeRuntimeError::Unsupported(format!(
                    "shared MoE experts are not supported by rsmf-moe-runtime: {}",
                    entry.tensor_name
                )));
            }
            let expert_id = entry.expert_id.unwrap_or(0);
            if let Some(n_experts) = moe.n_experts
                && expert_id >= n_experts
            {
                return Err(MoeRuntimeError::Shape(format!(
                    "layer {layer} expert {expert_id} is outside moe.n_experts={n_experts}"
                )));
            }
            let builder = builders.entry(expert_id).or_insert_with(|| ExpertBuilder {
                up_name: None,
                down_name: None,
                gate_name: None,
            });
            match &entry.role {
                MoeRole::Up => set_once(
                    &mut builder.up_name,
                    &entry.tensor_name,
                    "up",
                    layer,
                    expert_id,
                )?,
                MoeRole::Down => set_once(
                    &mut builder.down_name,
                    &entry.tensor_name,
                    "down",
                    layer,
                    expert_id,
                )?,
                MoeRole::Gate => set_once(
                    &mut builder.gate_name,
                    &entry.tensor_name,
                    "gate",
                    layer,
                    expert_id,
                )?,
                _ => {}
            }
        }

        let mut experts = BTreeMap::new();
        let mut output_width = None;
        for (expert_id, builder) in builders {
            let up_name = builder.up_name.ok_or_else(|| {
                MoeRuntimeError::Missing(format!("layer {layer} expert {expert_id} up tensor"))
            })?;
            let down_name = builder.down_name.ok_or_else(|| {
                MoeRuntimeError::Missing(format!("layer {layer} expert {expert_id} down tensor"))
            })?;
            let up = self.load_matrix(&up_name)?;
            let down = self.load_matrix(&down_name)?;
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
            let gate_name = builder.gate_name;
            let gate = if let Some(gate_name) = &gate_name {
                let gate = self.load_matrix(gate_name)?;
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
            let mut shard_tensor_names = vec![up_name.as_str(), down_name.as_str()];
            if let Some(gate_name) = &gate_name {
                shard_tensor_names.push(gate_name.as_str());
            }
            let shard_id = shared_expert_shard(&self.file, &shard_tensor_names)?;
            let placement = self.placement_for_shard(shard_id)?;
            let prefetch_groups = self.prefetch_groups_for_tensors(shard_tensor_names);
            experts.insert(
                expert_id,
                ExpertWeights {
                    up_name,
                    down_name,
                    gate_name,
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
        for expert_id in 0..router_rows {
            if !experts.contains_key(&expert_id) {
                return Err(MoeRuntimeError::Missing(format!(
                    "layer {layer} expert {expert_id} tensors"
                )));
            }
        }
        Ok(LoadedLayer {
            router,
            experts,
            output_width,
        })
    }

    fn plan_report(
        &self,
        layer: u32,
        input_width: usize,
        loaded: &LoadedLayer,
    ) -> Result<MoeLayerPlanReport> {
        let mut experts = Vec::with_capacity(loaded.experts.len());
        let mut devices: BTreeMap<u32, PlannedDeviceBuilder> = BTreeMap::new();
        let mut resident_bytes = 0usize;

        for (&expert_id, expert) in &loaded.experts {
            let expert_bytes = expert.resident_bytes()?;
            resident_bytes = resident_bytes.checked_add(expert_bytes).ok_or_else(|| {
                MoeRuntimeError::Shape("prepared layer resident byte count overflow".to_string())
            })?;
            experts.push(PlannedExpert {
                expert_id,
                shard_id: expert.shard_id,
                device_id: expert.device_id,
                up_tensor: expert.up_name.clone(),
                down_tensor: expert.down_name.clone(),
                gate_tensor: expert.gate_name.clone(),
                resident_bytes: expert_bytes,
                prefetch_groups: expert.prefetch_groups.clone(),
            });

            let descriptor = self.device_descriptor(expert.device_id)?;
            let device = devices
                .entry(expert.device_id)
                .or_insert_with(|| PlannedDeviceBuilder::new(descriptor));
            device.add_expert(
                expert_id,
                expert.shard_id,
                expert_bytes,
                &expert.prefetch_groups,
            )?;
        }

        let devices = devices
            .into_values()
            .map(PlannedDeviceBuilder::finish)
            .collect::<Vec<_>>();
        let multi_device = devices.len() > 1;
        Ok(MoeLayerPlanReport {
            layer,
            input_width,
            output_width: loaded.output_width,
            expert_count: loaded.experts.len(),
            resident_bytes,
            experts,
            devices,
            multi_device,
            tensor_parallelism: TensorParallelismStatus::NotRequired,
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

    fn device_descriptor(&self, device_id: u32) -> Result<&rsmf_core::DeviceDescriptor> {
        self.placement
            .devices
            .iter()
            .find(|device| device.id == device_id)
            .ok_or_else(|| MoeRuntimeError::Missing(format!("placement device {device_id}")))
    }

    fn load_matrix(&self, tensor_name: &str) -> Result<Matrix> {
        let view = self.file.tensor_view(tensor_name)?;
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
        let expected = output_len(rows, cols)
            .map_err(|_| MoeRuntimeError::Shape(format!("{tensor_name} element count overflow")))?;
        self.options
            .limits
            .check_tensor_elements(tensor_name, expected)?;
        let data = view.decode_f32()?;
        if data.len() != expected {
            return Err(MoeRuntimeError::Shape(format!(
                "{tensor_name} decoded {} values, expected {expected}",
                data.len()
            )));
        }
        Ok(Matrix { rows, cols, data })
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

    fn compute_batches(
        &self,
        loaded: &LoadedLayer,
        input: &[f32],
        input_width: usize,
        token_count: usize,
        device_batches: &[DeviceBatch],
    ) -> Result<(Vec<f32>, Vec<DeviceRunReport>)> {
        #[cfg(feature = "wgpu")]
        if let Some(executor) = &self.wgpu {
            return self.compute_batches_wgpu(
                executor,
                loaded,
                input,
                input_width,
                token_count,
                device_batches,
            );
        }
        self.compute_batches_cpu(loaded, input, input_width, token_count, device_batches)
    }

    fn compute_batches_cpu(
        &self,
        loaded: &LoadedLayer,
        input: &[f32],
        input_width: usize,
        token_count: usize,
        device_batches: &[DeviceBatch],
    ) -> Result<(Vec<f32>, Vec<DeviceRunReport>)> {
        let total_output_len = output_len(token_count, loaded.output_width)?;
        self.options
            .limits
            .check_output_elements(total_output_len)?;
        let mut output = vec![0.0; total_output_len];
        let mut device_runs = Vec::new();
        for (device_id, batches) in batches_by_device(device_batches) {
            let device_start = Instant::now();
            let batch_count = batches.len();
            let mut expert_ids = BTreeSet::new();
            let mut device_token_count = 0usize;
            for batch in batches {
                expert_ids.insert(batch.expert_id);
                device_token_count = device_token_count
                    .checked_add(batch.token_indices.len())
                    .ok_or_else(|| {
                        MoeRuntimeError::Shape("device token count overflow".to_string())
                    })?;
                let expert = loaded.experts.get(&batch.expert_id).ok_or_else(|| {
                    MoeRuntimeError::Missing(format!("expert {}", batch.expert_id))
                })?;
                for &token_idx in &batch.token_indices {
                    let input_row = &input[token_idx * input_width..(token_idx + 1) * input_width];
                    let row = eval_expert(input_row, expert, self.options.activation)?;
                    let out_start = token_idx * loaded.output_width;
                    output[out_start..out_start + loaded.output_width].copy_from_slice(&row);
                }
            }
            device_runs.push(DeviceRunReport {
                device_id,
                expert_ids: expert_ids.into_iter().collect(),
                batch_count,
                token_count: device_token_count,
                compute_time: device_start.elapsed(),
            });
        }
        Ok((output, device_runs))
    }

    #[cfg(feature = "wgpu")]
    fn compute_batches_wgpu(
        &self,
        executor: &crate::wgpu_compute::WgpuExecutor,
        loaded: &LoadedLayer,
        input: &[f32],
        input_width: usize,
        token_count: usize,
        device_batches: &[DeviceBatch],
    ) -> Result<(Vec<f32>, Vec<DeviceRunReport>)> {
        let total_output_len = output_len(token_count, loaded.output_width)?;
        self.options
            .limits
            .check_output_elements(total_output_len)?;
        let mut output = vec![0.0; total_output_len];
        let mut device_runs = Vec::new();
        for (device_id, batches) in batches_by_device(device_batches) {
            let device_start = Instant::now();
            let batch_count = batches.len();
            let mut expert_ids = BTreeSet::new();
            let mut device_token_count = 0usize;
            for batch in batches {
                expert_ids.insert(batch.expert_id);
                device_token_count = device_token_count
                    .checked_add(batch.token_indices.len())
                    .ok_or_else(|| {
                        MoeRuntimeError::Shape("device token count overflow".to_string())
                    })?;
                let expert = loaded.experts.get(&batch.expert_id).ok_or_else(|| {
                    MoeRuntimeError::Missing(format!("expert {}", batch.expert_id))
                })?;
                let batch_input_len = output_len(batch.token_indices.len(), input_width)?;
                let mut batch_input = Vec::with_capacity(batch_input_len);
                for &token_idx in &batch.token_indices {
                    batch_input.extend_from_slice(
                        &input[token_idx * input_width..(token_idx + 1) * input_width],
                    );
                }

                let up = executor.matmul(
                    &expert.up.data,
                    expert.up.rows,
                    expert.up.cols,
                    &batch_input,
                    batch.token_indices.len(),
                )?;
                let hidden = match self.options.activation {
                    ExpertActivation::Identity => up,
                    ExpertActivation::SiluGated => {
                        let gate = expert
                            .gate
                            .as_ref()
                            .ok_or_else(|| MoeRuntimeError::Missing("gate tensor".to_string()))?;
                        let gate_values = executor.matmul(
                            &gate.data,
                            gate.rows,
                            gate.cols,
                            &batch_input,
                            batch.token_indices.len(),
                        )?;
                        up.into_iter()
                            .zip(gate_values)
                            .map(|(up, gate)| up * silu(gate))
                            .collect()
                    }
                };
                let batch_output = executor.matmul(
                    &expert.down.data,
                    expert.down.rows,
                    expert.down.cols,
                    &hidden,
                    batch.token_indices.len(),
                )?;
                for (batch_row, &token_idx) in batch.token_indices.iter().enumerate() {
                    let src_start = batch_row * loaded.output_width;
                    let dst_start = token_idx * loaded.output_width;
                    output[dst_start..dst_start + loaded.output_width]
                        .copy_from_slice(&batch_output[src_start..src_start + loaded.output_width]);
                }
            }
            device_runs.push(DeviceRunReport {
                device_id,
                expert_ids: expert_ids.into_iter().collect(),
                batch_count,
                token_count: device_token_count,
                compute_time: device_start.elapsed(),
            });
        }
        Ok((output, device_runs))
    }
}

#[derive(Debug)]
struct LoadedLayer {
    router: Matrix,
    experts: BTreeMap<u32, ExpertWeights>,
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
    up_name: String,
    down_name: String,
    gate_name: Option<String>,
    up: Matrix,
    down: Matrix,
    gate: Option<Matrix>,
    shard_id: u64,
    device_id: u32,
    prefetch_groups: Vec<String>,
}

impl ExpertWeights {
    fn resident_bytes(&self) -> Result<usize> {
        let mut bytes = self.up.byte_len()?;
        bytes = bytes.checked_add(self.down.byte_len()?).ok_or_else(|| {
            MoeRuntimeError::Shape("expert resident byte count overflow".to_string())
        })?;
        if let Some(gate) = &self.gate {
            bytes = bytes.checked_add(gate.byte_len()?).ok_or_else(|| {
                MoeRuntimeError::Shape("expert resident byte count overflow".to_string())
            })?;
        }
        Ok(bytes)
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

#[derive(Debug)]
struct PlannedDeviceBuilder {
    device_id: u32,
    kind: DeviceKind,
    tier: MemoryTier,
    capacity_bytes: u64,
    resident_bytes: usize,
    expert_ids: Vec<u32>,
    shard_ids: BTreeSet<u64>,
    prefetch_groups: Vec<String>,
}

impl PlannedDeviceBuilder {
    fn new(descriptor: &rsmf_core::DeviceDescriptor) -> Self {
        Self {
            device_id: descriptor.id,
            kind: descriptor.kind,
            tier: descriptor.tier,
            capacity_bytes: descriptor.capacity_bytes,
            resident_bytes: 0,
            expert_ids: Vec::new(),
            shard_ids: BTreeSet::new(),
            prefetch_groups: Vec::new(),
        }
    }

    fn add_expert(
        &mut self,
        expert_id: u32,
        shard_id: u64,
        bytes: usize,
        prefetch_groups: &[String],
    ) -> Result<()> {
        self.expert_ids.push(expert_id);
        self.shard_ids.insert(shard_id);
        self.resident_bytes = self.resident_bytes.checked_add(bytes).ok_or_else(|| {
            MoeRuntimeError::Shape("planned device resident byte count overflow".to_string())
        })?;
        for group in prefetch_groups {
            if !self.prefetch_groups.contains(group) {
                self.prefetch_groups.push(group.clone());
            }
        }
        Ok(())
    }

    fn finish(self) -> PlannedDevice {
        PlannedDevice {
            device_id: self.device_id,
            kind: self.kind,
            tier: self.tier,
            capacity_bytes: self.capacity_bytes,
            resident_bytes: self.resident_bytes,
            expert_ids: self.expert_ids,
            shard_ids: self.shard_ids.into_iter().collect(),
            prefetch_groups: self.prefetch_groups,
        }
    }
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

fn output_len(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| MoeRuntimeError::Shape("element count overflow".to_string()))
}

fn batches_by_device(device_batches: &[DeviceBatch]) -> BTreeMap<u32, Vec<&DeviceBatch>> {
    let mut out: BTreeMap<u32, Vec<&DeviceBatch>> = BTreeMap::new();
    for batch in device_batches {
        out.entry(batch.device_id).or_default().push(batch);
    }
    out
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(MoeRuntimeError::Shape(format!(
            "comparison length mismatch: {} vs {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max))
}

fn speedup(reference: Duration, routed: Duration) -> f64 {
    let routed_secs = routed.as_secs_f64();
    if routed_secs == 0.0 {
        return f64::INFINITY;
    }
    reference.as_secs_f64() / routed_secs
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

    fn byte_len(&self) -> Result<usize> {
        self.data
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MoeRuntimeError::Shape("matrix byte length overflow".to_string()))
    }
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

fn set_once(
    slot: &mut Option<String>,
    tensor_name: &str,
    role: &str,
    layer: u32,
    expert_id: u32,
) -> Result<()> {
    if let Some(existing) = slot {
        return Err(MoeRuntimeError::Shape(format!(
            "layer {layer} expert {expert_id} has multiple {role} tensors: {existing} and {tensor_name}"
        )));
    }
    *slot = Some(tensor_name.to_string());
    Ok(())
}

#[cfg(feature = "wgpu")]
fn select_backend(
    placement: &PlacementManifest,
    prefer_wgpu: bool,
) -> (RuntimeBackend, Option<crate::wgpu_compute::WgpuExecutor>) {
    if !prefer_wgpu {
        return (
            RuntimeBackend::CpuFallback {
                reason: "prefer_wgpu=false".to_string(),
            },
            None,
        );
    }

    let requested_devices = placement
        .devices
        .iter()
        .filter(|device| device.kind == DeviceKind::Wgpu)
        .count();

    if requested_devices == 0 {
        return (
            RuntimeBackend::CpuFallback {
                reason: "prefer_wgpu=true but placement has no WGPU devices".to_string(),
            },
            None,
        );
    }

    let Some(executor) = crate::wgpu_compute::WgpuExecutor::new() else {
        return (
            RuntimeBackend::CpuFallback {
                reason: format!(
                    "requested {requested_devices} WGPU devices but no adapter is available"
                ),
            },
            None,
        );
    };
    (
        RuntimeBackend::WgpuCompute {
            requested_devices,
            available_adapters: 1,
            adapter_name: executor.adapter_name().to_string(),
        },
        Some(executor),
    )
}

#[cfg(not(feature = "wgpu"))]
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

    RuntimeBackend::CpuFallback {
        reason: format!(
            "requested {requested_devices} WGPU placement devices but rsmf-moe-runtime was built without the wgpu feature"
        ),
    }
}
