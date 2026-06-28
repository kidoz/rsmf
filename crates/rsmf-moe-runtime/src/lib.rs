//! Placement-aware expert-parallel runtime for RSMF MoE files.
//!
//! The crate consumes the metadata added in the MoE, placement, sharding, tier,
//! and prefetch milestones. It prepares resident validated layer plans,
//! performs host-side top-1 gating, batches tokens by destination expert and
//! placement device, runs per-expert F32 matmuls, and exposes a measured
//! single-device reference check.
//!
//! GPU execution is feature-gated behind `wgpu`. In the default build the
//! runtime reports a CPU fallback and still exercises the same routing,
//! placement, and sharded-read contracts. With the feature enabled and an
//! adapter available, expert matmuls run through a small WGPU compute shader.
//! Tensor-parallel collectives are not implemented yet and are reported
//! explicitly in prepared layer plans.
//!
//! ```
//! use rsmf_moe_runtime::batch_by_destination;
//!
//! let batches = batch_by_destination(&[1, 0, 1, 2]);
//! assert_eq!(batches[0].expert_id, 1);
//! assert_eq!(batches[0].token_indices, vec![0, 2]);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

use std::time::Duration;

use rsmf_core::RsmfError;

mod routing;
mod runtime;
#[cfg(feature = "wgpu")]
mod wgpu_compute;

pub use routing::{RoutingBatch, batch_by_destination};
pub use runtime::{
    CpuCollectives, DeviceRunReport, ExpertActivation, MoeCheckedRunOutput, MoeCollectiveKind,
    MoeCollectivePlan, MoeCollectiveStep, MoeLayerPlan, MoeLayerPlanReport, MoeReferenceComparison,
    MoeRoutingPolicy, MoeRuntime, MoeRuntimeOptions, MoeTransferKind, MoeTransferPlan,
    MoeTransferStep, MultiAdapterStatus, PlannedDevice, PlannedExpert, RuntimeLimits,
    TensorParallelismStatus,
};

/// Result alias for the MoE runtime crate.
pub type Result<T> = std::result::Result<T, MoeRuntimeError>;

/// Errors returned by the MoE runtime.
#[derive(Debug, thiserror::Error)]
pub enum MoeRuntimeError {
    /// Error propagated from `rsmf-core`.
    #[error(transparent)]
    Core(#[from] RsmfError),
    /// Required MoE metadata, placement metadata, or tensor is missing.
    #[error("missing runtime input: {0}")]
    Missing(String),
    /// Tensor shapes or input dimensions are inconsistent.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// Requested runtime mode is unsupported by this build.
    #[error("unsupported runtime mode: {0}")]
    Unsupported(String),
}

/// Backend used for a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeBackend {
    /// CPU fallback path. This is the default and is always available.
    CpuFallback {
        /// Human-readable reason for the fallback.
        reason: String,
    },
    /// WGPU was requested and a compute device was selected. The current WGPU
    /// executor uses one physical adapter as a logical executor for the
    /// placement devices recorded in the file when multiple physical adapters
    /// are not available.
    WgpuCompute {
        /// Number of WGPU placement devices requested by the placement manifest.
        requested_devices: usize,
        /// Number of WGPU adapters observed by the probe.
        available_adapters: usize,
        /// Human-readable adapter name.
        adapter_name: String,
    },
}

/// One expert batch after host-side routing and placement lookup.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBatch {
    /// Expert id selected by the router.
    pub expert_id: u32,
    /// Tensor shard id that owns this expert.
    pub shard_id: u64,
    /// Placement primary device id for `shard_id`.
    pub device_id: u32,
    /// Token indices routed to this expert.
    pub token_indices: Vec<usize>,
    /// Per-token combine weights aligned with `token_indices`.
    pub token_weights: Vec<f32>,
    /// Prefetch groups associated with this expert's variants.
    pub prefetch_groups: Vec<String>,
}

/// Timing and routing information from one MoE layer run.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeRunReport {
    /// Backend status for this run.
    pub backend: RuntimeBackend,
    /// Prepared layer placement and residency summary.
    pub plan: MoeLayerPlanReport,
    /// Number of input tokens.
    pub token_count: usize,
    /// Input model width.
    pub input_width: usize,
    /// Output model width.
    pub output_width: usize,
    /// Token batches grouped by destination expert/device.
    pub device_batches: Vec<DeviceBatch>,
    /// Per-device execution metrics.
    pub device_runs: Vec<DeviceRunReport>,
    /// Host gating time.
    pub gating_time: Duration,
    /// Dispatch grouping and placement lookup time.
    pub dispatch_time: Duration,
    /// Expert matmul time.
    pub compute_time: Duration,
    /// Output combine/scatter time.
    pub combine_time: Duration,
}

impl MoeRunReport {
    /// Tokens per second based on the sum of measured pipeline stages.
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        let total = self.gating_time + self.dispatch_time + self.compute_time + self.combine_time;
        let secs = total.as_secs_f64();
        if secs == 0.0 {
            return f64::INFINITY;
        }
        self.token_count as f64 / secs
    }
}

/// Output of one MoE layer run.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeRunOutput {
    /// Row-major output tensor, shape `[token_count, output_width]`.
    pub output: Vec<f32>,
    /// Number of output columns per token.
    pub output_width: usize,
    /// Run report with timing and routing details.
    pub report: MoeRunReport,
}
