//! Variant selection: execution mode, capabilities, scoring.

use crate::error::{Result, RsmfError};
use crate::manifest::Manifest;
use crate::tensor::descriptor::TensorDescriptor;
use crate::tensor::variant::{EncodingKind, TargetTag, VariantDescriptor};

/// Requested execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Select CPU-friendly variants, falling back to canonical when no packed
    /// CPU variant is available.
    CpuOnly,
    /// Select GPU-friendly variants, falling back to canonical.
    GpuOnly,
    /// Prefer the highest-scoring variant across both CPU and GPU.
    HybridAuto,
}

/// CPU feature flags detected on the host.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    /// AVX2 available on x86_64.
    pub avx2: bool,
    /// AVX-512 available on x86_64.
    pub avx512: bool,
    /// ARMv8 NEON available.
    pub neon: bool,
}

impl CpuFeatures {
    /// Detect CPU features on the current host.
    ///
    /// Best-effort: on platforms other than x86_64 / aarch64 all flags default
    /// to `false`.
    #[must_use]
    pub fn detect() -> Self {
        let mut f = Self::default();
        #[cfg(target_arch = "x86_64")]
        {
            f.avx2 = is_x86_feature_detected!("avx2");
            f.avx512 = is_x86_feature_detected!("avx512f");
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on aarch64 per the ABI.
            f.neon = true;
        }
        f
    }
}

/// Identifier for an available GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Portable WebGPU/WGPU backend.
    Wgpu,
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
}

/// Runtime capabilities for variant selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// Detected CPU features.
    pub cpu: CpuFeatures,
    /// Available GPU backend, if any.
    pub gpu: Option<GpuBackend>,
    /// Largest alignment the runtime can satisfy (in bytes). Variants that
    /// require a stricter alignment are treated as unselectable.
    pub max_alignment: u64,
}

impl Capabilities {
    /// Detect host capabilities with sensible defaults (no GPU, 4 KiB max
    /// alignment).
    #[must_use]
    pub fn detect() -> Self {
        Self {
            cpu: CpuFeatures::detect(),
            gpu: None,
            max_alignment: 4096,
        }
    }

    /// Builder helper: set the preferred GPU backend.
    #[must_use]
    pub fn with_gpu(mut self, gpu: Option<GpuBackend>) -> Self {
        self.gpu = gpu;
        self
    }

    /// Builder helper: override max alignment.
    #[must_use]
    pub fn with_max_alignment(mut self, max: u64) -> Self {
        self.max_alignment = max;
        self
    }
}

/// One entry in a [`TensorPlan`]: which variant was selected for which tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedVariant {
    /// Name of the tensor.
    pub tensor_name: String,
    /// Index of the selected variant in the manifest's variants array.
    pub variant_index: u32,
    /// Target tag of the selected variant.
    pub target: TargetTag,
    /// Encoding kind of the selected variant.
    pub encoding: EncodingKind,
    /// Score assigned by the selector; useful for diagnostics.
    pub score: i32,
}

/// A materialised variant selection plan for every tensor in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorPlan {
    /// One selection per tensor, in the manifest's tensor order.
    pub selections: Vec<SelectedVariant>,
}

impl TensorPlan {
    /// Total number of distinct target tags used in the plan. Useful for
    /// inspect output.
    #[must_use]
    pub fn targets_used(&self) -> Vec<TargetTag> {
        let mut out: Vec<TargetTag> = Vec::new();
        for s in &self.selections {
            if !out.contains(&s.target) {
                out.push(s.target);
            }
        }
        out
    }
}

/// Compute a variant plan from a parsed manifest + mode + capabilities.
///
/// # Correctness
///
/// Every tensor will receive a selection because the canonical variant is
/// always scored as a valid fallback.
pub fn select_variants(
    manifest: &Manifest,
    mode: ExecutionMode,
    caps: &Capabilities,
) -> Result<TensorPlan> {
    let mut out = TensorPlan::default();
    for tensor in &manifest.tensors {
        let selection = pick_for_tensor(manifest, tensor, mode, caps)?;
        out.selections.push(selection);
    }
    Ok(out)
}

fn pick_for_tensor<'a>(
    manifest: &'a Manifest,
    tensor: &TensorDescriptor,
    mode: ExecutionMode,
    caps: &Capabilities,
) -> Result<SelectedVariant> {
    let mut best: Option<(i32, u32, &'a VariantDescriptor)> = None;

    // Canonical variant is always a candidate.
    if let Some(v) = manifest.variants.get(tensor.canonical_variant as usize) {
        update_best(&mut best, tensor.canonical_variant, v, mode, caps);
    }
    for &idx in &tensor.packed_variants {
        if let Some(v) = manifest.variants.get(idx as usize) {
            update_best(&mut best, idx, v, mode, caps);
        }
    }

    let (score, idx, v) = best.ok_or_else(|| {
        RsmfError::not_found(format!(
            "tensor {} missing at least a canonical variant",
            tensor.name
        ))
    })?;
    Ok(SelectedVariant {
        tensor_name: tensor.name.clone(),
        variant_index: idx,
        target: v.target,
        encoding: v.encoding,
        score,
    })
}

fn update_best<'a>(
    best: &mut Option<(i32, u32, &'a VariantDescriptor)>,
    idx: u32,
    v: &'a VariantDescriptor,
    mode: ExecutionMode,
    caps: &Capabilities,
) {
    let score = score_variant(v, mode, caps);
    let better = match *best {
        None => true,
        Some((bs, _, bv)) => {
            score > bs
                || (score == bs && encoding_penalty(v.encoding) < encoding_penalty(bv.encoding))
                || (score == bs
                    && encoding_penalty(v.encoding) == encoding_penalty(bv.encoding)
                    && v.alignment < bv.alignment)
        }
    };
    if better {
        *best = Some((score, idx, v));
    }
}

fn score_variant(v: &VariantDescriptor, mode: ExecutionMode, caps: &Capabilities) -> i32 {
    let base = base_score(v.target, mode, caps);
    let align_pen: i32 = if u64::from(v.alignment) > caps.max_alignment {
        10_000
    } else {
        0
    };
    let enc_pen = encoding_penalty(v.encoding);
    base - align_pen - enc_pen
}

fn encoding_penalty(e: EncodingKind) -> i32 {
    match e {
        EncodingKind::Raw => 0,
        EncodingKind::CastF16 => 3,
        EncodingKind::BlockQuantized => 5, // More lossy than CastF16
    }
}

#[allow(clippy::too_many_lines)]
fn base_score(t: TargetTag, mode: ExecutionMode, caps: &Capabilities) -> i32 {
    let (cpu_only, gpu_only, hybrid) = match t {
        TargetTag::Canonical => (10, 5, 7),
        TargetTag::CpuGeneric => (20, 0, 18),
        TargetTag::CpuAvx2 => (
            if caps.cpu.avx2 { 25 } else { 0 },
            0,
            if caps.cpu.avx2 { 23 } else { 0 },
        ),
        TargetTag::CpuAvx512 => (
            if caps.cpu.avx512 { 28 } else { 0 },
            0,
            if caps.cpu.avx512 { 26 } else { 0 },
        ),
        TargetTag::CpuNeon => (
            if caps.cpu.neon { 25 } else { 0 },
            0,
            if caps.cpu.neon { 23 } else { 0 },
        ),
        TargetTag::Wgpu => {
            let ok = caps.gpu.is_some_and(|g| matches!(g, GpuBackend::Wgpu));
            let ok_hybrid = caps.gpu.is_some();
            (0, if ok { 25 } else { 0 }, if ok_hybrid { 24 } else { 0 })
        }
        TargetTag::Cuda => {
            let ok = caps.gpu.is_some_and(|g| matches!(g, GpuBackend::Cuda));
            (0, if ok { 28 } else { 0 }, if ok { 26 } else { 0 })
        }
        TargetTag::Metal => {
            let ok = caps.gpu.is_some_and(|g| matches!(g, GpuBackend::Metal));
            (0, if ok { 28 } else { 0 }, if ok { 26 } else { 0 })
        }
        // Variants for target tags that don't yet have a matching
        // `GpuBackend` / CPU-feature flag score 0 across the board so
        // they never get picked by mistake. They're still addressable
        // explicitly via `tensor_view_variant(name, idx)` so callers who
        // know better can opt in.
        TargetTag::Vulkan | TargetTag::RocmHip | TargetTag::Tpu => (0, 0, 0),
        TargetTag::CpuSve | TargetTag::CpuRiscvV => (0, 0, 0),
    };
    match mode {
        ExecutionMode::CpuOnly => cpu_only,
        ExecutionMode::GpuOnly => gpu_only,
        ExecutionMode::HybridAuto => hybrid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::CHECKSUM_LEN;
    use crate::section::SectionKind;
    use crate::tensor::descriptor::TensorDescriptor;
    use crate::tensor::dtype::{LogicalDtype, StorageDtype};
    use crate::tensor::variant::{LayoutTag, VariantMeta};

    fn variant(target: TargetTag, enc: EncodingKind) -> VariantDescriptor {
        VariantDescriptor {
            target,
            encoding: enc,
            storage_dtype: StorageDtype::Logical(LogicalDtype::F32),
            layout: LayoutTag::RowMajor,
            alignment: 64,
            section_relative_offset: 0,
            length: 16,
            checksum: [0u8; CHECKSUM_LEN],
            section_kind: SectionKind::CanonicalArena.to_raw() as u8,
            section_index: 0,
            meta: VariantMeta::default(),
        }
    }

    #[test]
    fn canonical_fallback_when_nothing_fits() {
        let canonical = variant(TargetTag::Canonical, EncodingKind::Raw);
        let tensor = TensorDescriptor {
            name: "w".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],
            canonical_variant: 0,
            packed_variants: vec![],
            shard_id: 0,
            metadata: vec![],
        };
        let m = Manifest {
            tensors: vec![tensor],
            variants: vec![canonical],
            ..Manifest::default()
        };
        let plan = select_variants(
            &m,
            ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(None),
        )
        .unwrap();
        assert_eq!(plan.selections.len(), 1);
        assert_eq!(plan.selections[0].target, TargetTag::Canonical);
    }

    #[test]
    fn gpu_mode_prefers_wgpu_variant_when_available() {
        let canonical = variant(TargetTag::Canonical, EncodingKind::Raw);
        let wgpu_variant = variant(TargetTag::Wgpu, EncodingKind::Raw);
        let tensor = TensorDescriptor {
            name: "w".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],
            canonical_variant: 0,
            packed_variants: vec![1],
            shard_id: 0,
            metadata: vec![],
        };
        let m = Manifest {
            tensors: vec![tensor],
            variants: vec![canonical, wgpu_variant],
            ..Manifest::default()
        };
        let plan = select_variants(
            &m,
            ExecutionMode::GpuOnly,
            &Capabilities::detect().with_gpu(Some(GpuBackend::Wgpu)),
        )
        .unwrap();
        assert_eq!(plan.selections[0].target, TargetTag::Wgpu);
    }

    #[test]
    fn cpu_mode_prefers_cpu_generic_over_canonical() {
        let canonical = variant(TargetTag::Canonical, EncodingKind::Raw);
        let cpu = variant(TargetTag::CpuGeneric, EncodingKind::Raw);
        let tensor = TensorDescriptor {
            name: "w".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],
            canonical_variant: 0,
            packed_variants: vec![1],
            shard_id: 0,
            metadata: vec![],
        };
        let m = Manifest {
            tensors: vec![tensor],
            variants: vec![canonical, cpu],
            ..Manifest::default()
        };
        let plan = select_variants(&m, ExecutionMode::CpuOnly, &Capabilities::detect()).unwrap();
        assert_eq!(plan.selections[0].target, TargetTag::CpuGeneric);
    }

    #[test]
    fn alignment_pathological_falls_back() {
        let canonical = variant(TargetTag::Canonical, EncodingKind::Raw);
        let mut huge = variant(TargetTag::CpuGeneric, EncodingKind::Raw);
        huge.alignment = 65536;
        let tensor = TensorDescriptor {
            name: "w".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],
            canonical_variant: 0,
            packed_variants: vec![1],
            shard_id: 0,
            metadata: vec![],
        };
        let m = Manifest {
            tensors: vec![tensor],
            variants: vec![canonical, huge],
            ..Manifest::default()
        };
        let plan = select_variants(
            &m,
            ExecutionMode::CpuOnly,
            &Capabilities::detect().with_max_alignment(64),
        )
        .unwrap();
        assert_eq!(plan.selections[0].target, TargetTag::Canonical);
    }
}
