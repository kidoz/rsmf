//! `rsmf select` — print the variant selection plan for a mode.

use std::path::PathBuf;

use clap::ValueEnum;
use rsmf_core::{Capabilities, ExecutionMode, GpuBackend, RsmfFile};

use super::CliError;

/// Execution modes exposed through the CLI.
#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "lower")]
pub enum SelectMode {
    /// CPU-only selection.
    Cpu,
    /// GPU-only selection.
    Gpu,
    /// Hybrid automatic selection.
    Hybrid,
}

impl From<SelectMode> for ExecutionMode {
    fn from(m: SelectMode) -> Self {
        match m {
            SelectMode::Cpu => ExecutionMode::CpuOnly,
            SelectMode::Gpu => ExecutionMode::GpuOnly,
            SelectMode::Hybrid => ExecutionMode::HybridAuto,
        }
    }
}

/// Arguments to `rsmf select`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Path to the RSMF file.
    pub file: PathBuf,
    /// Execution mode.
    #[arg(long = "mode", value_enum)]
    pub mode: SelectMode,
    /// Pretend a `wgpu` GPU backend is available when computing the plan
    /// (useful for reproducible output on machines without a real GPU).
    #[arg(long = "assume-wgpu")]
    pub assume_wgpu: bool,
}

/// Execute `rsmf select`.
pub fn run(args: Args) -> Result<(), CliError> {
    let file = RsmfFile::open(&args.file)?;
    let mut caps = Capabilities::detect();
    if args.assume_wgpu {
        caps = caps.with_gpu(Some(GpuBackend::Wgpu));
    }
    let plan = file.select_variants(args.mode.into(), &caps)?;
    println!(
        "mode={mode:?} gpu={gpu:?} avx2={avx2} avx512={avx512} neon={neon}",
        mode = args.mode,
        gpu = caps.gpu,
        avx2 = caps.cpu.avx2,
        avx512 = caps.cpu.avx512,
        neon = caps.cpu.neon,
    );
    for s in &plan.selections {
        println!(
            "  {name:<32} -> target={target:<10} encoding={encoding} score={score}",
            name = s.tensor_name,
            target = s.target.name(),
            encoding = s.encoding.name(),
            score = s.score,
        );
    }
    Ok(())
}
