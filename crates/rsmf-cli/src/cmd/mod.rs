//! CLI subcommand dispatch.

use clap::Subcommand;

pub mod error;
pub mod extract;
pub mod extract_asset;
pub mod import;
pub mod inspect;
pub mod pack;
pub mod rewrite;
pub mod select;
pub mod verify;

use anyhow::anyhow;
pub use error::CliError;
use rsmf_core::TargetTag;

/// Exit statuses returned by the CLI.
pub const EXIT_USER_ERROR: i32 = 1;
pub const EXIT_INTEGRITY: i32 = 2;

/// `rsmf` subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Import a model directly from HuggingFace.
    Import(import::Args),
    /// Print a human-readable summary of the file preamble and manifest.
    Inspect(inspect::Args),
    /// Verify a file's structural integrity, and optionally its checksums.
    Verify(verify::Args),
    /// Pack tensors from a source format (currently safetensors, GGUF, NPY) into RSMF.
    Pack(pack::Args),
    /// Extract a single tensor's canonical bytes to a file.
    Extract(extract::Args),
    /// Extract a named asset to a file.
    #[command(name = "extract-asset")]
    ExtractAsset(extract_asset::Args),
    /// Print the variant selection plan for a given execution mode.
    Select(select::Args),
    /// Copy an RSMF file while stripping variants, graphs, assets, or
    /// metadata. Useful for shipping a smaller production artifact from
    /// a dev-time bundle.
    Rewrite(rewrite::Args),
}

impl Command {
    /// Run the selected subcommand.
    pub fn run(self) -> Result<(), CliError> {
        match self {
            Self::Import(args) => import::run(args),
            Self::Inspect(args) => inspect::run(args),
            Self::Verify(args) => verify::run(args),
            Self::Pack(args) => pack::run(args),
            Self::Extract(args) => extract::run(args),
            Self::ExtractAsset(args) => extract_asset::run(args),
            Self::Select(args) => select::run(args),
            Self::Rewrite(args) => rewrite::run(args),
        }
    }
}

pub fn parse_target_tag(s: &str) -> Result<TargetTag, CliError> {
    Ok(match s {
        "cpu_generic" => TargetTag::CpuGeneric,
        "cpu_avx2" => TargetTag::CpuAvx2,
        "cpu_avx512" => TargetTag::CpuAvx512,
        "cpu_neon" => TargetTag::CpuNeon,
        "wgpu" => TargetTag::Wgpu,
        "cuda" => TargetTag::Cuda,
        "metal" => TargetTag::Metal,
        other => {
            return Err(CliError::user(anyhow!(
                "unknown target tag '{other}'; expected one of: cpu_generic, cpu_avx2, cpu_avx512, cpu_neon, wgpu, cuda, metal"
            )));
        }
    })
}
