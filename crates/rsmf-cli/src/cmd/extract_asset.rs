//! `rsmf extract-asset` — extract an asset by name to a file.

use std::path::PathBuf;

use anyhow::anyhow;
use rsmf_core::RsmfFile;

use super::CliError;

/// Arguments to `rsmf extract-asset`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Path to the RSMF file.
    pub file: PathBuf,
    /// Asset name.
    #[arg(long = "name", value_name = "NAME")]
    pub name: String,
    /// Output file.
    pub out: PathBuf,
}

/// Execute `rsmf extract-asset`.
pub fn run(args: Args) -> Result<(), CliError> {
    let file = RsmfFile::open(&args.file)?;
    let asset = file
        .asset(&args.name)
        .ok_or_else(|| CliError::user(anyhow!("asset {} not found", args.name)))?;
    std::fs::write(&args.out, asset.bytes)?;
    println!(
        "extracted asset {name} ({bytes} bytes) -> {out}",
        name = args.name,
        bytes = asset.bytes.len(),
        out = args.out.display(),
    );
    Ok(())
}
