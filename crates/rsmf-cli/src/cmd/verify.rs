//! `rsmf verify` — structural (default) or full checksum verification.

use std::path::PathBuf;

use rsmf_core::{RsmfFile, Validator};

use super::CliError;

/// Arguments to `rsmf verify`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Path to the RSMF file.
    pub file: PathBuf,
    /// Also re-hash every section / variant / asset payload and compare.
    #[arg(long)]
    pub full: bool,
}

/// Execute `rsmf verify`.
pub fn run(args: Args) -> Result<(), CliError> {
    let file = RsmfFile::open(&args.file)?;
    Validator::structural(&file)?;
    println!("structural: ok");
    if args.full {
        Validator::full(&file)?;
        println!("full checksum: ok");
    }
    Ok(())
}
