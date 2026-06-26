//! `rsmf verify` — structural (default) or full checksum verification.

use std::path::PathBuf;

use anyhow::anyhow;
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
    /// Attach an external shard as `ID=PATH`. May be repeated.
    #[arg(long = "shard", value_name = "ID=PATH")]
    pub shards: Vec<String>,
}

/// Execute `rsmf verify`.
pub fn run(args: Args) -> Result<(), CliError> {
    let shard_paths = args
        .shards
        .iter()
        .map(|spec| parse_shard_spec(spec))
        .collect::<Result<Vec<_>, _>>()?;
    let file = if shard_paths.is_empty() {
        RsmfFile::open(&args.file)?
    } else {
        RsmfFile::open_with_shards(&args.file, shard_paths)?
    };
    Validator::structural(&file)?;
    println!("structural: ok");
    if args.full {
        Validator::full(&file)?;
        println!("full checksum: ok");
    }
    Ok(())
}

fn parse_shard_spec(spec: &str) -> Result<(u64, PathBuf), CliError> {
    let (id, path) = spec
        .split_once('=')
        .ok_or_else(|| CliError::user(anyhow!("--shard must be formatted as ID=PATH")))?;
    let shard_id = id
        .parse::<u64>()
        .map_err(|e| CliError::user(anyhow!("invalid shard id {id:?}: {e}")))?;
    if shard_id == 0 {
        return Err(CliError::user(anyhow!("shard id must be non-zero")));
    }
    if path.is_empty() {
        return Err(CliError::user(anyhow!("shard path must not be empty")));
    }
    Ok((shard_id, PathBuf::from(path)))
}
