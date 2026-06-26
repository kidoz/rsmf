//! `rsmf shard` — emit a master RSMF plus external raw shard buffers.

use std::path::PathBuf;

use clap::ValueEnum;
use rsmf_core::{RsmfFile, ShardStrategy, ShardWriteOptions, write_sharded_file};

use super::CliError;

/// Built-in shard assignment strategies exposed by the CLI.
#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "lower")]
pub enum ShardBy {
    /// Balance tensors by total variant byte length.
    Size,
    /// Group tensors by variant `tier.intent` metadata.
    Tier,
    /// Group tensors by MoE expert metadata.
    Expert,
}

impl From<ShardBy> for ShardStrategy {
    fn from(value: ShardBy) -> Self {
        match value {
            ShardBy::Size => ShardStrategy::Size,
            ShardBy::Tier => ShardStrategy::Tier,
            ShardBy::Expert => ShardStrategy::Expert,
        }
    }
}

/// Arguments to `rsmf shard`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Source RSMF file.
    pub input: PathBuf,
    /// Assignment strategy.
    #[arg(long = "by", value_enum)]
    pub by: ShardBy,
    /// Number of shard files to emit.
    #[arg(long = "shards")]
    pub shards: u64,
    /// Output directory. Receives `master.rsmf` and `shard-<id>.bin` files.
    #[arg(long = "out-dir")]
    pub out_dir: PathBuf,
    /// Optional master output path. Defaults to `<out-dir>/master.rsmf`.
    #[arg(long = "master-out")]
    pub master_out: Option<PathBuf>,
}

/// Execute `rsmf shard`.
pub fn run(args: Args) -> Result<(), CliError> {
    let src = RsmfFile::open(&args.input)?;
    let master_path = args
        .master_out
        .clone()
        .unwrap_or_else(|| args.out_dir.join("master.rsmf"));
    let summary = write_sharded_file(
        &src,
        &master_path,
        &args.out_dir,
        &ShardWriteOptions {
            shard_count: args.shards,
            strategy: args.by.into(),
        },
    )?;

    println!(
        "sharded: {} -> {} (strategy={:?}, shards={})",
        args.input.display(),
        summary.master_path.display(),
        args.by,
        summary.shards.len(),
    );
    for shard in &summary.shards {
        println!(
            "  shard_id={} path={} bytes={} tensors={} checksum={}",
            shard.shard_id,
            shard.path.display(),
            shard.bytes,
            shard.tensor_count,
            hex::encode(shard.checksum),
        );
    }
    Ok(())
}
