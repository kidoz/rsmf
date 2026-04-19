//! `rsmf extract` — extract a tensor's canonical bytes to a file.

use std::path::PathBuf;

use rsmf_core::RsmfFile;

use super::CliError;

/// Arguments to `rsmf extract`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Path to the RSMF file.
    pub file: PathBuf,
    /// Tensor name.
    #[arg(long = "tensor", value_name = "NAME")]
    pub tensor: String,
    /// Output file.
    pub out: PathBuf,
}

/// Execute `rsmf extract`.
pub fn run(args: Args) -> Result<(), CliError> {
    let file = RsmfFile::open(&args.file)?;
    let view = file.tensor_view(&args.tensor)?;
    std::fs::write(&args.out, view.bytes())?;
    println!(
        "extracted tensor {tensor} ({bytes} bytes) -> {out}",
        tensor = args.tensor,
        bytes = view.bytes().len(),
        out = args.out.display(),
    );
    Ok(())
}
