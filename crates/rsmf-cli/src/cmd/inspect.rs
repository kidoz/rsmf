//! `rsmf inspect` — human-readable summary of a file.

use std::path::PathBuf;

use rsmf_core::RsmfFile;

use super::CliError;

/// Arguments to `rsmf inspect`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Path to the RSMF file.
    pub file: PathBuf,
}

/// Execute `rsmf inspect`.
pub fn run(args: Args) -> Result<(), CliError> {
    let file = RsmfFile::open(&args.file)?;
    let summary = file.inspect();

    println!(
        "File:     {}",
        summary
            .path
            .as_ref()
            .map_or_else(|| String::from("<memory>"), |p| p.display().to_string())
    );
    println!("Size:     {} bytes", summary.file_size);
    println!(
        "Version:  {}.{}",
        summary.format_major, summary.format_minor
    );
    println!("Sections: {}", summary.section_count);
    println!("Tensors:  {}", summary.tensor_count);
    println!("Variants: {}", summary.variant_count);
    if summary.graph_kinds.is_empty() {
        println!("Graphs:   (none)");
    } else {
        let kinds: Vec<String> = summary
            .graph_kinds
            .iter()
            .map(|k| k.name().to_string())
            .collect();
        println!(
            "Graphs:   {} ({})",
            summary.graph_kinds.len(),
            kinds.join(", ")
        );
    }
    println!("Assets:   {}", summary.asset_count);

    if !summary.metadata.is_empty() {
        println!();
        println!("Metadata:");
        for (k, v) in &summary.metadata {
            println!("  {k} = {v}");
        }
    }

    let manifest = file.manifest();
    if !manifest.tensors.is_empty() {
        println!();
        println!("Tensors:");
        for t in &manifest.tensors {
            let dtype = t.dtype.name();
            let shape: Vec<String> = t.shape.iter().map(ToString::to_string).collect();
            let packed = t.packed_variants.len();
            println!(
                "  {name:<32} dtype={dtype:<4} shape=[{shape}] packed_variants={packed}",
                name = t.name,
                shape = shape.join(","),
            );
            // Show canonical variant.
            if let Some(cv) = manifest.variants.get(t.canonical_variant as usize) {
                println!(
                    "    canonical: encoding={enc} storage={sdtype} layout={layout} bytes={len}",
                    enc = cv.encoding.name(),
                    sdtype = cv.storage_dtype.name(),
                    layout = cv.layout.name(),
                    len = cv.length,
                );
            }
            // Show packed variants.
            for &idx in &t.packed_variants {
                if let Some(pv) = manifest.variants.get(idx as usize) {
                    println!(
                        "    packed[{idx}]: target={target} encoding={enc} storage={sdtype} layout={layout} bytes={len}",
                        target = pv.target.name(),
                        enc = pv.encoding.name(),
                        sdtype = pv.storage_dtype.name(),
                        layout = pv.layout.name(),
                        len = pv.length,
                    );
                }
            }
        }
    }

    if !manifest.assets.is_empty() {
        println!();
        println!("Assets:");
        for a in &manifest.assets {
            println!("  {name:<32} len={len}", name = a.name, len = a.length);
        }
    }

    Ok(())
}
