//! `rsmf inspect` — human-readable summary of a file.

use std::path::PathBuf;

use rsmf_core::RsmfFile;

use super::CliError;

/// Arguments to `rsmf inspect`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Path to the RSMF file.
    pub file: PathBuf,
    /// Include Mixture-of-Experts grouping from `moe.*` metadata.
    #[arg(long)]
    pub moe: bool,
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

    if args.moe {
        let moe = file.moe_experts()?;
        println!();
        println!("MoE:");
        println!(
            "  n_experts={} top_k={} n_shared={} model_arch={}",
            display_opt_u32(moe.n_experts),
            display_opt_u32(moe.top_k),
            display_opt_u32(moe.n_shared),
            moe.model_arch.as_deref().unwrap_or("(none)")
        );
        if moe.groups.is_empty() {
            println!("  groups: (none)");
        } else {
            println!("  groups:");
            for group in &moe.groups {
                let expert = group
                    .expert_id
                    .map_or_else(|| String::from("none"), |id| id.to_string());
                println!(
                    "    layer={layer} expert={expert} shared={shared} role={role} tensors={tensors}",
                    layer = group.layer,
                    shared = group.shared,
                    role = group.role.as_str(),
                    tensors = group.tensor_names.join(","),
                );
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

fn display_opt_u32(value: Option<u32>) -> String {
    value.map_or_else(|| String::from("(none)"), |v| v.to_string())
}
