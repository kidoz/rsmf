//! `rsmf rewrite` — copy an RSMF into a new file while optionally
//! stripping variants, graphs, assets, or metadata.
//!
//! Typical workflows:
//!
//! - Ship a smaller production artifact by dropping dev-only variants:
//!   `rsmf rewrite in.rsmf out.rsmf --strip-variants cpu_generic`.
//! - Keep only the canonical path (drops every packed variant):
//!   `rsmf rewrite in.rsmf out.rsmf --keep-only-canonical`.
//! - Strip the bundled graph or selected assets before distribution.
//! - Re-compress an already-packed file by combining strips with the
//!   `--compress-*` flags.
//!
//! The current implementation uses the batch reader + batch writer —
//! every byte travels through RAM once. A streaming-rewrite mode is a
//! follow-up that would reuse `StreamingRsmfWriter` for the write side.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::anyhow;
use rsmf_core::manifest::GraphKind;
use rsmf_core::{
    AssetInput, GraphInput, RsmfFile, RsmfWriter, TargetTag, TensorInput, VariantInput,
    writer::{DEFAULT_ASSETS_ALIGNMENT, DEFAULT_GRAPH_ALIGNMENT},
};

use super::{CliError, parse_target_tag};

/// Arguments to `rsmf rewrite`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Source RSMF file.
    pub input: PathBuf,
    /// Destination RSMF file.
    pub output: PathBuf,
    /// Drop packed variants whose target tag matches this value. May be
    /// passed multiple times. Canonical variants are never dropped.
    #[arg(long = "strip-variants", value_name = "TARGET")]
    pub strip_variants: Vec<String>,
    /// Drop every packed variant, keeping only canonical variants for
    /// each tensor.
    #[arg(long = "keep-only-canonical")]
    pub keep_only_canonical: bool,
    /// Drop all graph sections from the output.
    #[arg(long = "strip-graphs")]
    pub strip_graphs: bool,
    /// Drop one asset by name. Repeat to strip several. Pass `*` to
    /// drop all assets.
    #[arg(long = "strip-asset", value_name = "NAME")]
    pub strip_assets: Vec<String>,
    /// Drop every manifest metadata key that starts with this prefix.
    #[arg(long = "strip-metadata-prefix", value_name = "PREFIX")]
    pub strip_metadata_prefix: Option<String>,
    /// Re-compress the tensor arenas in the output with zstd.
    #[arg(long)]
    pub compress_tensors: bool,
    /// Re-compress the graph section in the output with zstd.
    #[arg(long)]
    pub compress_graph: bool,
    /// Re-compress the assets section in the output with zstd.
    #[arg(long)]
    pub compress_assets: bool,
}

/// Execute `rsmf rewrite`.
pub fn run(args: Args) -> Result<(), CliError> {
    if args.input == args.output {
        return Err(CliError::user(anyhow!(
            "rewrite output must differ from input"
        )));
    }

    let strip_tags: Vec<TargetTag> = args
        .strip_variants
        .iter()
        .map(|s| parse_target_tag(s))
        .collect::<Result<Vec<_>, _>>()?;

    let strip_assets: HashSet<&str> = args.strip_assets.iter().map(String::as_str).collect();
    let strip_all_assets = strip_assets.contains("*");

    let src = RsmfFile::open(&args.input)?;

    let mut writer = RsmfWriter::new();

    // Global metadata — filter by prefix.
    for (k, v) in &src.manifest().metadata {
        if let Some(prefix) = args.strip_metadata_prefix.as_deref() {
            if k.starts_with(prefix) {
                continue;
            }
        }
        writer = writer.with_metadata(k.clone(), v.clone());
    }

    if args.compress_tensors {
        writer = writer
            .with_canonical_compression(3)
            .with_packed_compression(3);
    }

    // Tensors: copy canonical, filter packed variants.
    let tensor_names: Vec<String> = src
        .manifest()
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let mut summary_dropped_variants = 0usize;
    let mut summary_total_variants = 0usize;

    for name in &tensor_names {
        let tensor = src
            .manifest()
            .tensors
            .iter()
            .find(|t| &t.name == name)
            .expect("tensor listed but not found");

        // Canonical bytes come from the batch reader's `tensor_view`,
        // which transparently decompresses if the canonical arena was
        // compressed. `canonical_raw` rebuilds the variant under the
        // default (Raw, row-major, alignment=64, inherits dtype) shape.
        let canonical_view = src
            .tensor_view(name)
            .map_err(|e| CliError::user(anyhow!("{name}: {e}")))?;
        let canonical = VariantInput::canonical_raw(canonical_view.bytes().to_vec());

        let mut ti = TensorInput {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
            shard_id: tensor.shard_id,
            metadata: tensor.metadata.clone(),
            canonical,
            packed: Vec::new(),
        };

        if !args.keep_only_canonical {
            for &packed_idx in &tensor.packed_variants {
                summary_total_variants += 1;
                let vd = &src.manifest().variants[packed_idx as usize];
                if strip_tags.contains(&vd.target) {
                    summary_dropped_variants += 1;
                    continue;
                }
                let pview = src
                    .tensor_view_variant(name, packed_idx)
                    .map_err(|e| CliError::user(anyhow!("{name} variant: {e}")))?;
                ti.packed.push(VariantInput {
                    target: vd.target,
                    encoding: vd.encoding,
                    storage_dtype: Some(vd.storage_dtype),
                    layout: vd.layout,
                    alignment: vd.alignment,
                    bytes: pview.bytes().to_vec(),
                    meta: vd.meta.clone(),
                });
            }
        } else {
            summary_dropped_variants += tensor.packed_variants.len();
            summary_total_variants += tensor.packed_variants.len();
        }
        writer = writer.with_tensor(ti);
    }

    // Graphs.
    if !args.strip_graphs {
        let payloads = src.graph_payloads();
        for (i, p) in payloads.iter().enumerate() {
            let mut gi = match p.kind {
                GraphKind::Onnx => GraphInput::onnx(p.bytes.to_vec()),
                GraphKind::Ort => GraphInput::ort(p.bytes.to_vec()),
                GraphKind::Other => GraphInput {
                    kind: GraphKind::Other,
                    alignment: DEFAULT_GRAPH_ALIGNMENT,
                    bytes: p.bytes.to_vec(),
                    metadata: p.metadata.to_vec(),
                    compress: None,
                },
            };
            gi.metadata = p.metadata.to_vec();
            if args.compress_graph {
                gi = gi.with_compression(3);
            }
            writer = writer.with_graph(gi);
            let _ = i;
        }
    }

    // Assets.
    let mut kept_assets = 0usize;
    if !strip_all_assets {
        for ad in &src.manifest().assets {
            if strip_assets.contains(ad.name.as_str()) {
                continue;
            }
            let asset = src.asset(&ad.name).ok_or_else(|| {
                CliError::user(anyhow!(
                    "asset {} present in manifest but missing payload",
                    ad.name
                ))
            })?;
            let mut ai = AssetInput {
                name: ad.name.clone(),
                alignment: DEFAULT_ASSETS_ALIGNMENT.max(ad.alignment),
                bytes: asset.bytes.to_vec(),
                metadata: ad.metadata.clone(),
                compress: None,
            };
            if args.compress_assets {
                ai = ai.with_compression(3);
            }
            writer = writer.with_asset(ai);
            kept_assets += 1;
        }
    }

    writer.write_to_path(&args.output)?;

    println!(
        "rewrote: {} -> {} (tensors={}, packed_variants {}->{}, assets kept={})",
        args.input.display(),
        args.output.display(),
        tensor_names.len(),
        summary_total_variants,
        summary_total_variants - summary_dropped_variants,
        kept_assets,
    );
    Ok(())
}
