//! `rsmf import` — download a model from HuggingFace and convert to RSMF.

use anyhow::anyhow;
use hf_hub::api::sync::ApiBuilder;
use rsmf_core::safetensors_convert;
use rsmf_core::{LogicalDtype, RsmfWriter, TensorInput, VariantInput};
use std::path::PathBuf;

use super::CliError;

/// Arguments to `rsmf import`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// HuggingFace repository ID (e.g. `microsoft/resnet-50`).
    #[arg(long = "repo")]
    pub repo: String,
    /// Output RSMF file.
    #[arg(long = "out", value_name = "PATH")]
    pub out: PathBuf,
    /// Automatically generate a CastF16 packed variant for each f32 tensor.
    #[arg(long = "cast-f16", value_name = "TARGET")]
    pub cast_f16: Option<String>,
    /// Automatically generate an INT8 (Q8_0) block-quantized packed variant.
    #[arg(long = "quantize-q8_0", value_name = "TARGET")]
    pub quantize_q8_0: Option<String>,
    /// Automatically generate a Q4_0 block-quantized packed variant.
    #[arg(long = "quantize-q4_0", value_name = "TARGET")]
    pub quantize_q4_0: Option<String>,
    /// Automatically generate a Q3_K block-quantized packed variant.
    #[arg(long = "quantize-q3_k", value_name = "TARGET")]
    pub quantize_q3_k: Option<String>,
    /// Automatically generate an NF4 block-quantized packed variant.
    #[arg(long = "quantize-nf4", value_name = "TARGET")]
    pub quantize_nf4: Option<String>,
    /// Automatically generate an FP8 (E4M3) packed variant.
    #[arg(long = "quantize-fp8", value_name = "TARGET")]
    pub quantize_fp8: Option<String>,
    /// Block size for INT8 quantization (default: 32).
    #[arg(long = "block-size", default_value_t = 32)]
    pub block_size: usize,
    /// Compress the tensor arenas with zstd.
    #[arg(long)]
    pub compress_tensors: bool,
}

/// Execute `rsmf import`.
pub fn run(args: Args) -> Result<(), CliError> {
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| CliError::user(anyhow!("Failed to connect to HF: {e}")))?;
    let repo = api.repo(hf_hub::Repo::new(
        args.repo.clone(),
        hf_hub::RepoType::Model,
    ));

    let f16_target = args
        .cast_f16
        .as_deref()
        .map(super::parse_target_tag)
        .transpose()?;
    let q8_0_target = args
        .quantize_q8_0
        .as_deref()
        .map(super::parse_target_tag)
        .transpose()?;
    let q4_0_target = args
        .quantize_q4_0
        .as_deref()
        .map(super::parse_target_tag)
        .transpose()?;
    let q3_k_target = args
        .quantize_q3_k
        .as_deref()
        .map(super::parse_target_tag)
        .transpose()?;
    let nf4_target = args
        .quantize_nf4
        .as_deref()
        .map(super::parse_target_tag)
        .transpose()?;
    let fp8_target = args
        .quantize_fp8
        .as_deref()
        .map(super::parse_target_tag)
        .transpose()?;

    let mut writer = RsmfWriter::new().with_metadata("source", format!("hf:{}", args.repo));

    if args.compress_tensors {
        writer = writer
            .with_canonical_compression(3)
            .with_packed_compression(3);
    }

    // 1. Resolve safetensors files.
    let index_file = repo.get("model.safetensors.index.json");
    let mut safetensors_files = Vec::new();

    if let Ok(index_path) = index_file {
        println!("Detected sharded safetensors model.");
        let content = std::fs::read_to_string(&index_path)
            .map_err(|e| CliError::user(anyhow!("Failed to read index: {e}")))?;
        let index: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| CliError::user(anyhow!("Failed to parse index: {e}")))?;

        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| CliError::user(anyhow!("Index missing weight_map")))?;

        let mut unique_shards = std::collections::HashSet::new();
        for val in weight_map.values() {
            if let Some(s) = val.as_str() {
                unique_shards.insert(s.to_string());
            }
        }
        for shard in unique_shards {
            safetensors_files.push(shard);
        }
    } else {
        safetensors_files.push("model.safetensors".to_string());
    }

    // 2. Process shards
    for (shard_idx, shard_name) in safetensors_files.iter().enumerate() {
        println!("Downloading {}...", shard_name);
        let path = repo
            .get(shard_name)
            .map_err(|e| CliError::user(anyhow!("Failed to download {}: {e}", shard_name)))?;
        let bytes = std::fs::read(&path)
            .map_err(|e| CliError::user(anyhow!("Failed to read {}: {e}", shard_name)))?;

        let st = safetensors::tensor::SafeTensors::deserialize(&bytes)
            .map_err(|e| CliError::user(anyhow!("Safetensors error in {}: {e}", shard_name)))?;

        for name in st.names() {
            let tv = st
                .tensor(name)
                .map_err(|e| CliError::user(anyhow!("Tensor {}: {e}", name)))?;
            let dtype = safetensors_convert::map_dtype(tv.dtype())?;
            let shape: Vec<u64> = tv.shape().iter().map(|&d| d as u64).collect();
            let tensor_bytes = tv.data().to_vec();

            let mut tensor_input = TensorInput {
                shard_id: shard_idx as u64,
                name: (*name).clone(),
                dtype,
                shape,
                metadata: Vec::new(),
                canonical: VariantInput::canonical_raw(tensor_bytes),
                packed: Vec::new(),
            };

            if dtype == LogicalDtype::F32 {
                if let Some(target) = f16_target {
                    writer = writer.with_tensor_auto_f16(tensor_input.clone(), target);
                    tensor_input = writer.tensors().pop().unwrap();
                }
                if let Some(target) = q8_0_target {
                    writer = writer.with_tensor_auto_q8_0(tensor_input.clone(), target);
                    tensor_input = writer.tensors().pop().unwrap();
                }
                if let Some(target) = q4_0_target {
                    writer = writer.with_tensor_auto_q4_0(tensor_input.clone(), target);
                    tensor_input = writer.tensors().pop().unwrap();
                }
                if let Some(target) = q3_k_target {
                    writer = writer.with_tensor_auto_q3_k(tensor_input.clone(), target);
                    tensor_input = writer.tensors().pop().unwrap();
                }
                if let Some(target) = nf4_target {
                    writer = writer.with_tensor_auto_nf4(tensor_input.clone(), target);
                    tensor_input = writer.tensors().pop().unwrap();
                }
                if let Some(target) = fp8_target {
                    writer = writer.with_tensor_auto_fp8_e4m3(tensor_input.clone(), target);
                    tensor_input = writer.tensors().pop().unwrap();
                }
            }

            writer = writer.with_tensor(tensor_input);
        }
    }

    // 3. Optional assets and graphs. Bundle the standard transformers /
    //    tokenizers sidecar files so the imported rsmf is self-contained
    //    (model weights + tokenizer + runtime configs, one artifact).
    const HF_ASSET_CANDIDATES: &[&str] = &[
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "preprocessor_config.json",
        "chat_template.json",
    ];
    for asset_name in HF_ASSET_CANDIDATES {
        if let Ok(asset_path) = repo.get(asset_name) {
            println!("Bundling {} as asset...", asset_name);
            let bytes = std::fs::read(&asset_path)
                .map_err(|e| CliError::user(anyhow!("Failed to read {}: {e}", asset_name)))?;
            writer =
                writer.with_asset(rsmf_core::AssetInput::new((*asset_name).to_string(), bytes));
        }
    }

    // Try to find ONNX graph
    for graph_path_candidate in ["model.onnx", "onnx/model.onnx"] {
        if let Ok(graph_path) = repo.get(graph_path_candidate) {
            println!("Bundling {} as graph...", graph_path_candidate);
            let bytes = std::fs::read(&graph_path).map_err(|e| {
                CliError::user(anyhow!(
                    "Failed to read graph {}: {e}",
                    graph_path_candidate
                ))
            })?;
            let mut input = rsmf_core::GraphInput::onnx(bytes);
            #[cfg(feature = "compression")]
            {
                input = input.with_compression(3);
            }
            writer = writer.with_graph(input);
            break;
        }
    }

    writer.write_to_path(&args.out)?;
    println!(
        "Successfully imported {} to {}",
        args.repo,
        args.out.display()
    );

    Ok(())
}
