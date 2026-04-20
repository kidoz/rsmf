//! `rsmf pack` — convert a source file into RSMF. Currently supports
//! `--from-safetensors`, `--from-gguf` and `--from-npy`.

use std::path::PathBuf;

use anyhow::anyhow;
use rsmf_core::safetensors_convert;

use super::CliError;

/// Arguments to `rsmf pack`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Input safetensors file.
    #[arg(long = "from-safetensors", value_name = "PATH")]
    pub from_safetensors: Option<PathBuf>,
    /// Input GGUF file.
    #[arg(long = "from-gguf", value_name = "PATH")]
    pub from_gguf: Option<PathBuf>,
    /// Input NumPy (.npy) file.
    #[arg(long = "from-npy", value_name = "PATH")]
    pub from_npy: Option<PathBuf>,
    /// Input PyTorch checkpoint (.pt / .pth / .bin). Requires `python3`
    /// with `torch` and `safetensors` installed on PATH; rsmf shells out
    /// to it with `torch.load(..., weights_only=True)` so arbitrary-code
    /// execution is blocked during conversion. Set env var
    /// `RSMF_ALLOW_UNSAFE_PICKLE=1` only if you trust the source and the
    /// safe loader rejects it.
    #[arg(long = "from-torch", value_name = "PATH")]
    pub from_torch: Option<PathBuf>,
    /// Output RSMF file.
    #[arg(long = "out", value_name = "PATH")]
    pub out: PathBuf,
    /// Automatically generate a CastF16 packed variant for each f32 tensor,
    /// tagged with the given target (e.g. `cpu_generic`, `wgpu`).
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
    /// Optional graph files to embed.
    #[arg(long = "graph", value_name = "PATH")]
    pub graphs: Vec<PathBuf>,
    /// Optional asset files to embed (can be passed multiple times).
    #[arg(long = "asset", value_name = "PATH")]
    pub assets: Vec<PathBuf>,
    /// Compress the tensor arenas with zstd.
    #[arg(long)]
    pub compress_tensors: bool,
    /// Compress the graph section with zstd (requires `compression` feature
    /// on rsmf-core).
    #[arg(long)]
    pub compress_graph: bool,
    /// Compress the assets section with zstd.
    #[arg(long)]
    pub compress_assets: bool,
    /// Stream tensor, graph, and asset bodies from the source straight
    /// to disk without buffering them in RAM. Unlocks packing of
    /// multi-hundred-GB checkpoints and full model bundles in one pass.
    /// Currently compatible with `--from-safetensors` and `--from-npy`
    /// plus `--graph` and `--asset`; not yet with `--quantize-*`,
    /// `--cast-f16`, or `--compress-*` (those land in a follow-up).
    #[arg(long)]
    pub stream: bool,
}

/// Execute `rsmf pack`.
pub fn run(args: Args) -> Result<(), CliError> {
    let has_cast_f16 = args.cast_f16.is_some();
    let has_quant_q8_0 = args.quantize_q8_0.is_some();
    let has_quant_q4_0 = args.quantize_q4_0.is_some();
    let has_quant_q3_k = args.quantize_q3_k.is_some();
    let has_quant_nf4 = args.quantize_nf4.is_some();
    let has_quant_fp8 = args.quantize_fp8.is_some();

    // Fast path: --stream bypasses the in-memory RsmfWriter entirely.
    if args.stream {
        return run_streaming(&args);
    }

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

    // Resolve --from-torch by shelling out to python3 to produce a temp
    // safetensors file. Any --quantize / --cast-f16 flags then apply to it
    // through the existing safetensors pipeline below. The NamedTempFile is
    // held in a guard local so it lives until this function returns and is
    // unlinked automatically on drop.
    let _torch_tmp_guard;
    let effective_from_safetensors: Option<PathBuf> = if let Some(pt_path) = args.from_torch {
        let tmp = tempfile::Builder::new()
            .prefix("rsmf-torch-")
            .suffix(".safetensors")
            .tempfile()
            .map_err(|e| CliError::user(anyhow!("tempfile: {e}")))?;
        convert_torch_to_safetensors(&pt_path, tmp.path())?;
        let path = tmp.path().to_path_buf();
        _torch_tmp_guard = tmp;
        Some(path)
    } else {
        args.from_safetensors
    };

    let mut writer = if let Some(input) = effective_from_safetensors {
        let bytes = std::fs::read(&input)
            .map_err(|e| CliError::user(anyhow!("{}: {e}", input.display())))?;

        if has_cast_f16
            || has_quant_q8_0
            || has_quant_q4_0
            || has_quant_q3_k
            || has_quant_nf4
            || has_quant_fp8
        {
            let mut new_writer =
                rsmf_core::RsmfWriter::new().with_metadata("source", "safetensors");
            if let Ok((_, metadata)) = safetensors::tensor::SafeTensors::read_metadata(&bytes) {
                if let Some(map) = metadata.metadata() {
                    for (k, v) in map {
                        new_writer =
                            new_writer.with_metadata(format!("safetensors.{k}"), v.clone());
                    }
                }
            }
            let st = safetensors::tensor::SafeTensors::deserialize(&bytes)
                .map_err(|e| CliError::user(anyhow!("safetensors: {e}")))?;
            for name in st.names() {
                let tv = st
                    .tensor(name)
                    .map_err(|e| CliError::user(anyhow!("{e}")))?;
                let dtype = safetensors_convert::map_dtype(tv.dtype())?;
                let shape: Vec<u64> = tv.shape().iter().map(|&d| d as u64).collect();
                let tensor_bytes = tv.data().to_vec();
                let mut tensor_input = rsmf_core::TensorInput {
                    name: (*name).clone(),
                    dtype,
                    shape,
                    shard_id: 0,
                    metadata: Vec::new(),
                    canonical: rsmf_core::VariantInput::canonical_raw(tensor_bytes),
                    packed: Vec::new(),
                };

                if dtype == rsmf_core::LogicalDtype::F32 {
                    if let Some(target) = f16_target {
                        new_writer = new_writer.with_tensor_auto_f16(tensor_input.clone(), target);
                        tensor_input = new_writer.tensors().pop().unwrap();
                    }
                    if let Some(target) = q8_0_target {
                        new_writer = new_writer.with_tensor_auto_q8_0(tensor_input.clone(), target);
                        tensor_input = new_writer.tensors().pop().unwrap();
                    }
                    if let Some(target) = q4_0_target {
                        new_writer = new_writer.with_tensor_auto_q4_0(tensor_input.clone(), target);
                        tensor_input = new_writer.tensors().pop().unwrap();
                    }
                    if let Some(target) = q3_k_target {
                        new_writer = new_writer.with_tensor_auto_q3_k(tensor_input.clone(), target);
                        tensor_input = new_writer.tensors().pop().unwrap();
                    }
                    if let Some(target) = nf4_target {
                        new_writer = new_writer.with_tensor_auto_nf4(tensor_input.clone(), target);
                        tensor_input = new_writer.tensors().pop().unwrap();
                    }
                    if let Some(target) = fp8_target {
                        new_writer =
                            new_writer.with_tensor_auto_fp8_e4m3(tensor_input.clone(), target);
                        tensor_input = new_writer.tensors().pop().unwrap();
                    }
                }
                new_writer = new_writer.with_tensor(tensor_input);
            }
            new_writer
        } else {
            safetensors_convert::writer_from_safetensors_bytes(&bytes)?
        }
    } else if let Some(input) = args.from_gguf {
        #[cfg(feature = "gguf")]
        {
            rsmf_core::gguf_convert::writer_from_gguf_file(&input)?
        }
        #[cfg(not(feature = "gguf"))]
        {
            let _ = input;
            return Err(CliError::user(anyhow!("GGUF support not enabled")));
        }
    } else if let Some(input) = args.from_npy {
        #[cfg(feature = "npy")]
        {
            if has_cast_f16
                || has_quant_q8_0
                || has_quant_q4_0
                || has_quant_q3_k
                || has_quant_nf4
                || has_quant_fp8
            {
                let array: ndarray::ArrayD<f32> = ndarray_npy::read_npy(&input)
                    .map_err(|e| CliError::user(anyhow!("Failed to read .npy: {e}")))?;
                let shape: Vec<u64> = array.shape().iter().map(|&d| d as u64).collect();
                let standard = array.as_standard_layout().to_owned();
                let bytes = bytemuck::cast_slice::<f32, u8>(
                    standard
                        .as_slice()
                        .ok_or_else(|| anyhow!("NumPy array is not contiguous"))?,
                )
                .to_vec();
                let mut tensor_input = rsmf_core::TensorInput {
                    shard_id: 0,
                    name: input
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .ok_or_else(|| anyhow!("Invalid filename"))?
                        .to_string(),
                    dtype: rsmf_core::LogicalDtype::F32,
                    shape,
                    metadata: Vec::new(),
                    canonical: rsmf_core::VariantInput::canonical_raw(bytes),
                    packed: Vec::new(),
                };
                let mut w = rsmf_core::RsmfWriter::new().with_metadata("source", "npy");
                if let Some(target) = f16_target {
                    w = w.with_tensor_auto_f16(tensor_input.clone(), target);
                    tensor_input = w.tensors().pop().unwrap();
                }
                if let Some(target) = q8_0_target {
                    w = w.with_tensor_auto_q8_0(tensor_input.clone(), target);
                    tensor_input = w.tensors().pop().unwrap();
                }
                if let Some(target) = q4_0_target {
                    w = w.with_tensor_auto_q4_0(tensor_input.clone(), target);
                    tensor_input = w.tensors().pop().unwrap();
                }
                if let Some(target) = q3_k_target {
                    w = w.with_tensor_auto_q3_k(tensor_input.clone(), target);
                    tensor_input = w.tensors().pop().unwrap();
                }
                if let Some(target) = nf4_target {
                    w = w.with_tensor_auto_nf4(tensor_input.clone(), target);
                    tensor_input = w.tensors().pop().unwrap();
                }
                if let Some(target) = fp8_target {
                    w = w.with_tensor_auto_fp8_e4m3(tensor_input.clone(), target);
                    tensor_input = w.tensors().pop().unwrap();
                }
                w.with_tensor(tensor_input)
            } else {
                rsmf_core::npy_convert::writer_from_npy_file(&input)?
            }
        }
        #[cfg(not(feature = "npy"))]
        {
            let _ = input;
            return Err(CliError::user(anyhow!("NumPy support not enabled")));
        }
    } else {
        return Err(CliError::user(anyhow!(
            "one of --from-safetensors, --from-torch, --from-gguf, or --from-npy must be provided"
        )));
    };

    if args.compress_tensors {
        writer = writer
            .with_canonical_compression(3)
            .with_packed_compression(3);
    }

    for graph_path in args.graphs {
        let ext = graph_path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("")
            .to_lowercase();
        let kind = match ext.as_str() {
            "onnx" => rsmf_core::manifest::GraphKind::Onnx,
            "ort" => rsmf_core::manifest::GraphKind::Ort,
            _ => rsmf_core::manifest::GraphKind::Other,
        };
        let bytes = std::fs::read(&graph_path)
            .map_err(|e| CliError::user(anyhow!("{}: {e}", graph_path.display())))?;

        let mut input = rsmf_core::GraphInput {
            kind,
            alignment: rsmf_core::writer::DEFAULT_GRAPH_ALIGNMENT,
            bytes,
            metadata: Vec::new(),
            compress: None,
        };

        if args.compress_graph {
            input = input.with_compression(3);
        }

        writer = writer.with_graph(input);
    }

    for asset_path in args.assets {
        let name = asset_path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .ok_or_else(|| anyhow!("Invalid asset filename"))?
            .to_string();
        let bytes = std::fs::read(&asset_path)
            .map_err(|e| CliError::user(anyhow!("{}: {e}", asset_path.display())))?;

        let mut input = rsmf_core::AssetInput::new(name, bytes);

        if args.compress_assets {
            input = input.with_compression(3);
        }

        writer = writer.with_asset(input);
    }

    writer.write_to_path(&args.out)?;
    println!(
        "packed: wrote {output} from source",
        output = args.out.display(),
    );
    Ok(())
}

/// Python one-liner that converts a PyTorch checkpoint to safetensors.
///
/// Uses `torch.load(..., weights_only=True)` so arbitrary-code execution
/// on pickle load is blocked by default. Setting
/// `RSMF_ALLOW_UNSAFE_PICKLE=1` re-attempts with `weights_only=False` —
/// use only for trusted files that contain custom pickled classes.
///
/// The script walks up to two nesting levels (`state_dict`, `model`,
/// `model_state_dict`) to locate a tensor-valued dict, filters to
/// `torch.Tensor` values, and calls `safetensors.torch.save_file`.
const TORCH_TO_SAFETENSORS_SCRIPT: &str = r#"
import os
import sys

try:
    import torch
except ImportError:
    raise SystemExit("rsmf --from-torch requires Python with `torch` installed")
try:
    import safetensors.torch
except ImportError:
    raise SystemExit("rsmf --from-torch requires `safetensors` installed alongside torch")


def find_state_dict(obj):
    if isinstance(obj, torch.Tensor):
        return {"tensor": obj}
    if isinstance(obj, dict):
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        for k in ("state_dict", "model", "model_state_dict"):
            nested = obj.get(k)
            if isinstance(nested, dict) and any(isinstance(v, torch.Tensor) for v in nested.values()):
                return nested
    raise SystemExit(
        f"rsmf could not locate a tensor state_dict in {type(obj).__name__}"
    )


pt_path, out_path = sys.argv[1], sys.argv[2]
try:
    loaded = torch.load(pt_path, map_location="cpu", weights_only=True)
except Exception as err:
    if os.getenv("RSMF_ALLOW_UNSAFE_PICKLE", "0").lower() in ("1", "true", "yes"):
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
    else:
        raise SystemExit(
            f"torch.load rejected this file in safe mode: {err}\n"
            "Set RSMF_ALLOW_UNSAFE_PICKLE=1 only for files you trust."
        )

state = find_state_dict(loaded)
tensors = {}
for name, val in state.items():
    if not isinstance(val, torch.Tensor):
        continue
    # contiguous() + clone() guarantees owned, non-aliased storage that
    # safetensors.torch.save_file accepts without "shared memory" errors.
    tensors[name] = val.detach().contiguous().clone()

if not tensors:
    raise SystemExit("rsmf found no tensors in the input")

safetensors.torch.save_file(tensors, out_path)
sys.stderr.write(f"[rsmf] converted {len(tensors)} tensors via python subprocess\n")
"#;

/// Convert a PyTorch checkpoint at `pt_path` into a safetensors file at
/// `out_path` by shelling out to `python3`.
fn convert_torch_to_safetensors(
    pt_path: &std::path::Path,
    out_path: &std::path::Path,
) -> Result<(), CliError> {
    if !pt_path.exists() {
        return Err(CliError::user(anyhow!(
            "torch input not found: {}",
            pt_path.display()
        )));
    }

    let python = resolve_python_binary()?;
    let status = std::process::Command::new(&python)
        .arg("-c")
        .arg(TORCH_TO_SAFETENSORS_SCRIPT)
        .arg(pt_path)
        .arg(out_path)
        .status()
        .map_err(|e| {
            CliError::user(anyhow!(
                "failed to invoke {}: {e}. Install python3 with `torch` and `safetensors` to use --from-torch.",
                python
            ))
        })?;

    if !status.success() {
        return Err(CliError::user(anyhow!(
            "python torch→safetensors conversion failed (exit code {:?})",
            status.code()
        )));
    }

    if !out_path.exists() || std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0) == 0 {
        return Err(CliError::user(anyhow!(
            "torch→safetensors produced no output at {}",
            out_path.display()
        )));
    }
    Ok(())
}

/// Resolve the Python executable. Honours `RSMF_PYTHON_BIN` for
/// environments that need a specific venv interpreter.
fn resolve_python_binary() -> Result<String, CliError> {
    if let Ok(explicit) = std::env::var("RSMF_PYTHON_BIN") {
        if !explicit.is_empty() {
            return Ok(explicit);
        }
    }
    Ok("python3".to_string())
}

/// Streaming pack path.
///
/// Memory profile: tensor bodies never land in the process heap — the
/// safetensors path mmaps the source and hands each tensor's byte slice
/// directly to `StreamingRsmfWriter`, which pipes it through to disk in
/// 64 KiB chunks. Peak RSS stays in the low MB range regardless of source
/// size.
fn run_streaming(args: &Args) -> Result<(), CliError> {
    // Every mode-altering flag the MVP streaming writer can't honour is
    // rejected up-front with a clear message so users don't get a
    // surprising "quantize flag ignored" effect.
    if args.cast_f16.is_some()
        || args.quantize_q8_0.is_some()
        || args.quantize_q4_0.is_some()
        || args.quantize_q3_k.is_some()
        || args.quantize_nf4.is_some()
        || args.quantize_fp8.is_some()
    {
        return Err(CliError::user(anyhow!(
            "--stream does not yet support --quantize-* / --cast-f16; \
             use the default (buffered) pack path for those variants"
        )));
    }
    if args.compress_tensors || args.compress_graph || args.compress_assets {
        return Err(CliError::user(anyhow!(
            "--stream does not yet support --compress-* flags"
        )));
    }
    if args.from_gguf.is_some() {
        return Err(CliError::user(anyhow!(
            "--stream does not yet support --from-gguf"
        )));
    }
    if args.from_torch.is_some() {
        return Err(CliError::user(anyhow!(
            "--stream does not support --from-torch \
             (the torch subprocess already materialises a safetensors \
             tempfile which fits in RAM)"
        )));
    }

    if let Some(path) = args.from_safetensors.as_deref() {
        return stream_from_safetensors(path, &args.out, &args.graphs, &args.assets);
    }
    if let Some(path) = args.from_npy.as_deref() {
        #[cfg(feature = "npy")]
        {
            return stream_from_npy(path, &args.out, &args.graphs, &args.assets);
        }
        #[cfg(not(feature = "npy"))]
        {
            let _ = path;
            return Err(CliError::user(anyhow!("NumPy support not enabled")));
        }
    }
    Err(CliError::user(anyhow!(
        "--stream requires one of --from-safetensors or --from-npy"
    )))
}

/// Append the graphs and assets given on the CLI to the open
/// `StreamingRsmfWriter`. Called after all tensors have been streamed.
fn append_graphs_and_assets(
    writer: &mut rsmf_core::StreamingRsmfWriter,
    graphs: &[std::path::PathBuf],
    assets: &[std::path::PathBuf],
) -> Result<(), CliError> {
    for graph_path in graphs {
        let ext = graph_path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("")
            .to_lowercase();
        let kind = match ext.as_str() {
            "onnx" => rsmf_core::manifest::GraphKind::Onnx,
            "ort" => rsmf_core::manifest::GraphKind::Ort,
            _ => rsmf_core::manifest::GraphKind::Other,
        };
        let file = std::fs::File::open(graph_path)
            .map_err(|e| CliError::user(anyhow!("{}: {e}", graph_path.display())))?;
        writer.stream_graph(kind, file).map_err(CliError::from)?;
    }
    for asset_path in assets {
        let name = asset_path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .ok_or_else(|| anyhow!("invalid asset filename: {}", asset_path.display()))?
            .to_string();
        let file = std::fs::File::open(asset_path)
            .map_err(|e| CliError::user(anyhow!("{}: {e}", asset_path.display())))?;
        writer.stream_asset(name, file).map_err(CliError::from)?;
    }
    Ok(())
}

/// Stream every tensor out of an mmap'd safetensors file into a fresh rsmf.
fn stream_from_safetensors(
    src: &std::path::Path,
    out: &std::path::Path,
    graphs: &[std::path::PathBuf],
    assets: &[std::path::PathBuf],
) -> Result<(), CliError> {
    let file =
        std::fs::File::open(src).map_err(|e| CliError::user(anyhow!("{}: {e}", src.display())))?;
    // SAFETY: we treat the mmap as read-only; safetensors::deserialize
    // borrows from this slice for the duration of `writer.finish()`. The
    // mapping is dropped after `writer.finish()` completes.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| CliError::user(anyhow!("mmap {}: {e}", src.display())))?;

    let st = safetensors::SafeTensors::deserialize(&mmap[..])
        .map_err(|e| CliError::user(anyhow!("safetensors: {e}")))?;

    let mut writer = rsmf_core::StreamingRsmfWriter::new(out)
        .map_err(CliError::from)?
        .with_metadata("source", "safetensors");

    for name in st.names() {
        let tv = st
            .tensor(name)
            .map_err(|e| CliError::user(anyhow!("tensor {}: {e}", name)))?;
        let dtype = rsmf_core::safetensors_convert::map_dtype(tv.dtype())?;
        let shape: Vec<u64> = tv.shape().iter().map(|&d| d as u64).collect();
        writer
            .stream_canonical_tensor(name.clone(), dtype, shape, tv.data())
            .map_err(CliError::from)?;
    }
    append_graphs_and_assets(&mut writer, graphs, assets)?;
    writer.finish().map_err(CliError::from)?;
    println!(
        "packed (stream): wrote {} from {}",
        out.display(),
        src.display()
    );
    Ok(())
}

/// Stream a single-tensor `.npy` file into rsmf. The `ndarray_npy` crate
/// currently materialises the array in memory (its API doesn't expose a
/// streaming reader), so for NPY the "streaming" label refers to the
/// output side — bytes hit disk as they're produced without being
/// re-buffered through the batch `RsmfWriter`.
#[cfg(feature = "npy")]
fn stream_from_npy(
    src: &std::path::Path,
    out: &std::path::Path,
    graphs: &[std::path::PathBuf],
    assets: &[std::path::PathBuf],
) -> Result<(), CliError> {
    use std::io::Cursor;

    let array: ndarray::ArrayD<f32> = ndarray_npy::read_npy(src)
        .map_err(|e| CliError::user(anyhow!("read .npy {}: {e}", src.display())))?;
    let shape: Vec<u64> = array.shape().iter().map(|&d| d as u64).collect();
    let contiguous = array.as_standard_layout().to_owned();
    let slice = contiguous
        .as_slice()
        .ok_or_else(|| CliError::user(anyhow!("npy array is not contiguous after layout fix")))?;
    let bytes = bytemuck::cast_slice::<f32, u8>(slice);

    let name = src
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("tensor")
        .to_string();

    let mut writer = rsmf_core::StreamingRsmfWriter::new(out)
        .map_err(CliError::from)?
        .with_metadata("source", "npy");
    writer
        .stream_canonical_tensor(
            name,
            rsmf_core::LogicalDtype::F32,
            shape,
            Cursor::new(bytes),
        )
        .map_err(CliError::from)?;
    append_graphs_and_assets(&mut writer, graphs, assets)?;
    writer.finish().map_err(CliError::from)?;
    println!(
        "packed (stream): wrote {} from {}",
        out.display(),
        src.display()
    );
    Ok(())
}
