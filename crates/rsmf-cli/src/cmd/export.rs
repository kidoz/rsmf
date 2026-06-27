//! `rsmf export` — convert an RSMF file into another model container.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::anyhow;
use rsmf_core::manifest::GraphKind;
use rsmf_core::{
    EncodingKind, LayoutTag, LogicalDtype, RsmfFile, StorageDtype, TargetTag, TensorView,
};
use safetensors::tensor::{Dtype as StDtype, TensorView as StTensorView};

use super::CliError;

/// Arguments to `rsmf export`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Export destination format.
    #[command(subcommand)]
    pub format: Format,
}

/// Supported export formats.
#[derive(Debug, clap::Subcommand)]
pub enum Format {
    /// Export canonical tensors to a `.safetensors` file.
    Safetensors(SafetensorsArgs),
    /// Export an embedded ONNX graph payload.
    Onnx(OnnxArgs),
}

/// Arguments to `rsmf export safetensors`.
#[derive(Debug, clap::Args)]
pub struct SafetensorsArgs {
    /// Input RSMF file.
    pub file: PathBuf,
    /// Output safetensors file.
    pub out: PathBuf,
    /// Decode F32 logical tensors stored as packed, quantized, cast, or FP8
    /// variants into raw F32 safetensors payloads.
    #[arg(long = "decode-f32")]
    pub decode_f32: bool,
    /// Export a packed variant with the given target tag instead of the
    /// canonical variant. For example: `cpu_generic`, `wgpu`, `cuda`.
    #[arg(long = "target", value_name = "TARGET")]
    pub target: Option<String>,
}

/// Arguments to `rsmf export onnx`.
#[derive(Debug, clap::Args)]
pub struct OnnxArgs {
    /// Input RSMF file.
    pub file: PathBuf,
    /// Output ONNX file.
    pub out: PathBuf,
    /// Graph index to export.
    #[arg(long = "index", default_value_t = 0)]
    pub index: usize,
}

struct ExportTensor {
    name: String,
    dtype: StDtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// Execute `rsmf export`.
pub fn run(args: Args) -> Result<(), CliError> {
    match args.format {
        Format::Safetensors(args) => run_safetensors(args),
        Format::Onnx(args) => run_onnx(args),
    }
}

fn run_safetensors(args: SafetensorsArgs) -> Result<(), CliError> {
    let file = RsmfFile::open(&args.file)?;
    let target = args
        .target
        .as_deref()
        .map(super::parse_target_tag)
        .transpose()?;
    let mut tensors = Vec::with_capacity(file.manifest().tensors.len());

    for tensor in &file.manifest().tensors {
        let view = tensor_view_for_export(&file, &tensor.name, target)?;
        tensors.push(export_tensor(&tensor.name, &view, args.decode_f32)?);
    }

    let views = build_safetensors_views(&tensors)?;
    let metadata = export_metadata(&file);
    safetensors::serialize_to_file(views, metadata, &args.out)
        .map_err(|e| CliError::user(anyhow!("safetensors export failed: {e}")))?;

    println!(
        "exported {count} tensors -> {out}",
        count = tensors.len(),
        out = args.out.display()
    );
    Ok(())
}

fn run_onnx(args: OnnxArgs) -> Result<(), CliError> {
    let file = RsmfFile::open(&args.file)?;
    let payloads = file.graph_payloads();
    let payload = payloads
        .get(args.index)
        .ok_or_else(|| CliError::user(anyhow!("graph index {} not found", args.index)))?;
    if payload.kind != GraphKind::Onnx {
        return Err(CliError::user(anyhow!(
            "graph index {} is {:?}, not ONNX",
            args.index,
            payload.kind
        )));
    }

    std::fs::write(&args.out, payload.bytes)?;
    println!(
        "exported ONNX graph {index} ({bytes} bytes) -> {out}",
        index = args.index,
        bytes = payload.bytes.len(),
        out = args.out.display()
    );
    Ok(())
}

fn tensor_view_for_export<'a>(
    file: &'a RsmfFile,
    name: &str,
    target: Option<TargetTag>,
) -> Result<TensorView<'a>, CliError> {
    let Some(target) = target else {
        return Ok(file.tensor_view(name)?);
    };

    let tensor = file
        .manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| CliError::user(anyhow!("tensor {name} not found")))?;
    for &variant_idx in &tensor.packed_variants {
        let variant = file
            .manifest()
            .variants
            .get(variant_idx as usize)
            .ok_or_else(|| CliError::user(anyhow!("tensor {name} references missing variant")))?;
        if variant.target == target {
            return Ok(file.tensor_view_variant(name, variant_idx)?);
        }
    }

    Err(CliError::user(anyhow!(
        "tensor {name} has no packed variant for target {target:?}"
    )))
}

fn export_tensor(
    name: &str,
    view: &TensorView<'_>,
    decode_f32: bool,
) -> Result<ExportTensor, CliError> {
    let shape = shape_to_usize(view.shape())?;

    if should_decode_f32(view) {
        if !decode_f32 {
            return Err(CliError::user(anyhow!(
                "tensor {name} uses {encoding:?}/{storage:?}/{layout:?}; \
                 rerun with --decode-f32 to export it as raw F32 safetensors",
                encoding = view.encoding,
                storage = view.storage_dtype,
                layout = view.layout,
            )));
        }
        let element_count = element_count(view.shape())?;
        let mut decoded = view.decode_f32()?;
        if decoded.len() < element_count {
            return Err(CliError::user(anyhow!(
                "tensor {name} decoded to {} elements, but shape requires {element_count}",
                decoded.len()
            )));
        }
        decoded.truncate(element_count);
        let bytes = bytemuck::cast_slice::<f32, u8>(&decoded).to_vec();
        return Ok(ExportTensor {
            name: name.to_string(),
            dtype: StDtype::F32,
            shape,
            bytes,
        });
    }

    if view.encoding != EncodingKind::Raw || view.layout != LayoutTag::RowMajor {
        return Err(CliError::user(anyhow!(
            "tensor {name} is not raw row-major ({encoding:?}/{layout:?}); \
             only logical F32 tensors can be decoded during safetensors export",
            encoding = view.encoding,
            layout = view.layout,
        )));
    }

    let dtype = map_logical_dtype(view.dtype())?;
    Ok(ExportTensor {
        name: name.to_string(),
        dtype,
        shape,
        bytes: view.bytes().to_vec(),
    })
}

fn should_decode_f32(view: &TensorView<'_>) -> bool {
    view.dtype() == LogicalDtype::F32
        && (view.encoding != EncodingKind::Raw
            || view.layout != LayoutTag::RowMajor
            || matches!(
                view.storage_dtype,
                StorageDtype::Q4_0
                    | StorageDtype::Q8_0
                    | StorageDtype::Q3K
                    | StorageDtype::NF4
                    | StorageDtype::Fp8E4M3
                    | StorageDtype::Q4K
                    | StorageDtype::Q5_0
                    | StorageDtype::Q5K
                    | StorageDtype::Q6K
                    | StorageDtype::Q2K
                    | StorageDtype::Fp8E5M2
            ))
}

fn build_safetensors_views<'a>(
    tensors: &'a [ExportTensor],
) -> Result<Vec<(String, StTensorView<'a>)>, CliError> {
    tensors
        .iter()
        .map(|tensor| {
            let view = StTensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes)
                .map_err(|e| CliError::user(anyhow!("tensor {}: {e}", tensor.name)))?;
            Ok((tensor.name.clone(), view))
        })
        .collect()
}

fn export_metadata(file: &RsmfFile) -> Option<HashMap<String, String>> {
    let mut metadata = HashMap::new();
    metadata.insert("rsmf.exported_from".to_string(), "rsmf".to_string());
    for (key, value) in &file.manifest().metadata {
        metadata.insert(format!("rsmf.source.{key}"), value.clone());
    }
    Some(metadata)
}

fn shape_to_usize(shape: &[u64]) -> Result<Vec<usize>, CliError> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim)
                .map_err(|_| CliError::user(anyhow!("shape dimension {dim} does not fit usize")))
        })
        .collect()
}

fn element_count(shape: &[u64]) -> Result<usize, CliError> {
    let mut count = 1usize;
    for &dim in shape {
        let dim = usize::try_from(dim)
            .map_err(|_| CliError::user(anyhow!("shape dimension {dim} does not fit usize")))?;
        count = count
            .checked_mul(dim)
            .ok_or_else(|| CliError::user(anyhow!("shape element count overflows usize")))?;
    }
    Ok(count)
}

fn map_logical_dtype(dtype: LogicalDtype) -> Result<StDtype, CliError> {
    Ok(match dtype {
        LogicalDtype::Bool => StDtype::BOOL,
        LogicalDtype::U8 => StDtype::U8,
        LogicalDtype::I8 => StDtype::I8,
        LogicalDtype::U16 => StDtype::U16,
        LogicalDtype::I16 => StDtype::I16,
        LogicalDtype::U32 => StDtype::U32,
        LogicalDtype::I32 => StDtype::I32,
        LogicalDtype::U64 => StDtype::U64,
        LogicalDtype::I64 => StDtype::I64,
        LogicalDtype::F16 => StDtype::F16,
        LogicalDtype::BF16 => StDtype::BF16,
        LogicalDtype::F32 => StDtype::F32,
        LogicalDtype::F64 => StDtype::F64,
    })
}
