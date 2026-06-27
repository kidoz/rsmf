use std::collections::HashMap;
use std::fmt::Debug;

use ndarray::ArrayViewD;
use ort::session::SessionOutputs;
use ort::value::{DynValue, Tensor, TensorElementType};
use rsmf_core::{LogicalDtype, TensorView};
use serde::{Deserialize, Serialize};

use crate::{Result, RuntimeError, ort_error};

/// Owned tensor value accepted by and returned from `rsmf-runtime`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dtype", rename_all = "snake_case")]
pub enum RuntimeTensor {
    /// F32 tensor.
    F32 { shape: Vec<usize>, data: Vec<f32> },
    /// F64 tensor.
    F64 { shape: Vec<usize>, data: Vec<f64> },
    /// I64 tensor.
    I64 { shape: Vec<usize>, data: Vec<i64> },
    /// I32 tensor.
    I32 { shape: Vec<usize>, data: Vec<i32> },
    /// U8 tensor.
    U8 { shape: Vec<usize>, data: Vec<u8> },
    /// I8 tensor.
    I8 { shape: Vec<usize>, data: Vec<i8> },
    /// Boolean tensor.
    Bool { shape: Vec<usize>, data: Vec<bool> },
}

impl RuntimeTensor {
    pub(crate) fn into_ort_value(self) -> Result<DynValue> {
        match self {
            RuntimeTensor::F32 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::F64 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::I64 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::I32 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::U8 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::I8 { shape, data } => tensor_from_vec(shape, data),
            RuntimeTensor::Bool { shape, data } => tensor_from_vec(shape, data),
        }
    }
}

/// Named input tensor map.
pub type RuntimeInputs = HashMap<String, RuntimeTensor>;

/// Named output tensor map.
pub type RuntimeOutputs = HashMap<String, RuntimeTensor>;

pub(crate) fn runtime_tensor_shape(tensor: &RuntimeTensor) -> &[usize] {
    match tensor {
        RuntimeTensor::F32 { shape, .. }
        | RuntimeTensor::F64 { shape, .. }
        | RuntimeTensor::I64 { shape, .. }
        | RuntimeTensor::I32 { shape, .. }
        | RuntimeTensor::U8 { shape, .. }
        | RuntimeTensor::I8 { shape, .. }
        | RuntimeTensor::Bool { shape, .. } => shape,
    }
}

pub(crate) fn request_leading_batch_size(inputs: &RuntimeInputs) -> Option<usize> {
    let mut leading = None;
    for tensor in inputs.values() {
        let shape = runtime_tensor_shape(tensor);
        let &first_dim = shape.first()?;
        if leading.is_some_and(|existing| existing != first_dim) {
            return None;
        }
        leading = Some(first_dim);
    }
    leading
}

pub(crate) fn tensors_are_batch_compatible(
    first: &RuntimeTensor,
    candidate: &RuntimeTensor,
) -> bool {
    match (first, candidate) {
        (
            RuntimeTensor::F32 {
                shape: first_shape, ..
            },
            RuntimeTensor::F32 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::F64 {
                shape: first_shape, ..
            },
            RuntimeTensor::F64 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::I64 {
                shape: first_shape, ..
            },
            RuntimeTensor::I64 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::I32 {
                shape: first_shape, ..
            },
            RuntimeTensor::I32 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::U8 {
                shape: first_shape, ..
            },
            RuntimeTensor::U8 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::I8 {
                shape: first_shape, ..
            },
            RuntimeTensor::I8 {
                shape: candidate_shape,
                ..
            },
        )
        | (
            RuntimeTensor::Bool {
                shape: first_shape, ..
            },
            RuntimeTensor::Bool {
                shape: candidate_shape,
                ..
            },
        ) => {
            !first_shape.is_empty()
                && first_shape.len() == candidate_shape.len()
                && first_shape[1..] == candidate_shape[1..]
        }
        _ => false,
    }
}

pub(crate) fn runtime_inputs_data_bytes(inputs: &RuntimeInputs) -> Result<usize> {
    inputs.values().try_fold(0usize, |acc, tensor| {
        acc.checked_add(runtime_tensor_data_bytes(tensor)?)
            .ok_or_else(|| RuntimeError::Shape("runtime input byte count overflow".to_string()))
    })
}

fn runtime_tensor_data_bytes(tensor: &RuntimeTensor) -> Result<usize> {
    match tensor {
        RuntimeTensor::F32 { data, .. } => runtime_tensor_data_bytes_for::<f32>(data.len()),
        RuntimeTensor::F64 { data, .. } => runtime_tensor_data_bytes_for::<f64>(data.len()),
        RuntimeTensor::I64 { data, .. } => runtime_tensor_data_bytes_for::<i64>(data.len()),
        RuntimeTensor::I32 { data, .. } => runtime_tensor_data_bytes_for::<i32>(data.len()),
        RuntimeTensor::U8 { data, .. } => runtime_tensor_data_bytes_for::<u8>(data.len()),
        RuntimeTensor::I8 { data, .. } => runtime_tensor_data_bytes_for::<i8>(data.len()),
        RuntimeTensor::Bool { data, .. } => runtime_tensor_data_bytes_for::<bool>(data.len()),
    }
}

fn runtime_tensor_data_bytes_for<T>(len: usize) -> Result<usize> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| RuntimeError::Shape("runtime tensor byte count overflow".to_string()))
}

fn leading_batch_parts(name: &str, shape: &[usize], data_len: usize) -> Result<(usize, usize)> {
    let Some((&batch_size, trailing_shape)) = shape.split_first() else {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} has no leading batch dimension"
        )));
    };
    let trailing_elements = trailing_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| RuntimeError::Shape(format!("tensor {name} shape overflows usize")))
    })?;
    let expected = batch_size.checked_mul(trailing_elements).ok_or_else(|| {
        RuntimeError::Shape(format!("tensor {name} element count overflows usize"))
    })?;
    if expected != data_len {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} implies {expected} elements, got {data_len}",
            shape
        )));
    }
    Ok((batch_size, trailing_elements))
}

macro_rules! merge_runtime_tensor_variant {
    ($name:expr, $tensors:expr, $variant:ident) => {{
        let mut merged_shape = Vec::new();
        let mut merged_data = Vec::new();
        for (idx, tensor) in $tensors.iter().enumerate() {
            let RuntimeTensor::$variant { shape, data } = tensor else {
                return Err(RuntimeError::Shape(format!(
                    "input {} has mixed tensor dtypes for batching",
                    $name
                )));
            };
            let (batch_size, _) = leading_batch_parts($name, shape, data.len())?;
            if idx == 0 {
                merged_shape = shape.clone();
                merged_shape[0] = 0;
            } else if shape[1..] != merged_shape[1..] {
                return Err(RuntimeError::Shape(format!(
                    "input {} has incompatible trailing dimensions for batching",
                    $name
                )));
            }
            merged_shape[0] = merged_shape[0].checked_add(batch_size).ok_or_else(|| {
                RuntimeError::Shape(format!("input {} batch dimension overflows usize", $name))
            })?;
            merged_data.extend_from_slice(data);
        }
        Ok(RuntimeTensor::$variant {
            shape: merged_shape,
            data: merged_data,
        })
    }};
}

pub(crate) fn merge_runtime_tensors(
    name: &str,
    tensors: &[&RuntimeTensor],
) -> Result<RuntimeTensor> {
    let Some(first) = tensors.first() else {
        return Err(RuntimeError::Shape(format!(
            "input {name} has no tensors to batch"
        )));
    };
    match first {
        RuntimeTensor::F32 { .. } => merge_runtime_tensor_variant!(name, tensors, F32),
        RuntimeTensor::F64 { .. } => merge_runtime_tensor_variant!(name, tensors, F64),
        RuntimeTensor::I64 { .. } => merge_runtime_tensor_variant!(name, tensors, I64),
        RuntimeTensor::I32 { .. } => merge_runtime_tensor_variant!(name, tensors, I32),
        RuntimeTensor::U8 { .. } => merge_runtime_tensor_variant!(name, tensors, U8),
        RuntimeTensor::I8 { .. } => merge_runtime_tensor_variant!(name, tensors, I8),
        RuntimeTensor::Bool { .. } => merge_runtime_tensor_variant!(name, tensors, Bool),
    }
}

macro_rules! split_runtime_tensor_variant {
    ($name:expr, $shape:expr, $data:expr, $batch_sizes:expr, $variant:ident) => {{
        let total_batch = $batch_sizes.iter().try_fold(0usize, |acc, &batch_size| {
            acc.checked_add(batch_size).ok_or_else(|| {
                RuntimeError::Shape(format!("output {} batch dimension overflows usize", $name))
            })
        })?;
        let (batch_size, trailing_elements) = leading_batch_parts($name, &$shape, $data.len())?;
        if batch_size != total_batch {
            return Err(RuntimeError::Shape(format!(
                "output {} leading batch dimension is {batch_size}, expected {total_batch}",
                $name
            )));
        }
        let mut offset = 0usize;
        let mut split = Vec::with_capacity($batch_sizes.len());
        for &request_batch_size in $batch_sizes {
            let len = request_batch_size
                .checked_mul(trailing_elements)
                .ok_or_else(|| {
                    RuntimeError::Shape(format!(
                        "output {} split element count overflows usize",
                        $name
                    ))
                })?;
            let end = offset.checked_add(len).ok_or_else(|| {
                RuntimeError::Shape(format!("output {} split range overflows usize", $name))
            })?;
            let mut shape = $shape.clone();
            shape[0] = request_batch_size;
            split.push(RuntimeTensor::$variant {
                shape,
                data: $data[offset..end].to_vec(),
            });
            offset = end;
        }
        Ok(split)
    }};
}

fn split_runtime_tensor(
    name: &str,
    tensor: RuntimeTensor,
    batch_sizes: &[usize],
) -> Result<Vec<RuntimeTensor>> {
    match tensor {
        RuntimeTensor::F32 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, F32)
        }
        RuntimeTensor::F64 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, F64)
        }
        RuntimeTensor::I64 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, I64)
        }
        RuntimeTensor::I32 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, I32)
        }
        RuntimeTensor::U8 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, U8)
        }
        RuntimeTensor::I8 { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, I8)
        }
        RuntimeTensor::Bool { shape, data } => {
            split_runtime_tensor_variant!(name, shape, data, batch_sizes, Bool)
        }
    }
}

pub(crate) fn split_runtime_outputs(
    outputs: RuntimeOutputs,
    batch_sizes: &[usize],
) -> Result<Vec<RuntimeOutputs>> {
    let mut per_request = (0..batch_sizes.len())
        .map(|_| RuntimeOutputs::new())
        .collect::<Vec<_>>();
    for (name, tensor) in outputs {
        let split = split_runtime_tensor(&name, tensor, batch_sizes)?;
        for (request_outputs, tensor) in per_request.iter_mut().zip(split) {
            request_outputs.insert(name.clone(), tensor);
        }
    }
    Ok(per_request)
}

/// Helper to convert an RSMF `TensorView` into an `ndarray::ArrayViewD`.
pub fn tensor_view_to_ndarray<'a>(view: &'a TensorView<'a>) -> Result<ArrayViewD<'a, f32>> {
    if view.dtype() != LogicalDtype::F32 {
        return Err(RuntimeError::UnsupportedDtype(format!(
            "only F32 tensors support zero-copy ndarray conversion, got {:?}",
            view.dtype()
        )));
    }

    let shape: Vec<usize> = view
        .shape()
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                RuntimeError::Shape(format!("tensor dimension {dim} exceeds usize::MAX"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let data = view.as_slice::<f32>()?;

    ArrayViewD::from_shape(shape, data)
        .map_err(|e| RuntimeError::Shape(format!("ndarray shape mismatch: {e}")))
}

pub(crate) fn tensor_from_vec<T>(shape: Vec<usize>, data: Vec<T>) -> Result<DynValue>
where
    T: ort::value::PrimitiveTensorElementType + Clone + Debug + 'static,
{
    let expected = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| RuntimeError::Shape("runtime tensor element count overflow".to_string()))
    })?;
    if expected != data.len() {
        return Err(RuntimeError::Shape(format!(
            "shape {:?} implies {expected} elements, got {}",
            shape,
            data.len()
        )));
    }
    Tensor::<T>::from_array((shape, data.into_boxed_slice()))
        .map(|value| value.into_dyn())
        .map_err(|e| ort_error("tensor value creation", e))
}

pub(crate) fn materialize_outputs(outputs: SessionOutputs<'_>) -> Result<RuntimeOutputs> {
    outputs
        .into_iter()
        .map(|(name, value)| {
            let tensor = match value.dtype().tensor_type() {
                Some(TensorElementType::Float32) => materialize_tensor_f32(&value)?,
                Some(TensorElementType::Float64) => materialize_tensor_f64(&value)?,
                Some(TensorElementType::Int64) => materialize_tensor_i64(&value)?,
                Some(TensorElementType::Int32) => materialize_tensor_i32(&value)?,
                Some(TensorElementType::Uint8) => materialize_tensor_u8(&value)?,
                Some(TensorElementType::Int8) => materialize_tensor_i8(&value)?,
                Some(TensorElementType::Bool) => materialize_tensor_bool(&value)?,
                Some(other) => {
                    return Err(RuntimeError::UnsupportedDtype(other.to_string()));
                }
                None => {
                    return Err(RuntimeError::UnsupportedDtype(value.dtype().to_string()));
                }
            };
            Ok((name.to_string(), tensor))
        })
        .collect()
}

fn shape_to_usize(shape: &[i64]) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                RuntimeError::Shape(format!("output dimension {dim} cannot convert to usize"))
            })
        })
        .collect()
}

fn materialize_tensor_f32(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| ort_error("extract f32 tensor", e))?;
    Ok(RuntimeTensor::F32 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_f64(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<f64>()
        .map_err(|e| ort_error("extract f64 tensor", e))?;
    Ok(RuntimeTensor::F64 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_i64(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<i64>()
        .map_err(|e| ort_error("extract i64 tensor", e))?;
    Ok(RuntimeTensor::I64 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_i32(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<i32>()
        .map_err(|e| ort_error("extract i32 tensor", e))?;
    Ok(RuntimeTensor::I32 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_u8(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<u8>()
        .map_err(|e| ort_error("extract u8 tensor", e))?;
    Ok(RuntimeTensor::U8 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_i8(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<i8>()
        .map_err(|e| ort_error("extract i8 tensor", e))?;
    Ok(RuntimeTensor::I8 {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

fn materialize_tensor_bool(value: &DynValue) -> Result<RuntimeTensor> {
    let (shape, data) = value
        .try_extract_tensor::<bool>()
        .map_err(|e| ort_error("extract bool tensor", e))?;
    Ok(RuntimeTensor::Bool {
        shape: shape_to_usize(shape)?,
        data: data.to_vec(),
    })
}

pub(crate) fn runtime_tensor_kind(tensor: &RuntimeTensor) -> &'static str {
    match tensor {
        RuntimeTensor::F32 { .. } => "F32",
        RuntimeTensor::F64 { .. } => "F64",
        RuntimeTensor::I64 { .. } => "I64",
        RuntimeTensor::I32 { .. } => "I32",
        RuntimeTensor::U8 { .. } => "U8",
        RuntimeTensor::I8 { .. } => "I8",
        RuntimeTensor::Bool { .. } => "Bool",
    }
}
