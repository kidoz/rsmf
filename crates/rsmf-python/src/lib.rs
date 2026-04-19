#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion)]
// pyo3 v0.22's `create_exception!` macro expands to `cfg(feature = "gil-refs")`
// checks that rustc's check-cfg lint doesn't know about. The warning is
// cosmetic and goes away on pyo3 v0.23+.
#![allow(unexpected_cfgs)]
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyIndexError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;

use rsmf_core::manifest::GraphKind;
use rsmf_core::{LogicalDtype, RsmfError as CoreError, RsmfFile as CoreFile};

// ---------------------------------------------------------------------------
// Exception hierarchy. `RsmfError` is the shared base so callers can write
// `except rsmf.RsmfError` to catch any library-level failure. Concrete
// subclasses allow structured logging and error routing without string
// matching on `.args[0]`.
// ---------------------------------------------------------------------------

create_exception!(rsmf, RsmfError, PyException, "Base class for all rsmf errors.");
create_exception!(rsmf, RsmfNotFound, RsmfError, "Tensor, asset, graph, or shard not found.");
create_exception!(rsmf, RsmfStructuralError, RsmfError, "Malformed file: bad preamble, overlapping sections, bad manifest.");
create_exception!(rsmf, RsmfVerificationError, RsmfError, "BLAKE3 checksum mismatch on preamble, section, variant, graph, or asset.");
create_exception!(rsmf, RsmfIoError, RsmfError, "Underlying I/O failure.");
create_exception!(rsmf, RsmfUnsupportedError, RsmfError, "Format or dtype is not supported by this build.");

fn map_core_error(err: CoreError) -> PyErr {
    let msg = err.to_string();
    match err {
        CoreError::NotFound { .. } => RsmfNotFound::new_err(msg),
        CoreError::InvalidMagic { .. }
        | CoreError::UnsupportedVersion { .. }
        | CoreError::Structural(_) => RsmfStructuralError::new_err(msg),
        CoreError::ChecksumMismatch { .. } => RsmfVerificationError::new_err(msg),
        CoreError::Io(_) | CoreError::IoWithPath { .. } => RsmfIoError::new_err(msg),
        CoreError::Unsupported(_)
        | CoreError::SafetensorsConversion(_)
        | CoreError::GgufConversion(_) => RsmfUnsupportedError::new_err(msg),
    }
}

// ---------------------------------------------------------------------------

#[pyclass(name = "RsmfFile")]
struct RsmfFile {
    inner: CoreFile,
}

#[pymethods]
impl RsmfFile {
    #[new]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        // mmap + preamble decode + section-table decode + manifest decode +
        // cross-reference validation all run on the thread — drop the GIL so
        // concurrent Python work keeps progressing for large files.
        let inner = py
            .allow_threads(|| CoreFile::open(&path))
            .map_err(map_core_error)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        let m = self.inner.manifest();
        format!(
            "<RsmfFile tensors={} variants={} graphs={} assets={}>",
            m.tensors.len(),
            m.variants.len(),
            m.graphs.len(),
            m.assets.len(),
        )
    }

    fn tensor_names(&self) -> Vec<String> {
        self.inner
            .manifest()
            .tensors
            .iter()
            .map(|t| t.name.clone())
            .collect()
    }

    fn asset_names(&self) -> Vec<String> {
        self.inner
            .manifest()
            .assets
            .iter()
            .map(|a| a.name.clone())
            .collect()
    }

    fn graph_count(&self) -> usize {
        self.inner.manifest().graphs.len()
    }

    fn graph_kind(&self, idx: usize) -> PyResult<&'static str> {
        let g = self
            .inner
            .manifest()
            .graphs
            .get(idx)
            .ok_or_else(|| PyIndexError::new_err(format!("graph index {idx} out of range")))?;
        Ok(match g.kind {
            GraphKind::Onnx => "onnx",
            GraphKind::Ort => "ort",
            GraphKind::Other => "other",
        })
    }

    fn get_graph<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Bound<'py, PyBytes>> {
        let payloads = self.inner.graph_payloads();
        let payload = payloads
            .get(idx)
            .ok_or_else(|| PyIndexError::new_err(format!("graph index {idx} out of range")))?;
        Ok(PyBytes::new_bound(py, payload.bytes))
    }

    fn metadata(&self) -> HashMap<String, String> {
        self.inner.manifest().metadata.iter().cloned().collect()
    }

    fn file_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.inner.inspect();
        let d = PyDict::new_bound(py);
        d.set_item("file_size", info.file_size)?;
        d.set_item("format_major", info.format_major)?;
        d.set_item("format_minor", info.format_minor)?;
        d.set_item("section_count", info.section_count)?;
        d.set_item("tensor_count", info.tensor_count)?;
        d.set_item("variant_count", info.variant_count)?;
        d.set_item("asset_count", info.asset_count)?;
        d.set_item("graph_count", info.graph_kinds.len())?;
        Ok(d)
    }

    /// Lightweight tensor descriptor dump. Does **not** read any variant bytes.
    fn tensor_info<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let manifest = self.inner.manifest();
        let t = manifest
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RsmfNotFound::new_err(format!("tensor not found: {name}")))?;

        let d = PyDict::new_bound(py);
        d.set_item("name", &t.name)?;
        d.set_item("dtype", t.dtype.name())?;
        d.set_item("shape", t.shape.clone())?;
        d.set_item("element_count", t.element_count().unwrap_or(0))?;
        d.set_item("shard_id", t.shard_id)?;
        d.set_item("canonical_variant_idx", t.canonical_variant)?;
        d.set_item("packed_variant_idxs", t.packed_variants.to_vec())?;
        let md: HashMap<String, String> = t.metadata.iter().cloned().collect();
        d.set_item("metadata", md)?;
        Ok(d)
    }

    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        // BLAKE3 over every section + variant + graph + asset can run for
        // seconds on multi-GB files; release the GIL for the hash work.
        py.allow_threads(|| self.inner.full_verify())
            .map_err(map_core_error)
    }

    /// Load a tensor's bytes as a NumPy array. Dequantizes quantized storage
    /// (Q4_0, Q8_0, NF4, F16, BF16) to f32.
    ///
    /// When `target` is given (e.g. `"cpu_generic"`), the first matching
    /// variant is loaded; otherwise the canonical variant is used. If
    /// `target` is given but no variant matches, falls back to canonical by
    /// default — pass `strict=True` to raise instead.
    #[pyo3(signature = (name, target=None, strict=false))]
    fn get_tensor<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        target: Option<&str>,
        strict: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(t) = target {
            if let Some(idx) = self.find_variant_idx_by_target(name, t)? {
                return self.decode_variant(py, name, idx);
            }
            if strict {
                return Err(RsmfNotFound::new_err(format!(
                    "tensor {name} has no variant with target={t}"
                )));
            }
        }
        self.decode_canonical(py, name)
    }

    /// Return per-variant metadata for the named tensor. The first entry is
    /// the canonical variant; the rest are the packed variants in
    /// declaration order. Each dict includes the global `variant_idx` that
    /// [`get_tensor_variant`] accepts.
    fn tensor_variants<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let manifest = self.inner.manifest();
        let t = manifest
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RsmfNotFound::new_err(format!("tensor not found: {name}")))?;

        let mut idxs: Vec<u32> = Vec::with_capacity(1 + t.packed_variants.len());
        idxs.push(t.canonical_variant);
        idxs.extend(t.packed_variants.iter().copied());

        let mut out = Vec::with_capacity(idxs.len());
        for (local_idx, &g_idx) in idxs.iter().enumerate() {
            let v = manifest.variants.get(g_idx as usize).ok_or_else(|| {
                RsmfStructuralError::new_err(format!(
                    "tensor {name} references invalid variant index {g_idx}"
                ))
            })?;
            let d = PyDict::new_bound(py);
            d.set_item("variant_idx", g_idx)?;
            d.set_item("local_idx", local_idx)?;
            d.set_item("is_canonical", local_idx == 0)?;
            d.set_item("target", v.target.name())?;
            d.set_item("encoding", v.encoding.name())?;
            d.set_item("storage_dtype", v.storage_dtype.name())?;
            d.set_item("layout", v.layout.name())?;
            d.set_item("length", v.length)?;
            d.set_item("alignment", v.alignment)?;
            out.push(d);
        }
        Ok(out)
    }

    fn get_tensor_variant<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        variant_idx: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.decode_variant(py, name, variant_idx)
    }

    fn get_asset<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(asset) = self.inner.asset(name) {
            Ok(Some(PyBytes::new_bound(py, asset.bytes).into_any()))
        } else {
            Ok(None)
        }
    }
}

// Non-py-exposed helpers.
impl RsmfFile {
    fn find_variant_idx_by_target(&self, name: &str, target: &str) -> PyResult<Option<u32>> {
        let manifest = self.inner.manifest();
        let t = manifest
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RsmfNotFound::new_err(format!("tensor not found: {name}")))?;

        // canonical comes first in the search order so target="canonical" works
        let mut idxs: Vec<u32> = Vec::with_capacity(1 + t.packed_variants.len());
        idxs.push(t.canonical_variant);
        idxs.extend(t.packed_variants.iter().copied());
        for idx in idxs {
            if let Some(v) = manifest.variants.get(idx as usize) {
                if v.target.name() == target {
                    return Ok(Some(idx));
                }
            }
        }
        Ok(None)
    }

    fn decode_canonical<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let view = self.inner.tensor_view(name).map_err(map_core_error)?;
        decode_view_to_pyarray(py, &view)
    }

    fn decode_variant<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        variant_idx: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let view = self
            .inner
            .tensor_view_variant(name, variant_idx)
            .map_err(map_core_error)?;
        decode_view_to_pyarray(py, &view)
    }
}

/// Owned buffer produced by decoding a tensor variant off-GIL. The enum keeps
/// the per-dtype return path monomorphic while letting us do the heavy
/// copy/dequant in a single `py.allow_threads` block.
enum DecodedBuf {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I16(Vec<i16>),
    I8(Vec<i8>),
    U64(Vec<u64>),
    U32(Vec<u32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

fn decode_view_to_pyarray<'py>(
    py: Python<'py>,
    view: &rsmf_core::TensorView<'_>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
    let dtype = view.dtype();

    // Decode / copy happens without the GIL. The result is an owned `Vec`
    // moved into NumPy via `into_pyarray_bound` (no second copy).
    let buf: Result<DecodedBuf, rsmf_core::RsmfError> = py.allow_threads(|| match dtype {
        // F16 and BF16 have no stable numpy dtype; surface them as f32.
        LogicalDtype::F32 | LogicalDtype::F16 | LogicalDtype::BF16 => {
            view.decode_f32().map(DecodedBuf::F32)
        }
        LogicalDtype::F64 => view.to_vec::<f64>().map(DecodedBuf::F64),
        LogicalDtype::I64 => view.to_vec::<i64>().map(DecodedBuf::I64),
        LogicalDtype::I32 => view.to_vec::<i32>().map(DecodedBuf::I32),
        LogicalDtype::I16 => view.to_vec::<i16>().map(DecodedBuf::I16),
        LogicalDtype::I8 => view.to_vec::<i8>().map(DecodedBuf::I8),
        LogicalDtype::U64 => view.to_vec::<u64>().map(DecodedBuf::U64),
        LogicalDtype::U32 => view.to_vec::<u32>().map(DecodedBuf::U32),
        LogicalDtype::U16 => view.to_vec::<u16>().map(DecodedBuf::U16),
        LogicalDtype::U8 => view.to_vec::<u8>().map(DecodedBuf::U8),
        other => Err(rsmf_core::RsmfError::unsupported(format!(
            "dtype {other:?} not yet supported by rsmf-python"
        ))),
    });

    let decoded = buf.map_err(map_core_error)?;
    match decoded {
        DecodedBuf::F32(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::F64(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::I64(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::I32(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::I16(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::I8(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::U64(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::U32(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::U16(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
        DecodedBuf::U8(v) => Ok(v.into_pyarray_bound(py).reshape(shape)?.into_any()),
    }
}

#[pymodule]
fn _rsmf(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RsmfFile>()?;

    // Register exception types so callers can do `except rsmf.RsmfError`.
    m.add("RsmfError", py.get_type_bound::<RsmfError>())?;
    m.add("RsmfNotFound", py.get_type_bound::<RsmfNotFound>())?;
    m.add(
        "RsmfStructuralError",
        py.get_type_bound::<RsmfStructuralError>(),
    )?;
    m.add(
        "RsmfVerificationError",
        py.get_type_bound::<RsmfVerificationError>(),
    )?;
    m.add("RsmfIoError", py.get_type_bound::<RsmfIoError>())?;
    m.add(
        "RsmfUnsupportedError",
        py.get_type_bound::<RsmfUnsupportedError>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_structure() {}
}
