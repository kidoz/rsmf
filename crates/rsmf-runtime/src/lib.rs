use ndarray::{ArrayD, ArrayViewD};
use ort::session::Session;
use rsmf_core::{LogicalDtype, Result, RsmfFile, TensorView};
use std::collections::HashMap;

#[cfg(feature = "tracing")]
use tracing::info_span;

/// High-level inference engine for RSMF files.
pub struct Engine {
    file: RsmfFile,
}

impl Engine {
    /// Create a new Engine from an RSMF file.
    pub fn new(file: RsmfFile) -> Result<Self> {
        Ok(Self { file })
    }

    /// Return the underlying RSMF file.
    pub fn file(&self) -> &RsmfFile {
        &self.file
    }

    /// Create an ONNX Runtime session from the graph at the specified index.
    pub fn session(&self, graph_idx: usize) -> Result<Session> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::session", graph_idx).entered();

        let payloads = self.file.graph_payloads();
        let payload = payloads.get(graph_idx).ok_or_else(|| {
            rsmf_core::error::RsmfError::not_found(format!("graph at index {}", graph_idx))
        })?;

        Session::builder()
            .map_err(|e| {
                rsmf_core::error::RsmfError::unsupported(format!("ort session builder failed: {e}"))
            })?
            .commit_from_memory(payload.bytes)
            .map_err(|e| {
                rsmf_core::error::RsmfError::unsupported(format!(
                    "ort session creation failed: {e}"
                ))
            })
    }

    /// Run inference on a graph with flexible inputs.
    pub fn run_session<'a>(
        &self,
        session: &'a mut Session,
        inputs: Vec<(&str, ort::value::Value)>,
    ) -> Result<ort::session::SessionOutputs<'a>> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::run_session").entered();

        session
            .run(inputs)
            .map_err(|e| rsmf_core::error::RsmfError::unsupported(format!("ort run failed: {e}")))
    }

    /// Simple run helper for F32 models.
    pub fn run_f32(
        &self,
        graph_idx: usize,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        let mut session = self.session(graph_idx)?;

        let mut ort_inputs = Vec::new();
        for (name, array) in &inputs {
            let val = ort::value::Value::from_array(array.clone()).map_err(|e| {
                rsmf_core::error::RsmfError::unsupported(format!("ort value creation failed: {e}"))
            })?;
            ort_inputs.push((name.as_str(), val.into()));
        }

        let outputs = self.run_session(&mut session, ort_inputs)?;

        let mut result = HashMap::new();
        for (name, value) in outputs.iter() {
            let (shape, data) = value.try_extract_tensor::<f32>().map_err(|e| {
                rsmf_core::error::RsmfError::unsupported(format!("ort extract failed: {e}"))
            })?;

            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let array = ArrayD::from_shape_vec(shape_vec, data.to_vec()).map_err(|e| {
                rsmf_core::error::RsmfError::structural(format!("ort extract shape mismatch: {e}"))
            })?;

            result.insert(name.to_string(), array);
        }

        Ok(result)
    }
}

/// Helper to convert an RSMF `TensorView` into an `ndarray::ArrayViewD`.
pub fn tensor_view_to_ndarray<'a>(view: &'a TensorView<'a>) -> Result<ArrayViewD<'a, f32>> {
    if view.dtype() != LogicalDtype::F32 {
        return Err(rsmf_core::error::RsmfError::unsupported(format!(
            "Only F32 tensors supported for ndarray conversion, got {:?}",
            view.dtype()
        )));
    }

    let shape: Vec<usize> = view.shape().iter().map(|&s| s as usize).collect();
    let data = view.as_slice::<f32>()?;

    ArrayViewD::from_shape(shape, data).map_err(|e| {
        rsmf_core::error::RsmfError::structural(format!("ndarray shape mismatch: {e}"))
    })
}
