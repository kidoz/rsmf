use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, MutexGuard};

use ndarray::ArrayD;
use ort::ep::CPUExecutionProvider;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{RunOptions, Session};
use ort::value::DynValue;
use rsmf_core::manifest::GraphKind;
use rsmf_core::tensor::variant::{EncodingKind, LayoutTag};
use rsmf_core::{LogicalDtype, RsmfError, RsmfFile, StorageDtype, TensorView};

use crate::allocator_stats::provider_allocator_stats;
use crate::executor::RuntimeCancellationToken;
use crate::native_decoder::{
    NATIVE_DECODER_CHAT_TEMPLATE_ASSET, NATIVE_DECODER_TOKENIZER_ASSET,
    NATIVE_DECODER_TOKENIZER_CONFIG_ASSET, NativeDecoderChatMessage, NativeDecoderContract,
    NativeDecoderGenerateOutput, NativeDecoderReferenceLogitCheck,
    NativeDecoderReferenceLogitReport, NativeDecoderRunOptions, NativeDecoderSession,
    NativeDecoderTextGenerateOutput, NativeDecoderTokenizer, NativeDecoderWeightOptions,
    NativeDecoderWeights, load_native_decoder_weights, native_decoder_check_reference_logits,
    native_decoder_generate_with_backend, resolve_native_decoder_backend,
};
use crate::onnx::{OnnxTensorDataType, onnx_initializers};
use crate::session::{
    ExecutionProvider, InitializerBinding, InitializerMemoryReport, IoBindingPolicy,
    RuntimeMemoryMeasurement, SessionKey, SessionMemoryReport, SessionOptions, ValueInfo,
    current_process_resident_set_bytes, runtime_capability_report,
};
use crate::tensor::{materialize_outputs, runtime_tensor_kind, tensor_from_vec};
use crate::{Result, RuntimeError, RuntimeInputs, RuntimeOutputs, RuntimeTensor, ort_error};

#[cfg(feature = "tracing")]
use tracing::info_span;

/// Cached ORT session handle.
#[derive(Clone, Debug)]
pub struct SessionHandle {
    key: SessionKey,
    session: Arc<Mutex<Session>>,
    inputs: Vec<ValueInfo>,
    outputs: Vec<ValueInfo>,
    memory_report: SessionMemoryReport,
}

impl SessionHandle {
    /// Cache key used to create this session.
    #[must_use]
    pub fn key(&self) -> &SessionKey {
        &self.key
    }

    /// Graph input metadata captured when the session was built.
    #[must_use]
    pub fn inputs(&self) -> &[ValueInfo] {
        &self.inputs
    }

    /// Graph output metadata captured when the session was built.
    #[must_use]
    pub fn outputs(&self) -> &[ValueInfo] {
        &self.outputs
    }

    /// Memory accounting captured when this session was built.
    #[must_use]
    pub fn memory_report(&self) -> &SessionMemoryReport {
        &self.memory_report
    }

    /// Runtime capabilities relevant to this session's residency policy.
    #[must_use]
    pub fn capability_report(&self) -> crate::RuntimeCapabilityReport {
        runtime_capability_report()
    }

    /// Run this cached session with owned runtime tensors.
    pub fn run(&self, inputs: RuntimeInputs) -> Result<RuntimeOutputs> {
        self.run_with_cancellation(inputs, None)
    }

    pub(crate) fn run_with_cancellation(
        &self,
        inputs: RuntimeInputs,
        cancellation: Option<&RuntimeCancellationToken>,
    ) -> Result<RuntimeOutputs> {
        match cancellation {
            Some(cancellation) => self.run_with_cancellations(inputs, &[cancellation]),
            None => self.run_with_cancellations(inputs, &[]),
        }
    }

    fn run_with_cancellations(
        &self,
        inputs: RuntimeInputs,
        cancellations: &[&RuntimeCancellationToken],
    ) -> Result<RuntimeOutputs> {
        let ort_inputs = inputs
            .into_iter()
            .map(|(name, tensor)| Ok((name, tensor.into_ort_value()?)))
            .collect::<Result<Vec<_>>>()?;
        let mut session = self.lock_session()?;
        if !cancellations.is_empty() {
            let run_options =
                Arc::new(RunOptions::new().map_err(|e| ort_error("create run options", e))?);
            for cancellation in cancellations {
                if let Err(error) = cancellation.attach_run_options(Arc::clone(&run_options)) {
                    for cancellation in cancellations {
                        cancellation.clear_run_options();
                    }
                    return Err(error);
                }
            }
            let result = if self.key.options.io_binding == IoBindingPolicy::Cpu {
                run_with_cpu_io_binding(&mut session, &self.outputs, ort_inputs, Some(&run_options))
            } else {
                session
                    .run_with_options(ort_inputs, &*run_options)
                    .map_err(|e| ort_error("session run", e))
                    .and_then(materialize_outputs)
            };
            for cancellation in cancellations {
                cancellation.clear_run_options();
            }
            result
        } else {
            if self.key.options.io_binding == IoBindingPolicy::Cpu {
                run_with_cpu_io_binding(&mut session, &self.outputs, ort_inputs, None)
            } else {
                let outputs = session
                    .run(ort_inputs)
                    .map_err(|e| ort_error("session run", e))?;
                materialize_outputs(outputs)
            }
        }
    }

    fn lock_session(&self) -> Result<MutexGuard<'_, Session>> {
        self.session
            .lock()
            .map_err(|_| RuntimeError::SessionPoisoned)
    }
}

/// High-level inference engine for RSMF files.
pub struct Engine {
    file: RsmfFile,
    cache: Mutex<HashMap<SessionKey, SessionHandle>>,
}

struct BuiltSession {
    session: Session,
    memory_report: SessionMemoryReport,
}

struct InitializerValue {
    value: Arc<DynValue>,
    memory_report: InitializerMemoryReport,
}

impl Engine {
    /// Create a new engine from an opened RSMF file.
    pub fn new(file: RsmfFile) -> Result<Self> {
        Ok(Self {
            file,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Return the underlying RSMF file.
    #[must_use]
    pub fn file(&self) -> &RsmfFile {
        &self.file
    }

    /// Runtime capabilities relevant to residency and provider memory
    /// accounting.
    #[must_use]
    pub fn capability_report(&self) -> crate::RuntimeCapabilityReport {
        runtime_capability_report()
    }

    /// Resolve and validate the native decoder model contract for this RSMF
    /// file.
    ///
    /// R4.1 only validates assets, config, tensor names, tensor shapes, and
    /// weight dtypes. It does not execute a decoder block or parse tokenizer
    /// internals.
    pub fn native_decoder_contract(&self) -> Result<NativeDecoderContract> {
        NativeDecoderContract::from_file(&self.file)
    }

    /// Load native decoder weights from canonical RSMF tensor variants.
    ///
    /// The contract is validated first, then selected tensor variants are
    /// decoded to owned f32 buffers for the native runtime.
    pub fn native_decoder_weights(&self) -> Result<NativeDecoderWeights> {
        self.native_decoder_weights_with_options(&NativeDecoderWeightOptions::default())
    }

    /// Load native decoder weights from selected RSMF tensor variants.
    ///
    /// Missing entries in `options.tensor_variants` load canonical variants.
    /// Selected variants must belong to the requested tensor and decode to the
    /// expected row-major f32 element count.
    pub fn native_decoder_weights_with_options(
        &self,
        options: &NativeDecoderWeightOptions,
    ) -> Result<NativeDecoderWeights> {
        let contract = self.native_decoder_contract()?;
        load_native_decoder_weights(&self.file, contract.config, options)
    }

    /// Load a resident native decoder session with decoded weights and
    /// tokenizer.
    ///
    /// Use this when issuing multiple generation calls so weight decoding is
    /// paid once instead of on every request.
    pub fn native_decoder_session(&self) -> Result<NativeDecoderSession> {
        self.native_decoder_session_with_options(&NativeDecoderWeightOptions::default())
    }

    /// Load a resident native decoder session with selected RSMF tensor
    /// variants.
    pub fn native_decoder_session_with_options(
        &self,
        options: &NativeDecoderWeightOptions,
    ) -> Result<NativeDecoderSession> {
        Ok(NativeDecoderSession {
            weights: self.native_decoder_weights_with_options(options)?,
            tokenizer: self.native_decoder_tokenizer()?,
            prefix_cache: Default::default(),
        })
    }

    /// Load the native decoder tokenizer from `tokenizer.json`.
    ///
    /// Optional `tokenizer_config.json` and `chat_template.json` assets attach a
    /// supported chat template when present.
    pub fn native_decoder_tokenizer(&self) -> Result<NativeDecoderTokenizer> {
        let tokenizer_asset = self
            .file
            .asset(NATIVE_DECODER_TOKENIZER_ASSET)
            .ok_or_else(|| RuntimeError::NativeDecoderAssetMissing {
                asset_name: NATIVE_DECODER_TOKENIZER_ASSET.to_string(),
            })?;
        let tokenizer_config = self
            .file
            .asset(NATIVE_DECODER_TOKENIZER_CONFIG_ASSET)
            .map(|asset| asset.bytes);
        let chat_template = self
            .file
            .asset(NATIVE_DECODER_CHAT_TEMPLATE_ASSET)
            .map(|asset| asset.bytes);
        NativeDecoderTokenizer::from_json_with_assets(
            tokenizer_asset.bytes,
            tokenizer_config,
            chat_template,
        )
    }

    /// Run native decoder token-id generation.
    ///
    /// Defaults preserve greedy CPU-reference generation. Sampling options can
    /// enable deterministic top-k/top-p sampling. `Accelerated` resolves to
    /// Apple Accelerate on macOS when the `apple-accelerate` feature is enabled
    /// and otherwise falls back to CPU reference.
    pub fn native_decoder_greedy_decode(
        &self,
        input_token_ids: &[i64],
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderGenerateOutput> {
        let backend = resolve_native_decoder_backend(options.backend)?;
        let weights = self.native_decoder_weights_with_options(&options.weight_options)?;
        native_decoder_generate_with_backend(&weights, input_token_ids, options, backend)
    }

    /// Generate text with the native decoder tokenizer and token-id runtime.
    ///
    /// The prompt is encoded by [`NativeDecoderTokenizer`], passed to
    /// [`Self::native_decoder_greedy_decode`], and generated token ids are
    /// decoded back to text.
    pub fn native_decoder_generate_text(
        &self,
        prompt: &str,
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderTextGenerateOutput> {
        let weight_options = options.weight_options.clone();
        self.native_decoder_session_with_options(&weight_options)?
            .generate_text(prompt, options)
    }

    /// Generate text from role/content chat messages using the tokenizer's
    /// supported chat template.
    pub fn native_decoder_generate_chat(
        &self,
        messages: &[NativeDecoderChatMessage],
        options: NativeDecoderRunOptions,
    ) -> Result<NativeDecoderTextGenerateOutput> {
        let weight_options = options.weight_options.clone();
        self.native_decoder_session_with_options(&weight_options)?
            .generate_chat(messages, options)
    }

    /// Compare native decoder logits against a caller-supplied reference.
    ///
    /// This is intended for local tiny-model fixtures or exported reference
    /// logits from another runtime. The check feeds `input_token_ids` one step
    /// at a time and compares each next-token logits row against
    /// `expected_logits`.
    pub fn native_decoder_check_reference_logits(
        &self,
        check: NativeDecoderReferenceLogitCheck,
    ) -> Result<NativeDecoderReferenceLogitReport> {
        let backend = resolve_native_decoder_backend(check.backend)?;
        let weights = self.native_decoder_weights_with_options(&check.weight_options)?;
        native_decoder_check_reference_logits(&weights, check, backend)
    }

    /// Create a fresh default ONNX Runtime session for compatibility with the
    /// original runtime API.
    ///
    /// Prefer [`Self::session_handle`] for production callers that want cached
    /// sessions and metadata.
    pub fn session(&self, graph_idx: usize) -> Result<Session> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::session", graph_idx).entered();

        self.build_session(graph_idx, &SessionOptions::default())
            .map(|built| built.session)
    }

    /// Return a cached session handle for `graph_idx` and `options`.
    pub fn session_handle(
        &self,
        graph_idx: usize,
        options: SessionOptions,
    ) -> Result<SessionHandle> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::session_handle", graph_idx).entered();

        let key = SessionKey::new(graph_idx, options);
        let mut cache = self.cache.lock().map_err(|_| RuntimeError::CachePoisoned)?;
        if let Some(handle) = cache.get(&key) {
            return Ok(handle.clone());
        }

        let built = self.build_session(key.graph_idx, &key.options)?;
        let session = built.session;
        let handle = SessionHandle {
            inputs: session
                .inputs()
                .iter()
                .map(ValueInfo::from_outlet)
                .collect(),
            outputs: session
                .outputs()
                .iter()
                .map(ValueInfo::from_outlet)
                .collect(),
            session: Arc::new(Mutex::new(session)),
            memory_report: built.memory_report,
            key: key.clone(),
        };
        cache.insert(key, handle.clone());
        Ok(handle)
    }

    /// Run a graph using a cached session.
    pub fn run(
        &self,
        graph_idx: usize,
        options: SessionOptions,
        inputs: RuntimeInputs,
    ) -> Result<RuntimeOutputs> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::run", graph_idx).entered();

        self.session_handle(graph_idx, options)?.run(inputs)
    }

    pub(crate) fn run_with_cancellations(
        &self,
        graph_idx: usize,
        options: SessionOptions,
        inputs: RuntimeInputs,
        cancellations: &[&RuntimeCancellationToken],
    ) -> Result<RuntimeOutputs> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("Engine::run_with_cancellations", graph_idx).entered();

        self.session_handle(graph_idx, options)?
            .run_with_cancellations(inputs, cancellations)
    }

    /// Convenience helper for F32 models.
    pub fn run_f32(
        &self,
        graph_idx: usize,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        self.run_f32_with_options(graph_idx, SessionOptions::default(), inputs)
    }

    /// Convenience helper for F32 models with explicit session options.
    pub fn run_f32_with_options(
        &self,
        graph_idx: usize,
        options: SessionOptions,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        let runtime_inputs = inputs
            .into_iter()
            .map(|(name, array)| {
                let shape = array.shape().to_vec();
                let data = array.iter().copied().collect();
                (name, RuntimeTensor::F32 { shape, data })
            })
            .collect();
        let outputs = self.run(graph_idx, options, runtime_inputs)?;
        outputs
            .into_iter()
            .map(|(name, tensor)| match tensor {
                RuntimeTensor::F32 { shape, data } => {
                    let array = ArrayD::from_shape_vec(shape, data).map_err(|e| {
                        RuntimeError::Shape(format!("F32 output {name} shape mismatch: {e}"))
                    })?;
                    Ok((name, array))
                }
                other => Err(RuntimeError::UnsupportedDtype(format!(
                    "run_f32 expected F32 output, got {}",
                    runtime_tensor_kind(&other)
                ))),
            })
            .collect()
    }

    fn build_session(&self, graph_idx: usize, options: &SessionOptions) -> Result<BuiltSession> {
        let payloads = self.file.graph_payloads();
        let payload = payloads.get(graph_idx).ok_or(RuntimeError::GraphNotFound {
            graph_idx,
            graph_count: payloads.len(),
        })?;
        self.validate_initializer_bindings(payload.kind, payload.bytes, &options.initializers)?;

        let mut builder = Session::builder().map_err(|e| ort_error("session builder", e))?;
        builder = builder
            .with_optimization_level(options.graph_optimization.into())
            .map_err(|e| ort_error("set graph optimization level", e))?;
        builder = builder
            .with_parallel_execution(options.parallel_execution)
            .map_err(|e| ort_error("set execution mode", e))?;
        builder = builder
            .with_memory_pattern(options.memory_pattern)
            .map_err(|e| ort_error("set memory pattern", e))?;
        builder = builder
            .with_deterministic_compute(options.deterministic_compute)
            .map_err(|e| ort_error("set deterministic compute", e))?;

        if let Some(threads) = options.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|e| ort_error("set intra-op threads", e))?;
        }
        if let Some(threads) = options.inter_threads {
            builder = builder
                .with_inter_threads(threads)
                .map_err(|e| ort_error("set inter-op threads", e))?;
        }

        let execution_providers = options
            .execution_providers
            .iter()
            .map(|provider| match provider {
                ExecutionProvider::Cpu { arena } => CPUExecutionProvider::default()
                    .with_arena_allocator(*arena)
                    .build(),
            })
            .collect::<Vec<_>>();
        if !execution_providers.is_empty() {
            builder = builder
                .with_execution_providers(execution_providers)
                .map_err(|e| ort_error("register execution providers", e))?;
        }

        let mut initializer_reports = Vec::with_capacity(options.initializers.len());
        for binding in &options.initializers {
            let initializer = self.initializer_value(binding)?;
            builder = builder
                .with_external_initializer(&binding.initializer_name, initializer.value)
                .map_err(|e| ort_error("register external initializer", e))?;
            initializer_reports.push(initializer.memory_report);
        }

        let initializer_materialized_bytes =
            initializer_reports.iter().try_fold(0usize, |acc, report| {
                acc.checked_add(report.materialized_bytes).ok_or_else(|| {
                    RuntimeError::Shape("initializer materialized byte count overflow".to_string())
                })
            })?;
        let initializer_source_bytes =
            initializer_reports.iter().try_fold(0usize, |acc, report| {
                acc.checked_add(report.source_bytes).ok_or_else(|| {
                    RuntimeError::Shape("initializer source byte count overflow".to_string())
                })
            })?;
        let initializer_zero_copy_bytes =
            initializer_reports.iter().try_fold(0usize, |acc, report| {
                acc.checked_add(report.zero_copy_bytes).ok_or_else(|| {
                    RuntimeError::Shape("initializer zero-copy byte count overflow".to_string())
                })
            })?;
        let session = builder
            .commit_from_memory(payload.bytes)
            .map_err(|e| ort_error("session creation", e))?;
        let provider_allocator_stats = provider_allocator_stats(session.allocator());
        let provider_allocator_bytes = provider_allocator_stats.max_in_use_bytes().map_or_else(
            || {
                RuntimeMemoryMeasurement::unavailable(
                    "ORT allocator did not report a MaxInUse byte counter",
                )
            },
            RuntimeMemoryMeasurement::available,
        );
        let memory_report = SessionMemoryReport {
            graph_payload_bytes: payload.bytes.len(),
            initializer_materialized_bytes,
            initializer_source_bytes,
            initializer_zero_copy_bytes,
            initializers: initializer_reports,
            provider_allocator_bytes,
            provider_allocator_stats,
            process_resident_set_bytes: current_process_resident_set_bytes(),
            io_binding: options.io_binding,
        };
        Ok(BuiltSession {
            session,
            memory_report,
        })
    }

    fn initializer_value(&self, binding: &InitializerBinding) -> Result<InitializerValue> {
        let view = self.initializer_view(binding)?;
        validate_initializer_view(binding, &view)?;
        let shape = shape_u64_to_usize(view.shape()).map_err(|error| {
            RuntimeError::UnsupportedInitializer {
                initializer_name: binding.initializer_name.clone(),
                tensor_name: binding.tensor_name.clone(),
                reason: error,
            }
        })?;
        let (value, materialized_bytes) = initializer_tensor_from_view(binding, &view, shape)?;
        Ok(InitializerValue {
            value,
            memory_report: InitializerMemoryReport {
                initializer_name: binding.initializer_name.clone(),
                tensor_name: binding.tensor_name.clone(),
                variant_idx: binding.variant_idx,
                materialized_bytes,
                source_bytes: view.bytes().len(),
                zero_copy_bytes: 0,
            },
        })
    }

    fn validate_initializer_bindings(
        &self,
        graph_kind: GraphKind,
        graph_bytes: &[u8],
        bindings: &[InitializerBinding],
    ) -> Result<()> {
        if bindings.is_empty() || graph_kind != GraphKind::Onnx {
            return Ok(());
        }

        let initializers = onnx_initializers(graph_bytes).map_err(|reason| {
            RuntimeError::UnsupportedInitializer {
                initializer_name: "<onnx>".to_string(),
                tensor_name: "<graph>".to_string(),
                reason,
            }
        })?;
        for binding in bindings {
            let Some(info) = initializers.get(&binding.initializer_name) else {
                return Err(RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason: "initializer is not declared by the ONNX graph".to_string(),
                });
            };
            let view = self.initializer_view(binding)?;
            validate_initializer_view(binding, &view)?;
            let expected_data_type = OnnxTensorDataType::from_logical_dtype(view.dtype())
                .ok_or_else(|| RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason: format!("unsupported RSMF initializer dtype {:?}", view.dtype()),
                })?;
            if info.data_type != Some(expected_data_type) {
                return Err(RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason: format!(
                        "ONNX initializer dtype is {}, but RSMF tensor dtype is {}",
                        info.data_type.map_or_else(
                            || "missing".to_string(),
                            |data_type| data_type.name().to_string()
                        ),
                        expected_data_type.name()
                    ),
                });
            }

            let actual_shape = shape_u64_to_i64(view.shape()).map_err(|reason| {
                RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason,
                }
            })?;
            if info.shape != actual_shape {
                return Err(RuntimeError::UnsupportedInitializer {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                    reason: format!(
                        "ONNX initializer shape {:?} does not match RSMF tensor shape {:?}",
                        info.shape, actual_shape
                    ),
                });
            }
        }

        Ok(())
    }

    fn initializer_view<'a>(&'a self, binding: &InitializerBinding) -> Result<TensorView<'a>> {
        let result = if let Some(variant_idx) = binding.variant_idx {
            self.file
                .tensor_view_variant(&binding.tensor_name, variant_idx)
        } else {
            self.file.tensor_view(&binding.tensor_name)
        };
        result.map_err(|error| match error {
            RsmfError::NotFound { what } if what == format!("tensor {}", binding.tensor_name) => {
                RuntimeError::InitializerTensorNotFound {
                    initializer_name: binding.initializer_name.clone(),
                    tensor_name: binding.tensor_name.clone(),
                }
            }
            RsmfError::NotFound { what } => RuntimeError::UnsupportedInitializer {
                initializer_name: binding.initializer_name.clone(),
                tensor_name: binding.tensor_name.clone(),
                reason: what,
            },
            other => RuntimeError::Core(other),
        })
    }
}

fn run_with_cpu_io_binding(
    session: &mut Session,
    outputs: &[ValueInfo],
    ort_inputs: Vec<(String, DynValue)>,
    run_options: Option<&RunOptions>,
) -> Result<RuntimeOutputs> {
    let mut binding = session
        .create_binding()
        .map_err(|e| ort_error("create I/O binding", e))?;
    for (name, value) in &ort_inputs {
        binding
            .bind_input(name, value)
            .map_err(|e| ort_error("bind I/O input", e))?;
    }
    let cpu_memory = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )
    .map_err(|e| ort_error("create CPU memory info", e))?;
    for output in outputs {
        binding
            .bind_output_to_device(&output.name, &cpu_memory)
            .map_err(|e| ort_error("bind I/O output", e))?;
    }
    let outputs = if let Some(run_options) = run_options {
        session
            .run_binding_with_options(&binding, run_options)
            .map_err(|e| ort_error("session run with I/O binding", e))?
    } else {
        session
            .run_binding(&binding)
            .map_err(|e| ort_error("session run with I/O binding", e))?
    };
    materialize_outputs(outputs)
}

fn shape_u64_to_usize(shape: &[u64]) -> std::result::Result<Vec<usize>, String> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| format!("dimension {dim} cannot convert to usize"))
        })
        .collect()
}

fn shape_u64_to_i64(shape: &[u64]) -> std::result::Result<Vec<i64>, String> {
    shape
        .iter()
        .map(|&dim| {
            i64::try_from(dim).map_err(|_| format!("dimension {dim} cannot convert to i64"))
        })
        .collect()
}

fn validate_initializer_view(binding: &InitializerBinding, view: &TensorView<'_>) -> Result<()> {
    let reason = if !is_supported_initializer_dtype(view.dtype()) {
        Some(format!("unsupported initializer dtype {:?}", view.dtype()))
    } else if view.encoding != EncodingKind::Raw {
        Some(format!(
            "only raw canonical initializers are supported, got {:?}",
            view.encoding
        ))
    } else if view.layout != LayoutTag::RowMajor {
        Some(format!(
            "only row-major initializers are supported, got {:?}",
            view.layout
        ))
    } else if view.storage_dtype != StorageDtype::Logical(view.dtype()) {
        Some(format!(
            "initializer storage {:?} does not match logical dtype {:?}",
            view.storage_dtype,
            view.dtype()
        ))
    } else {
        None
    };

    if let Some(reason) = reason {
        Err(RuntimeError::UnsupportedInitializer {
            initializer_name: binding.initializer_name.clone(),
            tensor_name: binding.tensor_name.clone(),
            reason,
        })
    } else {
        Ok(())
    }
}

fn is_supported_initializer_dtype(dtype: LogicalDtype) -> bool {
    matches!(
        dtype,
        LogicalDtype::F32
            | LogicalDtype::F64
            | LogicalDtype::I64
            | LogicalDtype::I32
            | LogicalDtype::U8
            | LogicalDtype::I8
            | LogicalDtype::Bool
    )
}

fn initializer_tensor_from_view(
    binding: &InitializerBinding,
    view: &TensorView<'_>,
    shape: Vec<usize>,
) -> Result<(Arc<DynValue>, usize)> {
    match view.dtype() {
        LogicalDtype::F32 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<f32>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::F64 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<f64>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::I64 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<i64>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::I32 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<i32>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::U8 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<u8>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::I8 => initializer_tensor_from_vec(
            binding,
            &shape,
            view.to_vec::<i8>()
                .map_err(|source| unsupported_initializer(binding, source.to_string()))?,
        ),
        LogicalDtype::Bool => {
            let data = view
                .bytes()
                .iter()
                .map(|&byte| byte != 0)
                .collect::<Vec<_>>();
            initializer_tensor_from_vec(binding, &shape, data)
        }
        other => Err(RuntimeError::UnsupportedInitializer {
            initializer_name: binding.initializer_name.clone(),
            tensor_name: binding.tensor_name.clone(),
            reason: format!("unsupported initializer dtype {other:?}"),
        }),
    }
}

fn unsupported_initializer(binding: &InitializerBinding, reason: String) -> RuntimeError {
    RuntimeError::UnsupportedInitializer {
        initializer_name: binding.initializer_name.clone(),
        tensor_name: binding.tensor_name.clone(),
        reason,
    }
}

fn initializer_tensor_from_vec<T>(
    binding: &InitializerBinding,
    shape: &[usize],
    data: Vec<T>,
) -> Result<(Arc<DynValue>, usize)>
where
    T: ort::value::PrimitiveTensorElementType + Clone + Debug + 'static,
{
    let materialized_bytes = data
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| RuntimeError::UnsupportedInitializer {
            initializer_name: binding.initializer_name.clone(),
            tensor_name: binding.tensor_name.clone(),
            reason: "initializer materialized byte count overflow".to_string(),
        })?;
    let value = tensor_from_vec(shape.to_vec(), data).map(Arc::new)?;
    Ok((value, materialized_bytes))
}
