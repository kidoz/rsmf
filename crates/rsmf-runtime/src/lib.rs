//! Production-oriented ONNX Runtime integration for RSMF graph payloads.
//!
//! `rsmf-runtime` keeps RSMF's storage/container boundary intact: graph bytes
//! remain opaque ONNX / ORT payloads, while this crate owns session lifecycle,
//! typed request values, runtime options, and session caching.

mod allocator_stats;
mod engine;
mod error;
mod executor;
mod native_decoder;
mod network;
mod onnx;
mod session;
mod tensor;

pub use engine::{Engine, SessionHandle};
pub use error::{Result, RuntimeError};
pub use executor::{
    DynamicBatchingConfig, RuntimeAdmissionConfig, RuntimeCancellationResult,
    RuntimeCancellationToken, RuntimeExecutor, RuntimeExecutorConfig, RuntimeExecutorMetrics,
    RuntimeMemoryPressureConfig, RuntimeMemoryPressureLevel, RuntimeRequest, RuntimeRequestHandle,
    RuntimeRequestTimings, RuntimeResponse, RuntimeTenantMetrics,
};
pub use native_decoder::*;
pub use network::{
    RUNTIME_NETWORK_PROTOCOL_VERSION, RuntimeNetworkMetrics, RuntimeNetworkRunRequest,
    RuntimeNetworkRunResponse, RuntimeNetworkServer, RuntimeNetworkServerConfig,
    RuntimeNetworkServerHandle, RuntimeNetworkTimings,
};
pub use session::{
    ExecutionProvider, GraphOptimizationLevel, InitializerBinding, InitializerMemoryReport,
    IoBindingPolicy, RuntimeAllocatorStat, RuntimeAllocatorStats, RuntimeCapability,
    RuntimeCapabilityReport, RuntimeMemoryMeasurement, SessionKey, SessionMemoryReport,
    SessionOptions, ValueInfo,
};
pub use tensor::{RuntimeInputs, RuntimeOutputs, RuntimeTensor, tensor_view_to_ndarray};

pub(crate) fn ort_error(stage: &'static str, error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Ort {
        stage,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests;
