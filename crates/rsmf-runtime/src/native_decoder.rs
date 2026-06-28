use std::collections::HashMap;

use rsmf_core::tensor::variant::LayoutTag;
use rsmf_core::{LogicalDtype, RsmfError, RsmfFile, TensorDescriptor, TensorView};
use serde::{Deserialize, Serialize};

use crate::{Result, RuntimeError};

mod backends;
mod contract;
mod generation;
mod kv_cache;
mod ops;
mod options;
mod prefix_cache;
mod sampling;
mod scheduler;
mod tokenizer;
mod weights;

pub use contract::{
    NATIVE_DECODER_CHAT_TEMPLATE_ASSET, NATIVE_DECODER_CONFIG_ASSET,
    NATIVE_DECODER_GENERATION_CONFIG_ASSET, NATIVE_DECODER_TOKENIZER_ASSET,
    NATIVE_DECODER_TOKENIZER_CONFIG_ASSET, NativeDecoderAssets, NativeDecoderConfig,
    NativeDecoderContract, NativeDecoderFamily, NativeDecoderTensorBinding,
};
pub use generation::{
    NativeDecoderGenerateOutput, NativeDecoderResidencyReport, NativeDecoderSession,
    NativeDecoderStepOutput, NativeDecoderTextGenerateOutput,
};
pub use kv_cache::NativeDecoderKvCache;
pub use ops::{
    NativeDecoderCpuBlockInput, NativeDecoderCpuBlockOutput, NativeDecoderCpuLayerWeights,
    native_decoder_cpu_apply_llama_rope, native_decoder_cpu_cached_attention,
    native_decoder_cpu_causal_attention, native_decoder_cpu_linear, native_decoder_cpu_llama_block,
    native_decoder_cpu_rms_norm, native_decoder_cpu_silu,
};
pub use options::{
    NativeDecoderAttentionImplementation, NativeDecoderBackend, NativeDecoderPerformanceOptions,
    NativeDecoderReferenceLogitCheck, NativeDecoderReferenceLogitReport, NativeDecoderRunOptions,
    NativeDecoderSamplingOptions, NativeDecoderWeightOptions,
};
pub use prefix_cache::{NativeDecoderPerformanceReport, NativeDecoderPrefixCacheReport};
pub use scheduler::{
    NativeDecoderContinuousBatchOutput, NativeDecoderContinuousBatchReport,
    NativeDecoderContinuousBatchRequest,
};
pub use tokenizer::{NativeDecoderChatMessage, NativeDecoderTokenizer};
pub use weights::{NativeDecoderLayerWeights, NativeDecoderWeights};

pub(crate) use backends::*;
pub(crate) use contract::*;
pub(crate) use generation::*;
pub(crate) use kv_cache::*;
pub(crate) use ops::*;
pub(crate) use prefix_cache::*;
pub(crate) use sampling::*;
pub(crate) use weights::*;
