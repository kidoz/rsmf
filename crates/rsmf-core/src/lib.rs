//! # rsmf-core
//!
//! Core implementation of the Rust Split Model Format (RSMF). Provides the
//! on-disk layout, readers and writers, structural + checksum validation,
//! mmap-backed tensor views, and runtime variant selection.
//!
//! See [`docs/SPEC.md`](../../../docs/SPEC.md) for the authoritative binary
//! specification.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod adapter;
pub mod bit_shuffle;
pub mod checksum;
pub mod error;
pub mod lazy_reader;
pub mod manifest;
pub mod preamble;
pub mod reader;
pub mod section;
pub mod selection;
pub mod streaming_writer;
pub mod tensor;
pub mod validator;
pub mod writer;

#[cfg(feature = "safetensors")]
pub mod safetensors_convert;

#[cfg(feature = "gguf")]
pub mod gguf_convert;

#[cfg(feature = "npy")]
pub mod npy_convert;

pub use adapter::{
    Adapter, AdapterEntry, AdapterIndex, AdapterKind, AdapterRole, adapter_index_from_manifest,
};
pub use error::{Result, RsmfError};
pub use lazy_reader::{LazyRsmfFile, RangeReader, SliceRangeReader};
pub use preamble::{FORMAT_MAJOR, FORMAT_MINOR, MAGIC, Preamble};
pub use reader::{AssetRef, GraphPayload, RsmfFile};
pub use section::{SectionDescriptor, SectionKind};
pub use selection::{
    Capabilities, CpuFeatures, ExecutionMode, GpuBackend, SelectedVariant, TensorPlan,
};
pub use streaming_writer::StreamingRsmfWriter;
pub use tensor::descriptor::TensorDescriptor;
pub use tensor::dtype::{LogicalDtype, StorageDtype};
pub use tensor::variant::{
    ArenaGroup, EncodingKind, LayoutTag, TargetTag, VariantDescriptor, VariantMeta,
};
pub use tensor::view::TensorView;
pub use validator::Validator;
pub use writer::{
    AssetInput, GraphInput, RsmfWriter, TensorInput, VariantInput, convert_f32_to_f16_bytes,
};
