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
pub mod moe;
pub mod placement;
pub mod preamble;
pub mod reader;
pub mod section;
pub mod selection;
pub mod streaming_writer;
pub mod tensor;
pub mod tier;
pub mod validator;
pub mod writer;

/// Asynchronous streaming reads
#[cfg(feature = "async_io")]
pub mod async_reader;

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
pub use moe::{MoeEntry, MoeGroup, MoeIndex, MoeRole, moe_index_from_manifest};
pub use placement::{
    DeviceDescriptor, DeviceKind, MemoryTier, PLACEMENT_FLAG_COLD, PLACEMENT_FLAG_PIN,
    PLACEMENT_SECTION_KIND, PLACEMENT_VERSION, PlacementManifest, PlacementRecord,
};
pub use preamble::{FORMAT_MAJOR, FORMAT_MINOR, MAGIC, Preamble};
pub use reader::{AssetRef, CustomSectionPayload, GraphPayload, RsmfFile};
pub use section::{SectionDescriptor, SectionKind};
pub use selection::{
    Capabilities, CpuFeatures, ExecutionMode, GpuBackend, SelectedVariant, TensorPlan,
    select_variants_with_tier,
};
pub use streaming_writer::StreamingRsmfWriter;
pub use tensor::descriptor::TensorDescriptor;
pub use tensor::dtype::{LogicalDtype, StorageDtype};
pub use tensor::variant::{
    ArenaGroup, EncodingKind, LayoutTag, TargetTag, VariantDescriptor, VariantMeta,
};
pub use tensor::view::TensorView;
pub use tier::{TIER_CLASS_KEY, TIER_INTENT_KEY, Tier};
pub use validator::Validator;
pub use writer::{
    AssetInput, CustomSectionInput, GraphInput, RsmfWriter, TensorInput, VariantInput,
    convert_f32_to_f16_bytes,
};
