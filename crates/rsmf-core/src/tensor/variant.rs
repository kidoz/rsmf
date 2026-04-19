//! Variant and target-tag types plus the structured variant metadata.

use crate::checksum::CHECKSUM_LEN;
use crate::error::{Result, RsmfError};
use crate::tensor::dtype::{DTYPE_NONE, StorageDtype};

/// Backend / capability tag carried on a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum TargetTag {
    /// The canonical little-endian, row-major, uncompressed variant. Every
    /// tensor has exactly one.
    Canonical = 0,
    /// Generic CPU — works on any modern CPU target.
    CpuGeneric = 1,
    /// x86_64 with AVX2.
    CpuAvx2 = 2,
    /// x86_64 with AVX-512.
    CpuAvx512 = 3,
    /// ARMv8 NEON.
    CpuNeon = 4,
    /// WGPU-compatible layout (portable GPU).
    Wgpu = 5,
    /// CUDA-specific layout.
    Cuda = 6,
    /// Metal-specific layout.
    Metal = 7,
}

impl TargetTag {
    /// Decode from the on-disk discriminant.
    pub fn from_raw(raw: u16) -> Result<Self> {
        Ok(match raw {
            0 => Self::Canonical,
            1 => Self::CpuGeneric,
            2 => Self::CpuAvx2,
            3 => Self::CpuAvx512,
            4 => Self::CpuNeon,
            5 => Self::Wgpu,
            6 => Self::Cuda,
            7 => Self::Metal,
            other => {
                return Err(RsmfError::structural(format!("unknown target tag {other}")));
            }
        })
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::CpuGeneric => "cpu_generic",
            Self::CpuAvx2 => "cpu_avx2",
            Self::CpuAvx512 => "cpu_avx512",
            Self::CpuNeon => "cpu_neon",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }

    /// Return the arena grouping for this target tag. Packed variants are
    /// placed in a `PackedArena` per group so that CPU and GPU data can live
    /// in separate sections with independent alignment requirements.
    #[must_use]
    pub fn arena_group(self) -> ArenaGroup {
        match self {
            Self::Canonical => ArenaGroup::Cpu, // not used for packed, but defensively mapped
            Self::CpuGeneric | Self::CpuAvx2 | Self::CpuAvx512 | Self::CpuNeon => ArenaGroup::Cpu,
            Self::Wgpu => ArenaGroup::Wgpu,
            Self::Cuda => ArenaGroup::Cuda,
            Self::Metal => ArenaGroup::Metal,
        }
    }
}

/// Arena group used to bucket packed variants into separate `PackedArena`
/// sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArenaGroup {
    /// All CPU-targeted variants (generic, AVX2, AVX-512, NEON).
    Cpu = 0,
    /// WGPU-targeted variants.
    Wgpu = 1,
    /// CUDA-targeted variants.
    Cuda = 2,
    /// Metal-targeted variants.
    Metal = 3,
}

/// Kind of encoding transformation applied to get from logical tensor to
/// storage bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum EncodingKind {
    /// No transformation; storage bytes equal logical bytes.
    Raw = 0,
    /// f32 logical → f16 storage. Lossy. Decoded on load.
    CastF16 = 1,
    /// Block-quantized layout (e.g. INT8 with F16 scales). Lossy.
    BlockQuantized = 2,
}

impl EncodingKind {
    /// Decode from the on-disk discriminant.
    pub fn from_raw(raw: u16) -> Result<Self> {
        Ok(match raw {
            0 => Self::Raw,
            1 => Self::CastF16,
            2 => Self::BlockQuantized,
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown encoding kind {other}"
                )));
            }
        })
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::CastF16 => "cast_f16",
            Self::BlockQuantized => "block_quantized",
        }
    }
}

/// Storage layout tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum LayoutTag {
    /// Row-major contiguous.
    RowMajor = 0,
    /// Blocked layout; the block shape is carried in [`VariantMeta::block_shape`].
    Blocked = 1,
}

impl LayoutTag {
    /// Decode from the on-disk discriminant.
    pub fn from_raw(raw: u16) -> Result<Self> {
        Ok(match raw {
            0 => Self::RowMajor,
            1 => Self::Blocked,
            other => {
                return Err(RsmfError::structural(format!("unknown layout tag {other}")));
            }
        })
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::RowMajor => "row_major",
            Self::Blocked => "blocked",
        }
    }
}

/// Structured metadata for packed / quantized variants. Always present in the
/// descriptor, possibly empty.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VariantMeta {
    /// Block shape (empty means none).
    pub block_shape: Vec<u64>,
    /// Group size for grouped quantization. `0` means not used.
    pub group_size: u32,
    /// Storage dtype for scale factors (`None` ≡ not used).
    pub scale_dtype: Option<StorageDtype>,
    /// Storage dtype for zero points (`None` ≡ not used).
    pub zero_point_dtype: Option<StorageDtype>,
    /// Free-form metadata map.
    pub extra: Vec<(String, String)>,
}

impl VariantMeta {
    /// On-disk sentinel for an absent optional dtype.
    pub(crate) const NONE_DTYPE_SENTINEL: u16 = DTYPE_NONE;
}

/// Owning, parsed variant descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantDescriptor {
    /// Target backend / capability tag.
    pub target: TargetTag,
    /// Encoding kind.
    pub encoding: EncodingKind,
    /// Storage dtype.
    pub storage_dtype: StorageDtype,
    /// Layout tag.
    pub layout: LayoutTag,
    /// Required payload alignment (power of two).
    pub alignment: u32,
    /// Offset inside the owning arena section.
    pub section_relative_offset: u64,
    /// Length of the variant payload in bytes.
    pub length: u64,
    /// Truncated BLAKE3 of the variant payload.
    pub checksum: [u8; CHECKSUM_LEN],
    /// Which kind of arena owns the payload. Either
    /// [`crate::section::SectionKind::CanonicalArena`] or
    /// [`crate::section::SectionKind::PackedArena`]; stored as a raw u8 here
    /// so it matches the wire layout.
    pub section_kind: u8,
    /// Index among sections of `section_kind` (0-based).
    pub section_index: u8,
    /// Structured metadata.
    pub meta: VariantMeta,
}

impl VariantDescriptor {
    /// Shorthand for building a canonical `Raw` variant in the canonical arena.
    #[must_use]
    pub fn canonical_raw(
        storage_dtype: StorageDtype,
        alignment: u32,
        section_relative_offset: u64,
        length: u64,
        checksum: [u8; CHECKSUM_LEN],
    ) -> Self {
        Self {
            target: TargetTag::Canonical,
            encoding: EncodingKind::Raw,
            storage_dtype,
            layout: LayoutTag::RowMajor,
            alignment,
            section_relative_offset,
            length,
            checksum,
            section_kind: crate::section::SectionKind::CanonicalArena as u8,
            section_index: 0,
            meta: VariantMeta::default(),
        }
    }
}
