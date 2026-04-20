//! Section table entries.
//!
//! The section table is an array of fixed-size 64-byte entries pointing at
//! file regions. See `docs/SPEC.md` §3.

use crate::checksum::CHECKSUM_LEN;
use crate::error::{Result, RsmfError};

/// Size of one section-table entry, in bytes.
pub const SECTION_DESC_LEN: u64 = 64;

/// First discriminant reserved for vendor / user-defined custom sections.
///
/// Values in `1..CUSTOM_SECTION_RANGE_START` are claimed or reserved by
/// the standard; values `>= CUSTOM_SECTION_RANGE_START` are free for
/// third-party use and MUST be preserved as `SectionKind::Custom(raw)`
/// by every reader, even when the reader doesn't understand what they
/// contain. Compare PNG's "ancillary chunks" case-bit convention.
pub const CUSTOM_SECTION_RANGE_START: u16 = 128;

/// Kind of a section. Known values are named; unknown values in the
/// custom range (`>= 128`) are preserved as `Custom(raw)` so vendor
/// extensions can round-trip through readers that don't understand
/// them. Discriminants in `6..128` are reserved for the standard and
/// rejected with a structural error until promoted to a named variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionKind {
    /// The binary manifest.
    Manifest,
    /// The canonical tensor arena.
    CanonicalArena,
    /// A packed tensor arena (zero or more per file).
    PackedArena,
    /// Opaque graph payload (ONNX / ORT bytes).
    Graph,
    /// Concatenated named assets.
    Assets,
    /// Vendor or user-defined custom section. Its payload is opaque to
    /// the standard reader; the contained `u16` is the wire discriminant
    /// (always `>= CUSTOM_SECTION_RANGE_START`).
    Custom(u16),
}

impl SectionKind {
    /// Try to construct from its on-disk discriminant.
    pub fn from_raw(raw: u16) -> Result<Self> {
        Ok(match raw {
            1 => Self::Manifest,
            2 => Self::CanonicalArena,
            3 => Self::PackedArena,
            4 => Self::Graph,
            5 => Self::Assets,
            n if n >= CUSTOM_SECTION_RANGE_START => Self::Custom(n),
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown section kind {other} \
                     (reserved for future standard use; \
                     values >= {CUSTOM_SECTION_RANGE_START} are free for custom sections)"
                )));
            }
        })
    }

    /// Wire discriminant. Inverse of [`SectionKind::from_raw`].
    #[must_use]
    pub fn to_raw(self) -> u16 {
        match self {
            Self::Manifest => 1,
            Self::CanonicalArena => 2,
            Self::PackedArena => 3,
            Self::Graph => 4,
            Self::Assets => 5,
            Self::Custom(raw) => raw,
        }
    }

    /// Human-readable name for diagnostics.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Manifest => "Manifest",
            Self::CanonicalArena => "CanonicalArena",
            Self::PackedArena => "PackedArena",
            Self::Graph => "Graph",
            Self::Assets => "Assets",
            Self::Custom(_) => "Custom",
        }
    }

    /// Returns `true` if this is a vendor or user-defined custom section.
    #[must_use]
    pub fn is_custom(self) -> bool {
        matches!(self, Self::Custom(_))
    }
}

/// Section flag: payload bytes are zstd-compressed on disk.
pub const SECTION_FLAG_COMPRESSED: u32 = 0x1;

/// Section flag: payload bytes were passed through rsmf-core's
/// bit-shuffle pre-processor before any further encoding / compression.
///
/// The format has always applied bit-shuffling inside compressed F32
/// sections as a zstd ratio optimisation; this flag makes that contract
/// explicit so consumers who want the raw-compressed (un-shuffled)
/// bytes can skip the `un-shuffle` step.
pub const SECTION_FLAG_BIT_SHUFFLED: u32 = 0x2;

/// Bit mask of all defined flag bits. Unknown bits are rejected at parse time.
pub const SECTION_FLAG_KNOWN_MASK: u32 = SECTION_FLAG_COMPRESSED | SECTION_FLAG_BIT_SHUFFLED;

/// One section-table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionDescriptor {
    /// What this section contains.
    pub kind: SectionKind,
    /// Required alignment of the section payload offset. Power of two.
    pub align: u16,
    /// Flags word. Bit 0 = zstd-compressed payload. Unknown bits rejected.
    pub flags: u32,
    /// Absolute offset of the section payload in the file.
    pub offset: u64,
    /// Length of the section payload in bytes.
    pub length: u64,
    /// Truncated BLAKE3 of the section payload.
    pub checksum: [u8; CHECKSUM_LEN],
}

impl SectionDescriptor {
    /// Returns `true` if this section's payload is zstd-compressed.
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.flags & SECTION_FLAG_COMPRESSED != 0
    }

    /// Returns `true` if this section's payload was bit-shuffled before
    /// the final encoding step.
    #[must_use]
    pub fn is_bit_shuffled(&self) -> bool {
        self.flags & SECTION_FLAG_BIT_SHUFFLED != 0
    }

    /// Encode this entry into 64 bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; SECTION_DESC_LEN as usize] {
        let mut buf = [0u8; SECTION_DESC_LEN as usize];
        buf[0..2].copy_from_slice(&self.kind.to_raw().to_le_bytes());
        buf[2..4].copy_from_slice(&self.align.to_le_bytes());
        buf[4..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.offset.to_le_bytes());
        buf[16..24].copy_from_slice(&self.length.to_le_bytes());
        buf[24..40].copy_from_slice(&self.checksum);
        // Bytes 40..64 reserved, left as zeros.
        buf
    }

    /// Decode one entry from `bytes`, which must have length `>= 64`.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SECTION_DESC_LEN as usize {
            return Err(RsmfError::structural(
                "section-table entry truncated".to_string(),
            ));
        }
        let kind_raw = u16::from_le_bytes([bytes[0], bytes[1]]);
        let kind = SectionKind::from_raw(kind_raw)?;
        let align = u16::from_le_bytes([bytes[2], bytes[3]]);
        if align == 0 || (align & (align - 1)) != 0 {
            return Err(RsmfError::structural(format!(
                "section alignment not power of two: {align}"
            )));
        }
        let flags = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let unknown_flags = flags & !SECTION_FLAG_KNOWN_MASK;
        if unknown_flags != 0 {
            return Err(RsmfError::structural(format!(
                "unknown section flags 0x{unknown_flags:08x}"
            )));
        }
        let offset = read_u64(&bytes[8..16]);
        let length = read_u64(&bytes[16..24]);
        let mut checksum = [0u8; CHECKSUM_LEN];
        checksum.copy_from_slice(&bytes[24..24 + CHECKSUM_LEN]);
        for b in &bytes[40..64] {
            if *b != 0 {
                return Err(RsmfError::structural(
                    "section entry reserved bytes not zero".to_string(),
                ));
            }
        }
        Ok(Self {
            kind,
            align,
            flags,
            offset,
            length,
            checksum,
        })
    }
}

fn read_u64(bytes: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let d = SectionDescriptor {
            kind: SectionKind::CanonicalArena,
            align: 64,
            flags: 0,
            offset: 1024,
            length: 4096,
            checksum: [7u8; CHECKSUM_LEN],
        };
        let bytes = d.encode();
        let back = SectionDescriptor::decode(&bytes).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn reject_bad_alignment() {
        let mut bytes = SectionDescriptor {
            kind: SectionKind::Manifest,
            align: 64,
            flags: 0,
            offset: 0,
            length: 1,
            checksum: [0u8; CHECKSUM_LEN],
        }
        .encode();
        bytes[2] = 3; // align = 0x0003 — not a power of two
        bytes[3] = 0;
        let err = SectionDescriptor::decode(&bytes).unwrap_err();
        assert!(matches!(err, RsmfError::Structural(_)));
    }

    #[test]
    fn custom_kind_roundtrips() {
        // SectionKind discriminants >= 128 are the "ancillary" range —
        // readers must preserve them through decode/encode.
        let d = SectionDescriptor {
            kind: SectionKind::Custom(200),
            align: 8,
            flags: 0,
            offset: 2048,
            length: 512,
            checksum: [0xAB; CHECKSUM_LEN],
        };
        let bytes = d.encode();
        let back = SectionDescriptor::decode(&bytes).unwrap();
        assert!(back.kind.is_custom());
        assert_eq!(back.kind.to_raw(), 200);
        assert_eq!(back, d);
    }

    #[test]
    fn kind_in_reserved_future_range_is_rejected() {
        // Discriminants 6..128 are reserved for the standard — an old
        // reader must reject them loudly, not silently accept as Custom.
        let mut bytes = SectionDescriptor {
            kind: SectionKind::Manifest,
            align: 8,
            flags: 0,
            offset: 0,
            length: 1,
            checksum: [0u8; CHECKSUM_LEN],
        }
        .encode();
        bytes[0] = 42; // kind = 42 — reserved for future use
        bytes[1] = 0;
        let err = SectionDescriptor::decode(&bytes).unwrap_err();
        assert!(matches!(err, RsmfError::Structural(_)));
    }

    #[test]
    fn bit_shuffled_flag_roundtrips() {
        let d = SectionDescriptor {
            kind: SectionKind::CanonicalArena,
            align: 64,
            flags: SECTION_FLAG_COMPRESSED | SECTION_FLAG_BIT_SHUFFLED,
            offset: 256,
            length: 4096,
            checksum: [0x11; CHECKSUM_LEN],
        };
        let bytes = d.encode();
        let back = SectionDescriptor::decode(&bytes).unwrap();
        assert!(back.is_compressed());
        assert!(back.is_bit_shuffled());
        assert_eq!(back, d);
    }

    #[test]
    fn unknown_flag_bit_rejected() {
        let mut bytes = SectionDescriptor {
            kind: SectionKind::Manifest,
            align: 8,
            flags: 0,
            offset: 0,
            length: 1,
            checksum: [0u8; CHECKSUM_LEN],
        }
        .encode();
        // Set bit 7 — unknown; bytes 4..8 is the little-endian flags u32.
        bytes[4] = 0x80;
        let err = SectionDescriptor::decode(&bytes).unwrap_err();
        assert!(matches!(err, RsmfError::Structural(_)));
    }
}
