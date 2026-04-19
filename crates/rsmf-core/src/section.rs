//! Section table entries.
//!
//! The section table is an array of fixed-size 64-byte entries pointing at
//! file regions. See `docs/SPEC.md` §3.

use crate::checksum::CHECKSUM_LEN;
use crate::error::{Result, RsmfError};

/// Size of one section-table entry, in bytes.
pub const SECTION_DESC_LEN: u64 = 64;

/// Kind of a section. Values match the wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum SectionKind {
    /// The binary manifest.
    Manifest = 1,
    /// The canonical tensor arena.
    CanonicalArena = 2,
    /// A packed tensor arena (zero or more per file).
    PackedArena = 3,
    /// Opaque graph payload (ONNX / ORT bytes).
    Graph = 4,
    /// Concatenated named assets.
    Assets = 5,
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
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown section kind {other}"
                )));
            }
        })
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
        }
    }
}

/// Section flag: payload bytes are zstd-compressed on disk.
///
/// Indicates that the section payload is zstd-compressed on disk.
pub const SECTION_FLAG_COMPRESSED: u32 = 0x1;

/// Bit mask of all defined flag bits. Unknown bits are rejected at parse time.
pub const SECTION_FLAG_KNOWN_MASK: u32 = SECTION_FLAG_COMPRESSED;

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

    /// Encode this entry into 64 bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; SECTION_DESC_LEN as usize] {
        let mut buf = [0u8; SECTION_DESC_LEN as usize];
        buf[0..2].copy_from_slice(&(self.kind as u16).to_le_bytes());
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
}
