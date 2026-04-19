//! Fixed-size file preamble.
//!
//! The preamble is the first 64 bytes of every RSMF file. It carries the magic
//! bytes, format version, pointers to the section table and manifest, and a
//! truncated BLAKE3 hash over its own fixed-field prefix. See `docs/SPEC.md`
//! §2 for the authoritative layout.

use crate::checksum::{PREAMBLE_CHECKSUM_LEN, digest_64};
use crate::error::{Result, RsmfError};

/// Magic bytes that start every RSMF file. Exactly 8 bytes.
pub const MAGIC: [u8; 8] = *b"RSMF\0\0\0\x01";

/// Current format major version written by this crate.
pub const FORMAT_MAJOR: u16 = 1;

/// Current format minor version written by this crate.
pub const FORMAT_MINOR: u16 = 0;

/// Length of the preamble on disk, in bytes.
pub const PREAMBLE_LEN: u64 = 64;

/// Size of the fixed-field prefix covered by the preamble checksum.
pub const PREAMBLE_CHECKSUMMED_PREFIX: usize = 0x38;

/// Parsed contents of the preamble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Preamble {
    /// Magic bytes. Must equal [`MAGIC`].
    pub magic: [u8; 8],
    /// Format major version.
    pub major: u16,
    /// Format minor version.
    pub minor: u16,
    /// Reserved flags word. Must be zero in v1.
    pub flags: u32,
    /// Length of this preamble, in bytes. Always equals [`PREAMBLE_LEN`] in v1.
    pub header_len: u64,
    /// Absolute offset of the section table.
    pub section_tbl_off: u64,
    /// Number of section-table entries.
    pub section_tbl_count: u64,
    /// Absolute offset of the manifest payload.
    pub manifest_off: u64,
    /// Length of the manifest payload, in bytes.
    pub manifest_len: u64,
    /// Truncated BLAKE3 hash of the preamble prefix bytes `[0..0x38]`.
    pub preamble_checksum: [u8; PREAMBLE_CHECKSUM_LEN],
}

impl Preamble {
    /// Serialise this preamble into a fixed 64-byte buffer, recomputing the
    /// checksum over the prefix fields.
    #[must_use]
    pub fn encode(mut self) -> [u8; PREAMBLE_LEN as usize] {
        // Zero the stored checksum while we compute it so the bytes we hash
        // are deterministic regardless of the caller.
        self.preamble_checksum = [0u8; PREAMBLE_CHECKSUM_LEN];
        let mut buf = [0u8; PREAMBLE_LEN as usize];
        self.write_fixed_fields(&mut buf);
        let checksum = digest_64(&buf[..PREAMBLE_CHECKSUMMED_PREFIX]);
        buf[PREAMBLE_CHECKSUMMED_PREFIX..PREAMBLE_CHECKSUMMED_PREFIX + PREAMBLE_CHECKSUM_LEN]
            .copy_from_slice(&checksum);
        buf
    }

    /// Parse a 64-byte buffer into a [`Preamble`], verifying magic and checksum.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < PREAMBLE_LEN as usize {
            return Err(RsmfError::structural(format!(
                "file too short for preamble: {} bytes",
                bytes.len()
            )));
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        if magic != MAGIC {
            return Err(RsmfError::InvalidMagic {
                expected: MAGIC,
                got: magic,
            });
        }
        let major = u16::from_le_bytes([bytes[8], bytes[9]]);
        let minor = u16::from_le_bytes([bytes[10], bytes[11]]);
        if major != FORMAT_MAJOR {
            return Err(RsmfError::UnsupportedVersion { major, minor });
        }
        let flags = u32_at(bytes, 12);
        if flags != 0 {
            return Err(RsmfError::structural(format!(
                "preamble flags reserved in v1, got 0x{flags:08x}"
            )));
        }
        let header_len = u64_at(bytes, 16);
        if header_len != PREAMBLE_LEN {
            return Err(RsmfError::structural(format!(
                "preamble header_len must be {PREAMBLE_LEN}, got {header_len}"
            )));
        }
        let section_tbl_off = u64_at(bytes, 24);
        let section_tbl_count = u64_at(bytes, 32);
        let manifest_off = u64_at(bytes, 40);
        let manifest_len = u64_at(bytes, 48);
        let mut preamble_checksum = [0u8; PREAMBLE_CHECKSUM_LEN];
        preamble_checksum.copy_from_slice(
            &bytes
                [PREAMBLE_CHECKSUMMED_PREFIX..PREAMBLE_CHECKSUMMED_PREFIX + PREAMBLE_CHECKSUM_LEN],
        );

        // Recompute the checksum over a buffer that has the checksum field
        // zeroed, then compare.
        let mut recompute = [0u8; PREAMBLE_LEN as usize];
        recompute[..PREAMBLE_CHECKSUMMED_PREFIX]
            .copy_from_slice(&bytes[..PREAMBLE_CHECKSUMMED_PREFIX]);
        let expected = digest_64(&recompute[..PREAMBLE_CHECKSUMMED_PREFIX]);
        if expected != preamble_checksum {
            return Err(RsmfError::ChecksumMismatch {
                kind: "preamble".into(),
            });
        }

        Ok(Self {
            magic,
            major,
            minor,
            flags,
            header_len,
            section_tbl_off,
            section_tbl_count,
            manifest_off,
            manifest_len,
            preamble_checksum,
        })
    }

    fn write_fixed_fields(&self, buf: &mut [u8; PREAMBLE_LEN as usize]) {
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..10].copy_from_slice(&self.major.to_le_bytes());
        buf[10..12].copy_from_slice(&self.minor.to_le_bytes());
        buf[12..16].copy_from_slice(&self.flags.to_le_bytes());
        buf[16..24].copy_from_slice(&self.header_len.to_le_bytes());
        buf[24..32].copy_from_slice(&self.section_tbl_off.to_le_bytes());
        buf[32..40].copy_from_slice(&self.section_tbl_count.to_le_bytes());
        buf[40..48].copy_from_slice(&self.manifest_off.to_le_bytes());
        buf[48..56].copy_from_slice(&self.manifest_len.to_le_bytes());
        // Bytes 56..64 reserved for the checksum; written by `encode`.
    }
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let p = Preamble {
            magic: MAGIC,
            major: FORMAT_MAJOR,
            minor: FORMAT_MINOR,
            flags: 0,
            header_len: PREAMBLE_LEN,
            section_tbl_off: 64,
            section_tbl_count: 3,
            manifest_off: 256,
            manifest_len: 1024,
            preamble_checksum: [0u8; PREAMBLE_CHECKSUM_LEN],
        };
        let encoded = p.encode();
        let decoded = Preamble::decode(&encoded).expect("decode");
        assert_eq!(decoded.major, FORMAT_MAJOR);
        assert_eq!(decoded.minor, FORMAT_MINOR);
        assert_eq!(decoded.section_tbl_off, 64);
        assert_eq!(decoded.manifest_len, 1024);
    }

    #[test]
    fn reject_bad_magic() {
        let mut bytes = [0u8; PREAMBLE_LEN as usize];
        bytes[0..8].copy_from_slice(b"BADMAGIC");
        let err = Preamble::decode(&bytes).unwrap_err();
        assert!(matches!(err, RsmfError::InvalidMagic { .. }));
    }

    #[test]
    fn reject_bad_major() {
        let mut bytes = Preamble {
            magic: MAGIC,
            major: FORMAT_MAJOR,
            minor: 0,
            flags: 0,
            header_len: PREAMBLE_LEN,
            section_tbl_off: 64,
            section_tbl_count: 0,
            manifest_off: 0,
            manifest_len: 0,
            preamble_checksum: [0u8; PREAMBLE_CHECKSUM_LEN],
        }
        .encode();
        // Corrupt the major version, then recompute a fake checksum over the
        // corrupted prefix so decode fails on version rather than checksum.
        bytes[8..10].copy_from_slice(&99u16.to_le_bytes());
        let mut prefix = [0u8; PREAMBLE_CHECKSUMMED_PREFIX];
        prefix.copy_from_slice(&bytes[..PREAMBLE_CHECKSUMMED_PREFIX]);
        let checksum = digest_64(&prefix);
        bytes[PREAMBLE_CHECKSUMMED_PREFIX..PREAMBLE_CHECKSUMMED_PREFIX + PREAMBLE_CHECKSUM_LEN]
            .copy_from_slice(&checksum);
        let err = Preamble::decode(&bytes).unwrap_err();
        assert!(matches!(
            err,
            RsmfError::UnsupportedVersion { major: 99, .. }
        ));
    }

    #[test]
    fn reject_corrupt_checksum() {
        let mut bytes = Preamble {
            magic: MAGIC,
            major: FORMAT_MAJOR,
            minor: 0,
            flags: 0,
            header_len: PREAMBLE_LEN,
            section_tbl_off: 64,
            section_tbl_count: 0,
            manifest_off: 0,
            manifest_len: 0,
            preamble_checksum: [0u8; PREAMBLE_CHECKSUM_LEN],
        }
        .encode();
        bytes[30] ^= 0xFF; // flip a byte inside the checksummed prefix
        let err = Preamble::decode(&bytes).unwrap_err();
        assert!(matches!(err, RsmfError::ChecksumMismatch { .. }));
    }
}
