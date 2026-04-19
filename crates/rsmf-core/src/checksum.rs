//! BLAKE3 helpers.
//!
//! Two flavours are used on disk:
//!
//! * **128-bit (16 bytes)** — stored in section, variant, graph, and asset
//!   descriptors for targeted integrity checks.
//! * **64-bit (8 bytes)** — stored in the preamble as a tamper-evident seal
//!   over the fixed header bytes.
//!
//! Both are derived as a left-truncation of the full 256-bit BLAKE3 digest, so
//! a normal BLAKE3 implementation can reproduce them without extensions.

use blake3::Hasher;

/// Length, in bytes, of the truncated BLAKE3 value stored in section /
/// variant / graph / asset descriptors.
pub const CHECKSUM_LEN: usize = 16;

/// Length, in bytes, of the truncated BLAKE3 value embedded in the preamble.
pub const PREAMBLE_CHECKSUM_LEN: usize = 8;

/// Compute the 128-bit (16-byte) BLAKE3 digest of `bytes`.
#[must_use]
pub fn digest_128(bytes: &[u8]) -> [u8; CHECKSUM_LEN] {
    let full = blake3::hash(bytes);
    let full = full.as_bytes();
    let mut out = [0u8; CHECKSUM_LEN];
    out.copy_from_slice(&full[..CHECKSUM_LEN]);
    out
}

/// Compute the 64-bit (8-byte) BLAKE3 digest of `bytes`.
#[must_use]
pub fn digest_64(bytes: &[u8]) -> [u8; PREAMBLE_CHECKSUM_LEN] {
    let full = blake3::hash(bytes);
    let full = full.as_bytes();
    let mut out = [0u8; PREAMBLE_CHECKSUM_LEN];
    out.copy_from_slice(&full[..PREAMBLE_CHECKSUM_LEN]);
    out
}

/// Streaming 128-bit BLAKE3 helper. Useful for hashing large tensor payloads
/// without materialising the entire buffer first.
#[derive(Default)]
pub struct StreamingDigest {
    hasher: Hasher,
}

impl StreamingDigest {
    /// Create a new, empty streaming hasher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hasher: Hasher::new(),
        }
    }

    /// Feed more bytes into the digest.
    pub fn update(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }

    /// Finalise and return the 128-bit truncated digest.
    #[must_use]
    pub fn finalize_128(self) -> [u8; CHECKSUM_LEN] {
        let full = self.hasher.finalize();
        let full = full.as_bytes();
        let mut out = [0u8; CHECKSUM_LEN];
        out.copy_from_slice(&full[..CHECKSUM_LEN]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_128_is_first_16_bytes_of_full() {
        let data = b"hello rsmf";
        let full = blake3::hash(data);
        let trunc = digest_128(data);
        assert_eq!(&trunc[..], &full.as_bytes()[..16]);
    }

    #[test]
    fn digest_64_is_first_8_bytes_of_full() {
        let data = b"preamble";
        let full = blake3::hash(data);
        let trunc = digest_64(data);
        assert_eq!(&trunc[..], &full.as_bytes()[..8]);
    }

    #[test]
    fn streaming_matches_oneshot() {
        let data = b"some bytes to hash";
        let mut s = StreamingDigest::new();
        s.update(&data[..4]);
        s.update(&data[4..]);
        assert_eq!(s.finalize_128(), digest_128(data));
    }
}
