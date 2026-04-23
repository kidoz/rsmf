//! Property-based coverage for malformed-input parsers.
//!
//! The guarantees we care about are:
//! * No input byte sequence, however malformed, may cause the parser
//!   to panic or trigger a debug-mode arithmetic check.
//! * Every failure surfaces as a typed [`RsmfError`] — never a raw
//!   `io::Error`, never an index-out-of-bounds, never a `thread 'main'
//!   panicked at ...`.
//!
//! Any regression here would trip one of the asserts below or surface
//! as a proptest shrink report pointing at the offending bytes.

use proptest::prelude::*;
use rsmf_core::manifest::Manifest;
use rsmf_core::preamble::{MAGIC, PREAMBLE_LEN, Preamble};
use rsmf_core::section::{SECTION_DESC_LEN, SectionDescriptor};
use rsmf_core::{RsmfError, RsmfFile};

// -- preamble -----------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        // Each iteration is cheap (no IO), so go wide.
        cases: 512,
        .. ProptestConfig::default()
    })]

    /// Arbitrary bytes fed to `Preamble::decode` must never panic.
    #[test]
    fn preamble_decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..256)) {
        let _ = Preamble::decode(&bytes);
    }

    /// When the input is shorter than `PREAMBLE_LEN`, decode must refuse.
    #[test]
    fn preamble_decode_rejects_short_inputs(bytes in prop::collection::vec(any::<u8>(), 0..(PREAMBLE_LEN as usize))) {
        prop_assert!(Preamble::decode(&bytes).is_err());
    }

    /// Flipping a single bit inside the preamble's checksummed prefix
    /// must cause decode to fail (self-hash mismatch or magic mismatch
    /// or structural field check — all typed errors).
    #[test]
    fn preamble_rejects_bitflip_inside_checksummed_prefix(
        bit in 0usize..(8 * 0x38)
    ) {
        let good = valid_preamble_bytes();
        let mut bad = good;
        bad[bit / 8] ^= 1 << (bit % 8);
        // MAGIC check + header_len + checksum between them cover every
        // bit of the prefix; a bitflip must not produce a valid decode.
        prop_assert!(Preamble::decode(&bad).is_err());
    }
}

fn valid_preamble_bytes() -> [u8; PREAMBLE_LEN as usize] {
    let p = Preamble {
        magic: MAGIC,
        major: 1,
        minor: 0,
        flags: 0,
        header_len: PREAMBLE_LEN,
        section_tbl_off: PREAMBLE_LEN,
        section_tbl_count: 0,
        manifest_off: PREAMBLE_LEN,
        manifest_len: 0,
        preamble_checksum: [0u8; 8],
    };
    p.encode()
}

// -- section descriptor -------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

    /// Arbitrary 64-byte buffers must never panic during decode.
    #[test]
    fn section_decode_never_panics(bytes in prop::array::uniform32(any::<u8>()).prop_flat_map(|a| (Just(a), prop::array::uniform32(any::<u8>())).prop_map(|(a, b)| {
        let mut out = [0u8; SECTION_DESC_LEN as usize];
        out[..32].copy_from_slice(&a);
        out[32..].copy_from_slice(&b);
        out
    }))) {
        match SectionDescriptor::decode(&bytes) {
            Ok(_) | Err(RsmfError::Structural(_)) => {},
            Err(other) => prop_assert!(false, "unexpected error variant: {other:?}"),
        }
    }

    /// Truncated section-table entries must reject with a typed error,
    /// never panic.
    #[test]
    fn section_decode_rejects_truncated_inputs(bytes in prop::collection::vec(any::<u8>(), 0..(SECTION_DESC_LEN as usize))) {
        let err = SectionDescriptor::decode(&bytes).unwrap_err();
        prop_assert!(matches!(err, RsmfError::Structural(_)));
    }
}

// -- manifest body ------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

    /// Arbitrary bytes handed to Manifest::decode must never panic.
    #[test]
    fn manifest_decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
        match Manifest::decode(&bytes) {
            Ok(_) | Err(RsmfError::Structural(_)) => {},
            Err(other) => prop_assert!(false, "unexpected error variant: {other:?}"),
        }
    }
}

// -- whole-file RsmfFile::from_mmap --------------------------------------------

proptest! {
    // Fewer cases — each iteration writes a small temp file and runs
    // the full open path. Still large enough to surface panics quickly.
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

    /// Feed RsmfFile::open arbitrary small files. Must never panic; must
    /// always surface a typed RsmfError or an Io error. Parsing a random
    /// byte blob as a structured model artifact should fail, not crash.
    #[test]
    fn rsmf_file_open_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..1024)) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fuzz.rsmf");
        std::fs::write(&path, &bytes).unwrap();
        match RsmfFile::open(&path) {
            Ok(_) => {},
            Err(RsmfError::InvalidMagic { .. })
            | Err(RsmfError::UnsupportedVersion { .. })
            | Err(RsmfError::Structural(_))
            | Err(RsmfError::ChecksumMismatch { .. })
            | Err(RsmfError::NotFound { .. })
            | Err(RsmfError::Unsupported(_))
            | Err(RsmfError::Io(_))
            | Err(RsmfError::IoWithPath { .. }) => {},
            Err(other) => {
                prop_assert!(false, "unexpected error variant: {other:?}");
            }
        }
    }

    /// When we start from a valid preamble and append random bytes to
    /// fill out the rest of the file, the open path must still not
    /// panic. This exercises section-table + manifest decoders through
    /// the real mmap-backed `open` entry point (not just the in-memory
    /// decoders above).
    #[test]
    fn valid_preamble_plus_random_tail_never_panics(tail in prop::collection::vec(any::<u8>(), 0..2048)) {
        let mut bytes = Vec::with_capacity(tail.len() + PREAMBLE_LEN as usize);
        bytes.extend_from_slice(&valid_preamble_bytes());
        bytes.extend_from_slice(&tail);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fuzz.rsmf");
        std::fs::write(&path, &bytes).unwrap();
        // We don't assert a specific error variant here — the point is
        // *no panic*, not what the error is. A successful open is
        // possible when `tail` happens to form a coherent manifest.
        let _ = RsmfFile::open(&path);
    }
}
