//! Test requirements #2, #3, #4: corrupt header, bad offsets/overlaps,
//! checksum mismatch.

mod common;

use memmap2::Mmap;
use rsmf_core::{RsmfError, RsmfFile, Validator};
use std::fs::OpenOptions;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn corrupt_magic_rejected() {
    let mut bytes = common::build_basic_file_bytes();
    bytes[0] = 0xFF;
    bytes[1] = 0xFF;
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let err = RsmfFile::open(tmp.path()).unwrap_err();
    assert!(matches!(err, RsmfError::InvalidMagic { .. }), "got {err:?}");
}

#[test]
fn bumped_major_version_rejected() {
    // A raw copy of a valid file with `major` bumped. The preamble checksum
    // will re-fail but our reader rejects the version first if the checksum
    // passes; either outcome is acceptable. Here we corrupt the major byte
    // AND recompute the checksum so the UnsupportedVersion error path runs.
    use rsmf_core::preamble::{PREAMBLE_CHECKSUMMED_PREFIX, PREAMBLE_LEN};

    let mut bytes = common::build_basic_file_bytes();
    bytes[8] = 99; // major low byte
    bytes[9] = 0; // major high byte
    // Recompute preamble checksum over the prefix so version check runs first.
    let full = blake3::hash(&bytes[..PREAMBLE_CHECKSUMMED_PREFIX]);
    bytes[PREAMBLE_CHECKSUMMED_PREFIX..PREAMBLE_LEN as usize]
        .copy_from_slice(&full.as_bytes()[..8]);

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let err = RsmfFile::open(tmp.path()).unwrap_err();
    assert!(
        matches!(err, RsmfError::UnsupportedVersion { major: 99, .. }),
        "got {err:?}"
    );
}

#[test]
fn overlapping_sections_rejected() {
    // Hand-build a section table with two sections whose byte ranges
    // overlap. We write a valid file first, then corrupt the section table.
    use rsmf_core::preamble::PREAMBLE_LEN;
    use rsmf_core::section::SECTION_DESC_LEN;

    let mut bytes = common::build_basic_file_bytes();
    // The canonical-arena section is the second section in our fixture (index 1).
    // We'll flip its offset to the manifest section's offset so they overlap.
    let table_off = PREAMBLE_LEN as usize;
    let manifest_entry_off = table_off;
    let canonical_entry_off = table_off + SECTION_DESC_LEN as usize;

    // offset field lives at +8 within each 64-byte entry.
    let manifest_offset_field = manifest_entry_off + 8;
    let canonical_offset_field = canonical_entry_off + 8;
    // Read the manifest's offset.
    let mut manifest_off_bytes = [0u8; 8];
    manifest_off_bytes.copy_from_slice(&bytes[manifest_offset_field..manifest_offset_field + 8]);
    // Point the canonical entry at it.
    bytes[canonical_offset_field..canonical_offset_field + 8].copy_from_slice(&manifest_off_bytes);

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let err = RsmfFile::open(tmp.path()).unwrap_err();
    assert!(matches!(err, RsmfError::Structural(_)), "got {err:?}");
}

#[test]
fn out_of_bounds_offset_rejected() {
    use rsmf_core::preamble::PREAMBLE_LEN;
    use rsmf_core::section::SECTION_DESC_LEN;

    let mut bytes = common::build_basic_file_bytes();
    let table_off = PREAMBLE_LEN as usize;
    let canonical_entry_off = table_off + SECTION_DESC_LEN as usize;
    let canonical_offset_field = canonical_entry_off + 8;
    // Put the offset wildly beyond EOF.
    bytes[canonical_offset_field..canonical_offset_field + 8]
        .copy_from_slice(&(u64::MAX - 1).to_le_bytes());

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    let err = RsmfFile::open(tmp.path()).unwrap_err();
    assert!(matches!(err, RsmfError::Structural(_)), "got {err:?}");
}

#[test]
fn checksum_mismatch_detected_only_on_full() {
    // Write a valid file, then corrupt ONE byte inside the canonical arena
    // (i.e. NOT in the preamble nor the section table nor the manifest).
    // Structural validation passes, full verification fails.
    let bytes = common::build_basic_file_bytes();
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &bytes).unwrap();
    // Open once to locate canonical arena offset.
    let file = RsmfFile::open(tmp.path()).unwrap();
    let canonical = file
        .sections()
        .iter()
        .find(|s| s.kind == rsmf_core::SectionKind::CanonicalArena)
        .unwrap();
    let canonical_off = canonical.offset;
    drop(file);

    // Corrupt one byte in the canonical arena.
    {
        let mut f = OpenOptions::new()
            .write(true)
            .read(true)
            .open(tmp.path())
            .unwrap();
        use std::io::{Seek, SeekFrom, Write};
        f.seek(SeekFrom::Start(canonical_off)).unwrap();
        f.write_all(&[0xFFu8]).unwrap();
        f.flush().unwrap();
    }

    let file = RsmfFile::open(tmp.path()).expect("reopen");
    Validator::structural(&file).expect("structural still ok");
    let err = Validator::full(&file).unwrap_err();
    assert!(
        matches!(err, RsmfError::ChecksumMismatch { .. }),
        "got {err:?}"
    );
}

#[test]
fn file_smaller_than_preamble_rejected() {
    let tiny: [u8; 32] = [0u8; 32];
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&tiny).unwrap();
    tmp.flush().unwrap();
    let err = RsmfFile::open(tmp.path()).unwrap_err();
    assert!(
        matches!(
            err,
            RsmfError::InvalidMagic { .. } | RsmfError::Structural(_)
        ),
        "got {err:?}"
    );
}

// Keeps the `Mmap` import exercised so new tests can use it without a warning.
#[allow(dead_code)]
fn _mmap_marker(_m: Mmap) {}
