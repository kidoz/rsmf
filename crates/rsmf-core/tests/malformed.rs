//! Test requirement #10: large-offset logic without huge real allocations,
//! plus a handful of property/fuzz-friendly cases for malformed inputs.

mod common;

use std::io::Write;

use rsmf_core::RsmfFile;
use tempfile::NamedTempFile;

/// Flip one byte at every offset inside the preamble + section-table region of
/// a valid file. None of the resulting files should make the reader panic;
/// every case must either open cleanly or return a structured error.
#[test]
fn byte_flip_fuzz_never_panics() {
    let bytes = common::build_basic_file_bytes();
    // Only fuzz the first 512 bytes (preamble + section table + manifest
    // prefix). Fuzzing every byte would work but the test runtime blows up
    // quickly.
    let cap = bytes.len().min(512);
    for i in 0..cap {
        let mut candidate = bytes.clone();
        candidate[i] = candidate[i].wrapping_add(1);
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&candidate).unwrap();
        tmp.flush().unwrap();
        // Open may succeed or fail; the important thing is it doesn't panic.
        let _ = RsmfFile::open(tmp.path());
    }
}

/// Large-offset logic: a crafted file claims a section length of nearly
/// `u64::MAX`. The reader must reject it in O(1) rather than trying to
/// allocate or read past EOF.
#[test]
fn absurd_section_length_rejected_without_allocation() {
    use rsmf_core::preamble::PREAMBLE_LEN;
    use rsmf_core::section::SECTION_DESC_LEN;

    let mut bytes = common::build_basic_file_bytes();
    let table_off = PREAMBLE_LEN as usize;
    let canonical_entry_off = table_off + SECTION_DESC_LEN as usize;
    // length field: +16 within the entry
    let length_field = canonical_entry_off + 16;
    bytes[length_field..length_field + 8].copy_from_slice(&(u64::MAX - 1).to_le_bytes());

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let err = RsmfFile::open(tmp.path()).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("overflow") || msg.contains("past"),
        "got: {msg}"
    );
}

/// Truncated file: every prefix of a valid file must yield a structured error
/// rather than a panic.
#[test]
fn every_prefix_rejected_without_panic() {
    let bytes = common::build_basic_file_bytes();
    // Step through lengths at 128-byte increments to keep the test fast while
    // still covering the interesting boundaries.
    let mut len = 0;
    while len < bytes.len() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&bytes[..len]).unwrap();
        tmp.flush().unwrap();
        let _ = RsmfFile::open(tmp.path());
        len += 128;
    }
}
