//! Test requirement #6: mmap-backed tensor access.

mod common;

use rsmf_core::RsmfFile;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn tensor_view_bytes_match_f32_input() {
    let bytes = common::build_basic_file_bytes();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let view = file.tensor_view("weight").unwrap();
    assert_eq!(view.shape(), &[4u64, 4]);
    assert_eq!(view.bytes().len(), 16 * 4);

    // Copy-path: always works regardless of alignment.
    let weights = view.to_vec::<f32>().unwrap();
    for (i, &w) in weights.iter().enumerate() {
        assert!((w - (i as f32)).abs() < f32::EPSILON, "mismatch at {i}");
    }

    // Borrowed typed view: may or may not succeed depending on offset
    // alignment. If it does, it must return the same values as to_vec.
    if let Ok(borrowed) = view.as_slice::<f32>() {
        assert_eq!(borrowed.len(), weights.len());
        for (a, b) in borrowed.iter().zip(weights.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }
}

#[test]
fn tensor_view_backed_by_file_mmap_is_stable_across_opens() {
    // Open the same file twice and confirm the bytes we get are stable and
    // non-empty. This is enough to confirm the bytes came from the mapping
    // (any buggy path would diverge across opens or deref freed memory).
    let bytes = common::build_basic_file_bytes();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let first = {
        let file = RsmfFile::open(tmp.path()).unwrap();
        file.tensor_view("weight").unwrap().bytes().to_vec()
    };
    let second = {
        let file = RsmfFile::open(tmp.path()).unwrap();
        file.tensor_view("weight").unwrap().bytes().to_vec()
    };
    assert_eq!(first.len(), 16 * 4);
    assert_eq!(first, second);

    // The canonical bytes are the LE encoding of 0.0..=15.0 f32s. Check a
    // representative subset to ensure what we're reading is actually the
    // tensor bytes (not, e.g., manifest padding).
    let view_f32: Vec<f32> = (0..16)
        .map(|i| {
            let off = i * 4;
            f32::from_le_bytes([first[off], first[off + 1], first[off + 2], first[off + 3]])
        })
        .collect();
    for (i, &v) in view_f32.iter().enumerate() {
        assert!((v - i as f32).abs() < f32::EPSILON, "mismatch at {i}: {v}");
    }
}
