//! Regression test for the cross-tensor variant bug: calling
//! `tensor_view_variant(name, variant_idx)` with a variant index that
//! belongs to a different tensor previously returned that other tensor's
//! bytes under the requested tensor's descriptor. It must now return
//! `RsmfError::NotFound`.
//!
//! The writer emits all canonical variants first, then all packed variants,
//! so for `common::build_basic_file_bytes` (tensors: weight, bias; weight
//! has a wgpu packed variant) the global variant layout is:
//!   0: weight canonical
//!   1: bias   canonical
//!   2: weight wgpu packed
//! Asking for `tensor_view_variant("bias", 0)` or ("bias", 2) must be
//! rejected because those indices are owned by `weight`, not `bias`.

mod common;

use rsmf_core::{RsmfError, RsmfFile};
use std::io::Write;
use tempfile::NamedTempFile;

fn write_fixture_to_disk() -> NamedTempFile {
    let bytes = common::build_basic_file_bytes();
    let mut tmp = NamedTempFile::new().expect("tempfile");
    tmp.write_all(&bytes).expect("write fixture");
    tmp
}

#[test]
fn variant_index_owned_by_another_tensor_is_rejected() {
    let tmp = write_fixture_to_disk();
    let file = RsmfFile::open(tmp.path()).expect("open");

    // Variant 0 (weight canonical) and variant 2 (weight wgpu packed) are
    // owned by `weight`, not `bias`. Both must be rejected.
    for bad_idx in [0u32, 2u32] {
        let err = file
            .tensor_view_variant("bias", bad_idx)
            .expect_err("must reject cross-tensor variant index");
        match err {
            RsmfError::NotFound { what } => {
                assert!(
                    what.contains("bias") && what.contains("does not own"),
                    "unexpected NotFound message: {what}"
                );
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }
}

#[test]
fn variant_index_owned_by_the_named_tensor_is_accepted() {
    let tmp = write_fixture_to_disk();
    let file = RsmfFile::open(tmp.path()).expect("open");

    // `weight` owns canonical=0 and packed=[2].
    file.tensor_view_variant("weight", 0)
        .expect("canonical variant of weight must load");
    file.tensor_view_variant("weight", 2)
        .expect("packed variant of weight must load");

    // `bias` owns canonical=1 and no packed variants.
    file.tensor_view_variant("bias", 1)
        .expect("canonical variant of bias must load");
}

#[test]
fn unknown_tensor_name_is_rejected() {
    let tmp = write_fixture_to_disk();
    let file = RsmfFile::open(tmp.path()).expect("open");

    let err = file
        .tensor_view_variant("nonexistent", 0)
        .expect_err("must reject unknown tensor");
    assert!(matches!(err, RsmfError::NotFound { .. }));
}
