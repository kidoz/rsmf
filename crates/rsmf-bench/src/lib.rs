//! Shared helpers for rsmf benchmarks (synthetic fixtures, etc.).
#![warn(missing_docs)]

use std::path::Path;

use anyhow::Result;
use rsmf_core::{EncodingKind, LayoutTag, LogicalDtype, StorageDtype, TargetTag};
use rsmf_core::{RsmfWriter, TensorInput, VariantInput};

/// Default bench tensor row count — small enough to run on a laptop in seconds.
pub const DEFAULT_ROWS: usize = 512;
/// Default bench tensor column count.
pub const DEFAULT_COLS: usize = 512;

/// Build a synthetic RSMF file with a single canonical `f32` tensor at `path`.
///
/// Used by multiple benches to avoid duplicating fixture setup.
pub fn build_fixture(path: &Path, rows: usize, cols: usize) -> Result<()> {
    let total = rows * cols;
    let mut bytes = Vec::with_capacity(total * 4);
    for i in 0..total {
        let v = (i as f32).sin();
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    let writer = RsmfWriter::new().with_tensor(TensorInput {
        name: "weight".into(),
        dtype: LogicalDtype::F32,
        shape: vec![rows as u64, cols as u64],
        shard_id: 0,
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(bytes),
        packed: vec![],
    });

    writer.write_to_path(path)?;
    Ok(())
}

/// Convenience alias for the bench fixture layout tag (row-major).
pub const LAYOUT_ROW_MAJOR: LayoutTag = LayoutTag::RowMajor;

/// Convenience alias for the canonical target tag.
pub const CANONICAL_TARGET: TargetTag = TargetTag::Canonical;

/// Convenience alias for the canonical encoding kind.
pub const CANONICAL_ENCODING: EncodingKind = EncodingKind::Raw;

/// Convenience alias for the canonical storage dtype.
pub const CANONICAL_STORAGE: StorageDtype = StorageDtype::Logical(LogicalDtype::F32);
