//! End-to-end example for the experimental multi-file sharding path.
//!
//! The master file is a normal RSMF artifact. Tensors tagged with
//! `shard_id = N` redirect reads to an attached shard file instead of
//! the master's canonical arena. Shard files are **raw arena byte
//! buffers** — they have no preamble, section table, or manifest of
//! their own. A variant at `section_relative_offset = K` with
//! `length = L` lives at `shard_bytes[K..K + L]`.
//!
//! v1 of RSMF does not yet provide a writer-side API for producing
//! matching master + shard pairs, so this example constructs both
//! sides by hand: the master writes placeholder bytes at the
//! canonical arena offset so it round-trips through structural
//! validation, and the shard file is written separately with the real
//! tensor data.

use memmap2::Mmap;
use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile};
use std::fs;

fn main() -> anyhow::Result<()> {
    fs::create_dir_all("tmp")?;

    // Real tensor bytes — 100 f32 values = 400 bytes. These live in
    // the shard, not the master.
    let shard_values: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let shard_bytes: Vec<u8> = shard_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(shard_bytes.len(), 400);

    // 1. Master: tensor descriptor with shard_id=1. The canonical
    //    variant still carries 400 bytes (any 400 bytes) so the
    //    master round-trips through structural validation, but those
    //    bytes will never be read once the shard is attached.
    let placeholder_master_bytes = vec![0u8; 400];
    let master_writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 1,
        name: "entity_weights".into(),
        dtype: LogicalDtype::F32,
        shape: vec![100],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(placeholder_master_bytes),
        packed: Vec::new(),
    });
    master_writer.write_to_path("tmp/master.rsmf")?;

    // 2. Shard: raw bytes at offset 0 because the single tensor's
    //    variant has section_relative_offset = 0. Writer-side
    //    sharding would compute and write this layout automatically.
    fs::write("tmp/shard1.bin", &shard_bytes)?;

    // 3. Read.
    let master = RsmfFile::open("tmp/master.rsmf")?;
    let shard_file = fs::File::open("tmp/shard1.bin")?;
    // SAFETY: the shard file is not mutated for the duration of the
    // mmap lifetime; this is the same assumption RSMF makes about the
    // master file it mmaps during `open`.
    let shard_mmap = unsafe { Mmap::map(&shard_file)? };
    let master = master.with_shard(1, shard_mmap);

    let view = master.tensor_view("entity_weights")?;
    let data = view.as_slice::<f32>()?;
    assert_eq!(data.len(), 100);
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], 0.5);
    assert_eq!(data[99], 49.5);

    println!(
        "Sharded read OK: first value {}, last value {} (shard bytes, not master placeholder)",
        data[0], data[99]
    );
    Ok(())
}
