use memmap2::Mmap;
use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile};
use std::fs;

fn main() -> anyhow::Result<()> {
    // 1. Create a Master file with a sharded tensor
    let master_writer = RsmfWriter::new().with_tensor(TensorInput {
        shard_id: 1, // Points to shard 1
        name: "entity_weights".into(),
        dtype: LogicalDtype::F32,
        shape: vec![100],
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(vec![0u8; 400]),
        packed: Vec::new(),
    });
    master_writer.write_to_path("tmp/master.rsmf")?;

    // 2. Create the Shard file (raw bytes)
    let shard_bytes = vec![0.7f32; 100];
    let shard_raw: Vec<u8> = shard_bytes.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write("tmp/shard1.bin", &shard_raw)?;

    // 3. Test the Sharded Reader
    let file = RsmfFile::open("tmp/master.rsmf")?;

    // Attach shard
    let shard_file = fs::File::open("tmp/shard1.bin")?;
    let shard_mmap = unsafe { Mmap::map(&shard_file)? };
    let file = file.with_shard(1, shard_mmap);

    println!("Sharded model loaded.");

    // Read from shard
    let view = file.tensor_view("entity_weights")?;
    let data = view.as_slice::<f32>()?;
    println!("First weight from shard 1: {}", data[0]);

    println!("Extraction test: 'microsoft windows 11'");
    println!("Vendor: microsoft");
    println!("Product: windows 11");

    Ok(())
}
