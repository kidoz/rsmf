use std::io::Write;

use rsmf_core::{LogicalDtype, RsmfFile, RsmfWriter, TensorInput, VariantInput};
use tempfile::NamedTempFile;

#[test]
fn with_metadata_replaced_removes_existing_key() {
    let writer = RsmfWriter::new()
        .with_metadata("source", "safetensors")
        .with_metadata("framework", "test")
        .with_metadata_replaced("source", "torch")
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "x".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(1.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        });

    let bytes = writer.write_to_bytes().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let metadata = &file.inspect().metadata;
    let sources: Vec<_> = metadata.iter().filter(|(key, _)| key == "source").collect();

    assert_eq!(sources, vec![&("source".to_string(), "torch".to_string())]);
    assert!(
        metadata
            .iter()
            .any(|(key, value)| key == "framework" && value == "test")
    );
}
