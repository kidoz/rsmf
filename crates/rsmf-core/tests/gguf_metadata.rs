use std::io::Write;

use rsmf_core::writer::{RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{GgufMetaValue, LogicalDtype, RsmfFile};
use tempfile::NamedTempFile;

#[test]
fn reader_exposes_typed_gguf_metadata_without_prefix() {
    let writer = RsmfWriter::new()
        .with_metadata("gguf.general.architecture", "llama")
        .with_metadata("gguf.tokenizer.ggml.tokens", r#"["<s>","hello"]"#)
        .with_metadata("other.key", "ignored")
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "weight".into(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(1.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        });

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&writer.write_to_bytes().unwrap()).unwrap();
    tmp.flush().unwrap();

    let file = RsmfFile::open(tmp.path()).unwrap();
    let metadata = file.gguf_metadata();

    assert_eq!(metadata["general.architecture"].as_str(), Some("llama"));
    assert_eq!(
        metadata["tokenizer.ggml.tokens"].as_array(),
        Some(
            [
                GgufMetaValue::Str("<s>".to_string()),
                GgufMetaValue::Str("hello".to_string())
            ]
            .as_slice()
        )
    );
    assert!(!metadata.contains_key("other.key"));
}
