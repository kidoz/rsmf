use rsmf_core::LogicalDtype;
use rsmf_core::writer::{AssetInput, RsmfWriter, TensorInput, VariantInput};

fn main() -> anyhow::Result<()> {
    // 1. Create fake weights for "microsoft" and "windows"
    let microsoft_vec = vec![1.0f32, 0.0f32, 0.0f32];
    let windows_vec = vec![0.0f32, 1.0f32, 0.0f32];
    let eleven_vec = vec![0.0f32, 0.0f32, 1.0f32];

    let mut bytes = Vec::new();
    for v in &[microsoft_vec, windows_vec, eleven_vec] {
        for f in v {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
    }

    let writer = RsmfWriter::new()
        .with_metadata("model_type", "entity_extractor")
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "vocab_embeddings".into(),
            dtype: LogicalDtype::F32,
            shape: vec![3, 3],
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(bytes),
            packed: Vec::new(),
        })
        .with_asset(AssetInput::new(
            "labels.json",
            b"{\"0\": \"vendor:microsoft\", \"1\": \"product:windows\", \"2\": \"version:11\"}"
                .to_vec(),
        ));

    writer.write_to_path("tmp/extractor.rsmf")?;
    Ok(())
}
