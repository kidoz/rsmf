//! Shared fixtures for integration tests. No dependencies on the network or
//! external files.

use rsmf_core::writer::{AssetInput, GraphInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{EncodingKind, LayoutTag, LogicalDtype, TargetTag, VariantMeta};

/// Build a small synthetic RSMF in-memory with two f32 tensors, an ONNX-tagged
/// graph blob, one asset, and one packed wgpu variant.
#[must_use]
#[allow(dead_code)]
pub fn build_basic_file_bytes() -> Vec<u8> {
    let weight: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let bias: Vec<u8> = (0..4u32)
        .flat_map(|i| ((i + 100) as f32).to_le_bytes())
        .collect();

    // A packed wgpu variant that just mirrors the canonical bytes. Enough for
    // selection tests without needing real f16 data.
    let wgpu_weight: Vec<u8> = weight.clone();

    let writer = RsmfWriter::new()
        .with_metadata("framework", "rsmf")
        .with_metadata("arch", "test")
        .with_tensor(TensorInput {
            shard_id: 0,
            name: "weight".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4, 4],
            metadata: vec![("layer".into(), "0".into())],
            canonical: VariantInput::canonical_raw(weight),
            packed: vec![VariantInput {
                target: TargetTag::Wgpu,
                encoding: EncodingKind::Raw,
                storage_dtype: Some(rsmf_core::StorageDtype::Logical(LogicalDtype::F32)),
                layout: LayoutTag::RowMajor,
                alignment: 64,
                bytes: wgpu_weight,
                meta: VariantMeta::default(),
            }],
        })
        .with_tensor(TensorInput {
            shard_id: 0,

            name: "bias".into(),
            dtype: LogicalDtype::F32,
            shape: vec![4],

            metadata: vec![],
            canonical: VariantInput::canonical_raw(bias),
            packed: vec![],
        })
        .with_graph(GraphInput::onnx(b"fake-onnx-bytes".to_vec()))
        .with_asset(AssetInput::new("tokenizer.json", b"{\"t\":1}".to_vec()));

    writer.write_to_bytes().expect("write bytes")
}
