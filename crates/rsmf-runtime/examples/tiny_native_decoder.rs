use rsmf_core::writer::{AssetInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile};
use rsmf_runtime::{
    Engine, NATIVE_DECODER_CONFIG_ASSET, NATIVE_DECODER_TOKENIZER_ASSET, NativeDecoderRunOptions,
};
use tempfile::tempdir;

fn main() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("tiny-native-decoder.rsmf");

    let mut writer = RsmfWriter::new()
        .with_metadata("model.arch", "llama")
        .with_asset(AssetInput::new(
            NATIVE_DECODER_CONFIG_ASSET,
            tiny_config_json().into_bytes(),
        ))
        .with_asset(AssetInput::new(
            NATIVE_DECODER_TOKENIZER_ASSET,
            tiny_tokenizer_json().into_bytes(),
        ));

    for (name, shape) in tiny_tensor_specs() {
        writer = writer.with_tensor(tensor_with_values(name, shape, tensor_values(name, shape)));
    }
    writer.write_to_path(&path)?;

    let engine = Engine::new(RsmfFile::open(&path)?)?;
    let session = engine.native_decoder_session()?;
    let output = session.generate_text(
        "zero",
        NativeDecoderRunOptions {
            max_new_tokens: 3,
            ..NativeDecoderRunOptions::default()
        },
    )?;

    println!("wrote {}", path.display());
    println!("prompt:     {}", output.prompt);
    println!("generated:  {}", output.generated_text);
    println!("full text:  {}", output.text);
    println!("token ids:  {:?}", output.token_ids);
    println!("backend:    {:?}", output.backend);

    Ok(())
}

fn tiny_config_json() -> String {
    serde_json::json!({
        "model_type": "llama",
        "hidden_size": 2,
        "intermediate_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 4,
        "max_position_embeddings": 8,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": false
    })
    .to_string()
}

fn tiny_tokenizer_json() -> String {
    serde_json::json!({
        "model": {
            "type": "WordLevel",
            "vocab": {
                "zero": 0,
                "one": 1,
                "two": 2,
                "minus": 3
            }
        }
    })
    .to_string()
}

fn tiny_tensor_specs() -> &'static [(&'static str, &'static [u64])] {
    &[
        ("model.embed_tokens.weight", &[4, 2]),
        ("model.norm.weight", &[2]),
        ("lm_head.weight", &[4, 2]),
        ("model.layers.0.input_layernorm.weight", &[2]),
        ("model.layers.0.post_attention_layernorm.weight", &[2]),
        ("model.layers.0.self_attn.q_proj.weight", &[2, 2]),
        ("model.layers.0.self_attn.k_proj.weight", &[2, 2]),
        ("model.layers.0.self_attn.v_proj.weight", &[2, 2]),
        ("model.layers.0.self_attn.o_proj.weight", &[2, 2]),
        ("model.layers.0.mlp.gate_proj.weight", &[2, 2]),
        ("model.layers.0.mlp.up_proj.weight", &[2, 2]),
        ("model.layers.0.mlp.down_proj.weight", &[2, 2]),
    ]
}

fn tensor_values(name: &str, shape: &[u64]) -> Vec<f32> {
    let elements = shape.iter().product::<u64>() as usize;
    let mut values = vec![0.0; elements];
    match name {
        "model.embed_tokens.weight" => {
            values.copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        }
        "model.norm.weight"
        | "model.layers.0.input_layernorm.weight"
        | "model.layers.0.post_attention_layernorm.weight" => {
            values.fill(1.0);
        }
        "lm_head.weight" => {
            values.copy_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, -1.0]);
        }
        _ => {}
    }
    values
}

fn tensor_with_values(name: &str, shape: &[u64], values: Vec<f32>) -> TensorInput {
    assert_eq!(values.len(), shape.iter().product::<u64>() as usize);
    TensorInput {
        name: name.to_string(),
        dtype: LogicalDtype::F32,
        shape: shape.to_vec(),
        shard_id: 0,
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(f32_bytes(&values)),
        packed: Vec::new(),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
