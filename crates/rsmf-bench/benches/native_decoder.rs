use criterion::{Criterion, criterion_group, criterion_main};
use rsmf_core::writer::{AssetInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{LogicalDtype, RsmfFile};
use rsmf_runtime::{
    Engine, NATIVE_DECODER_CONFIG_ASSET, NATIVE_DECODER_TOKENIZER_ASSET,
    NativeDecoderAttentionImplementation, NativeDecoderBackend,
    NativeDecoderContinuousBatchRequest, NativeDecoderPerformanceOptions, NativeDecoderRunOptions,
};
use tempfile::tempdir;

fn bench_native_decoder(c: &mut Criterion) {
    let dir = tempdir().expect("create native decoder bench tempdir");
    let path = dir.path().join("native-decoder-bench.rsmf");
    build_tiny_native_decoder(&path);
    let engine = Engine::new(RsmfFile::open(path).expect("open native decoder fixture"))
        .expect("create runtime engine");
    let resident_session = engine
        .native_decoder_session()
        .expect("create resident native decoder session");

    let prompt_one = vec![0i64];
    c.bench_function("native_decoder/token_generation_cpu_reference", |b| {
        b.iter(|| {
            engine
                .native_decoder_greedy_decode(
                    std::hint::black_box(&prompt_one),
                    NativeDecoderRunOptions {
                        max_new_tokens: 2,
                        ..NativeDecoderRunOptions::default()
                    },
                )
                .expect("generate native decoder tokens")
        });
    });

    c.bench_function("native_decoder/session_build_weight_residency", |b| {
        b.iter(|| {
            engine
                .native_decoder_session()
                .expect("create resident native decoder session")
        });
    });

    c.bench_function(
        "native_decoder/resident_token_generation_cpu_reference",
        |b| {
            b.iter(|| {
                resident_session
                    .generate_token_ids(
                        std::hint::black_box(&prompt_one),
                        NativeDecoderRunOptions {
                            max_new_tokens: 2,
                            ..NativeDecoderRunOptions::default()
                        },
                    )
                    .expect("generate resident native decoder tokens")
            });
        },
    );

    c.bench_function(
        "native_decoder/resident_text_generation_cpu_reference",
        |b| {
            b.iter(|| {
                resident_session
                    .generate_text(
                        std::hint::black_box("zero"),
                        NativeDecoderRunOptions {
                            max_new_tokens: 2,
                            ..NativeDecoderRunOptions::default()
                        },
                    )
                    .expect("generate resident native decoder text")
            });
        },
    );

    c.bench_function("native_decoder/threaded_logits", |b| {
        b.iter(|| {
            engine
                .native_decoder_greedy_decode(
                    std::hint::black_box(&prompt_one),
                    NativeDecoderRunOptions {
                        max_new_tokens: 2,
                        backend: NativeDecoderBackend::CpuThreaded,
                        performance: NativeDecoderPerformanceOptions {
                            cpu_threads: Some(2),
                            ..NativeDecoderPerformanceOptions::default()
                        },
                        ..NativeDecoderRunOptions::default()
                    },
                )
                .expect("generate native decoder tokens with threaded logits")
        });
    });

    c.bench_function("native_decoder/resident_threaded_logits", |b| {
        b.iter(|| {
            resident_session
                .generate_token_ids(
                    std::hint::black_box(&prompt_one),
                    NativeDecoderRunOptions {
                        max_new_tokens: 2,
                        backend: NativeDecoderBackend::CpuThreaded,
                        performance: NativeDecoderPerformanceOptions {
                            cpu_threads: Some(2),
                            ..NativeDecoderPerformanceOptions::default()
                        },
                        ..NativeDecoderRunOptions::default()
                    },
                )
                .expect("generate resident native decoder tokens with threaded logits")
        });
    });

    c.bench_function("native_decoder/paged_kv_page_1", |b| {
        b.iter(|| {
            engine
                .native_decoder_greedy_decode(
                    std::hint::black_box(&prompt_one),
                    NativeDecoderRunOptions {
                        max_new_tokens: 2,
                        performance: NativeDecoderPerformanceOptions {
                            kv_cache_page_size_tokens: Some(1),
                            ..NativeDecoderPerformanceOptions::default()
                        },
                        ..NativeDecoderRunOptions::default()
                    },
                )
                .expect("generate native decoder tokens with paged kv")
        });
    });

    let prompt_three = vec![0i64, 1, 0];
    c.bench_function("native_decoder/prompt_len_3_chunked_prefill", |b| {
        b.iter(|| {
            engine
                .native_decoder_greedy_decode(
                    std::hint::black_box(&prompt_three),
                    NativeDecoderRunOptions {
                        max_new_tokens: 1,
                        performance: NativeDecoderPerformanceOptions {
                            prefill_chunk_size: Some(2),
                            ..NativeDecoderPerformanceOptions::default()
                        },
                        ..NativeDecoderRunOptions::default()
                    },
                )
                .expect("generate native decoder tokens with chunked prefill")
        });
    });

    c.bench_function(
        "native_decoder/resident_prompt_len_3_chunked_prefill",
        |b| {
            b.iter(|| {
                resident_session
                    .generate_token_ids(
                        std::hint::black_box(&prompt_three),
                        NativeDecoderRunOptions {
                            max_new_tokens: 1,
                            performance: NativeDecoderPerformanceOptions {
                                prefill_chunk_size: Some(2),
                                ..NativeDecoderPerformanceOptions::default()
                            },
                            ..NativeDecoderRunOptions::default()
                        },
                    )
                    .expect("generate resident native decoder tokens with chunked prefill")
            });
        },
    );

    let prefix_options = NativeDecoderRunOptions {
        max_new_tokens: 1,
        performance: NativeDecoderPerformanceOptions {
            prefix_cache_max_entries: Some(8),
            ..NativeDecoderPerformanceOptions::default()
        },
        ..NativeDecoderRunOptions::default()
    };
    resident_session
        .generate_token_ids(&prompt_three, prefix_options.clone())
        .expect("warm resident native decoder prefix cache");
    c.bench_function("native_decoder/resident_prefix_cache_hit", |b| {
        b.iter(|| {
            resident_session
                .generate_token_ids(std::hint::black_box(&prompt_three), prefix_options.clone())
                .expect("generate resident native decoder tokens with prefix cache")
        });
    });

    c.bench_function("native_decoder/cpu_tiled_attention", |b| {
        b.iter(|| {
            resident_session
                .generate_token_ids(
                    std::hint::black_box(&prompt_three),
                    NativeDecoderRunOptions {
                        max_new_tokens: 1,
                        performance: NativeDecoderPerformanceOptions {
                            attention: NativeDecoderAttentionImplementation::CpuTiled,
                            kv_cache_page_size_tokens: Some(1),
                            ..NativeDecoderPerformanceOptions::default()
                        },
                        ..NativeDecoderRunOptions::default()
                    },
                )
                .expect("generate resident native decoder tokens with tiled attention")
        });
    });

    c.bench_function("native_decoder/continuous_batch_two_requests", |b| {
        b.iter(|| {
            resident_session
                .generate_token_ids_continuous_batch(vec![
                    NativeDecoderContinuousBatchRequest {
                        request_id: "0".to_string(),
                        input_token_ids: std::hint::black_box(prompt_one.clone()),
                        options: NativeDecoderRunOptions {
                            max_new_tokens: 2,
                            performance: NativeDecoderPerformanceOptions {
                                continuous_batch_max_requests: Some(2),
                                ..NativeDecoderPerformanceOptions::default()
                            },
                            ..NativeDecoderRunOptions::default()
                        },
                        deadline: None,
                        cancelled: false,
                    },
                    NativeDecoderContinuousBatchRequest {
                        request_id: "1".to_string(),
                        input_token_ids: std::hint::black_box(prompt_one.clone()),
                        options: NativeDecoderRunOptions {
                            max_new_tokens: 2,
                            performance: NativeDecoderPerformanceOptions {
                                continuous_batch_max_requests: Some(2),
                                ..NativeDecoderPerformanceOptions::default()
                            },
                            ..NativeDecoderRunOptions::default()
                        },
                        deadline: None,
                        cancelled: false,
                    },
                ])
                .expect("generate continuous native decoder batch")
        });
    });
}

fn build_tiny_native_decoder(path: &std::path::Path) {
    let mut writer = RsmfWriter::new()
        .with_metadata("model.arch", "llama")
        .with_asset(AssetInput::new(
            NATIVE_DECODER_CONFIG_ASSET,
            tiny_native_decoder_config_json().into_bytes(),
        ))
        .with_asset(AssetInput::new(
            NATIVE_DECODER_TOKENIZER_ASSET,
            tiny_native_decoder_tokenizer_json().into_bytes(),
        ));
    for (name, shape) in tiny_native_decoder_tensor_specs() {
        writer = writer.with_tensor(native_decoder_tensor_with_values(
            name,
            shape,
            tiny_native_decoder_tensor_values(name, shape),
        ));
    }
    writer
        .write_to_path(path)
        .expect("write native decoder fixture");
}

fn tiny_native_decoder_config_json() -> String {
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

fn tiny_native_decoder_tokenizer_json() -> String {
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

fn tiny_native_decoder_tensor_specs() -> Vec<(&'static str, &'static [u64])> {
    vec![
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

fn tiny_native_decoder_tensor_values(name: &str, shape: &[u64]) -> Vec<f32> {
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

fn native_decoder_tensor_with_values(name: &str, shape: &[u64], values: Vec<f32>) -> TensorInput {
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

criterion_group!(benches, bench_native_decoder);
criterion_main!(benches);
