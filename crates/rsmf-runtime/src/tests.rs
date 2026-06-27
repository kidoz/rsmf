use super::native_decoder::*;
use super::*;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

use rsmf_core::writer::{AssetInput, RsmfWriter, TensorInput, VariantInput};
use rsmf_core::{
    EncodingKind, GraphInput, LayoutTag, LogicalDtype, StorageDtype, TargetTag, VariantMeta,
};
use tempfile::tempdir;

#[derive(Debug, Deserialize)]
struct TinyHfNativeDecoderReference {
    prompt: String,
    prompt_token_ids: Vec<i64>,
    max_new_tokens: usize,
    expected_generated_token_ids: Vec<i64>,
    expected_text: String,
    expected_logits: Vec<Vec<f32>>,
    tolerance_abs: f32,
}

#[test]
fn native_decoder_config_parses_llama_defaults() {
    let config = NativeDecoderConfig::from_hf_config_json(
        br#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 6,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "vocab_size": 8,
                "max_position_embeddings": 16,
                "eos_token_id": [2, 3],
                "tie_word_embeddings": true
            }"#,
    )
    .unwrap();

    assert_eq!(config.family, NativeDecoderFamily::Llama);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.rms_norm_eps, 1e-6);
    assert_eq!(config.rope_theta, 10_000.0);
    assert_eq!(config.eos_token_ids, vec![2, 3]);
    assert!(config.tie_word_embeddings);
}

#[test]
fn native_decoder_config_rejects_unsupported_family() {
    let err = NativeDecoderConfig::from_hf_config_json(
        br#"{
                "model_type": "gpt_neox",
                "hidden_size": 4,
                "intermediate_size": 6,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "vocab_size": 8,
                "max_position_embeddings": 16
            }"#,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        RuntimeError::UnsupportedNativeDecoder { family } if family == "gpt_neox"
    ));
}

#[test]
fn engine_native_decoder_contract_validates_tiny_llama() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_engine(
        dir.path().join("native-decoder.rsmf"),
        NativeDecoderFixtureOptions::default(),
    );

    let contract = engine.native_decoder_contract().unwrap();
    assert_eq!(contract.config.family, NativeDecoderFamily::Llama);
    assert_eq!(contract.config.hidden_size, 4);
    assert_eq!(
        contract.assets.generation_config_asset.as_deref(),
        Some(NATIVE_DECODER_GENERATION_CONFIG_ASSET)
    );
    assert_eq!(contract.tensors.len(), 12);
    assert!(contract.tensors.iter().any(|tensor| {
        tensor.role == "layers.0.self_attn.k_proj"
            && tensor.tensor_name == "model.layers.0.self_attn.k_proj.weight"
            && tensor.shape == vec![2, 4]
            && tensor.dtype == "F32"
    }));
}

#[test]
fn engine_native_decoder_contract_requires_tokenizer_asset() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_engine(
        dir.path().join("native-decoder-missing-tokenizer.rsmf"),
        NativeDecoderFixtureOptions {
            include_tokenizer: false,
            ..NativeDecoderFixtureOptions::default()
        },
    );

    let err = engine.native_decoder_contract().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::NativeDecoderAssetMissing { asset_name }
            if asset_name == NATIVE_DECODER_TOKENIZER_ASSET
    ));
}

#[test]
fn engine_native_decoder_contract_requires_declared_tensors() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_engine(
        dir.path().join("native-decoder-missing-tensor.rsmf"),
        NativeDecoderFixtureOptions {
            omit_tensor: Some("model.layers.0.mlp.down_proj.weight".to_string()),
            ..NativeDecoderFixtureOptions::default()
        },
    );

    let err = engine.native_decoder_contract().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::NativeDecoderTensorMissing { tensor_name }
            if tensor_name == "model.layers.0.mlp.down_proj.weight"
    ));
}

#[test]
fn engine_native_decoder_contract_rejects_shape_mismatch() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_engine(
        dir.path().join("native-decoder-bad-shape.rsmf"),
        NativeDecoderFixtureOptions {
            bad_shape: Some(("model.embed_tokens.weight".to_string(), vec![7, 4])),
            ..NativeDecoderFixtureOptions::default()
        },
    );

    let err = engine.native_decoder_contract().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::NativeDecoderTensorShapeMismatch {
            tensor_name,
            expected_shape,
            actual_shape,
        } if tensor_name == "model.embed_tokens.weight"
            && expected_shape == "[8,4]"
            && actual_shape == "[7,4]"
    ));
}

#[test]
fn engine_native_decoder_weights_loads_f32_tensors_from_rsmf() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let weights = engine.native_decoder_weights().unwrap();
    assert_eq!(weights.config.hidden_size, 2);
    assert_eq!(weights.layers.len(), 1);
    assert_eq!(weights.token_embedding.len(), 8);
    assert_eq!(weights.final_norm, vec![1.0, 1.0]);
    assert!(weights.lm_head.is_some());
}

#[test]
fn engine_native_decoder_tokenizer_encodes_and_decodes_wordlevel_text() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let tokenizer = engine.native_decoder_tokenizer().unwrap();
    assert_eq!(tokenizer.model_type, "WordLevel");
    assert_eq!(tokenizer.encode("zero one").unwrap(), vec![0, 1]);
    assert_eq!(tokenizer.decode(&[0, 1, 2]).unwrap(), "zero one two");
}

#[test]
fn engine_native_decoder_tokenizer_rejects_unsupported_model_type() {
    let err = NativeDecoderTokenizer::from_json(br#"{"model": {"type": "Unigram"}}"#).unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderTokenizerInvalid { reason } if reason.contains("only WordLevel and BPE"))
    );
}

#[test]
fn engine_native_decoder_tokenizer_rejects_unknown_token_without_unk() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let tokenizer = engine.native_decoder_tokenizer().unwrap();

    let err = tokenizer.encode("missing").unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderTokenizerTokenUnknown { token } if token == "missing")
    );
}

#[test]
fn native_decoder_bpe_tokenizer_encodes_merges_and_bytelevel_space() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "h": 0,
                    "e": 1,
                    "l": 2,
                    "o": 3,
                    "hello": 4,
                    "Ġworld": 5,
                    "<unk>": 6
                },
                "merges": ["h e", "he l", "hel l", "hell o"],
                "unk_token": "<unk>"
            },
            "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("hello world").unwrap(), vec![4, 5]);
    assert_eq!(tokenizer.decode(&[4, 5]).unwrap(), "hello world");
}

#[test]
fn native_decoder_bpe_tokenizer_accepts_added_special_tokens() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": { "hello": 0, "<unk>": 1 },
                "merges": [],
                "unk_token": "<unk>"
            },
            "added_tokens": [
                { "id": 2, "content": "<eos>", "special": true }
            ]
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("<eos>").unwrap(), vec![2]);
    assert_eq!(tokenizer.decode(&[2]).unwrap(), "<eos>");
}

#[test]
fn native_decoder_tokenizer_rejects_unsupported_normalizer() {
    let err = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "normalizer": { "type": "Lowercase" },
            "model": {
                "type": "BPE",
                "vocab": { "hello": 0 },
                "merges": []
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderTokenizerInvalid { reason } if reason.contains("normalizer"))
    );
}

#[test]
fn engine_native_decoder_greedy_decode_generates_expected_tokens() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let output = engine
        .native_decoder_greedy_decode(
            &[0],
            NativeDecoderRunOptions {
                max_new_tokens: 3,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(output.backend, NativeDecoderBackend::CpuReference);
    assert_eq!(output.generated_token_ids, vec![1, 2]);
    assert_eq!(output.token_ids, vec![0, 1, 2]);
    assert_eq!(output.logits.len(), 2);
    assert!(output.logits[0][1] > output.logits[0][0]);
    assert!(output.logits[1][2] > output.logits[1][1]);
}

#[test]
fn engine_native_decoder_generate_text_decodes_generated_tokens() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let output = engine
        .native_decoder_generate_text(
            "zero",
            NativeDecoderRunOptions {
                max_new_tokens: 3,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(output.prompt, "zero");
    assert_eq!(output.generated_token_ids, vec![1, 2]);
    assert_eq!(output.generated_text, "one two");
    assert_eq!(output.text, "zero one two");
    assert_eq!(output.backend, NativeDecoderBackend::CpuReference);
}

#[test]
fn engine_native_decoder_session_reuses_resident_weights() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let session = engine.native_decoder_session().unwrap();

    let tokens = session
        .generate_token_ids(
            &[0],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();
    let text = session
        .generate_text(
            "zero",
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(tokens.generated_token_ids, vec![1, 2]);
    assert_eq!(text.generated_token_ids, vec![1, 2]);
    assert_eq!(text.text, "zero one two");
}

#[test]
fn engine_native_decoder_return_prompt_logits_reports_prefill_rows() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let output = engine
        .native_decoder_greedy_decode(
            &[0, 1],
            NativeDecoderRunOptions {
                max_new_tokens: 1,
                return_prompt_logits: true,
                performance: NativeDecoderPerformanceOptions {
                    prefill_chunk_size: Some(1),
                    ..NativeDecoderPerformanceOptions::default()
                },
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(output.prompt_logits.len(), 2);
    assert_eq!(output.prompt_logits[0].len(), 4);
    assert_eq!(output.generated_token_ids, vec![2]);
}

#[test]
fn engine_native_decoder_stop_token_override_stops_before_config_eos() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let output = engine
        .native_decoder_greedy_decode(
            &[0],
            NativeDecoderRunOptions {
                max_new_tokens: 3,
                stop_token_ids: vec![1],
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(output.generated_token_ids, vec![1]);
}

#[test]
fn engine_native_decoder_min_new_tokens_delays_stop_token() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let output = engine
        .native_decoder_greedy_decode(
            &[1],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                min_new_tokens: 2,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(output.generated_token_ids.len(), 2);
    assert_eq!(output.generated_token_ids[0], 2);
}

#[test]
fn native_decoder_repetition_penalty_can_change_argmax() {
    let adjusted = apply_native_decoder_repetition_penalty(
        &[10.0, 9.0],
        &[0],
        &NativeDecoderSamplingOptions {
            repetition_penalty: Some(2.0),
            ..NativeDecoderSamplingOptions::default()
        },
    )
    .unwrap();
    let token = select_native_decoder_token(
        &adjusted,
        &NativeDecoderSamplingOptions::default(),
        &mut NativeDecoderSamplerRng::new(1),
    )
    .unwrap();

    assert_eq!(token, 1);
}

#[test]
fn native_decoder_sampling_rejects_invalid_repetition_penalty() {
    let err = validate_native_decoder_sampling_options(&NativeDecoderSamplingOptions {
        repetition_penalty: Some(0.5),
        ..NativeDecoderSamplingOptions::default()
    })
    .unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderSamplingInvalid { reason } if reason.contains("repetition_penalty"))
    );
}

#[test]
fn engine_native_decoder_rejects_min_new_tokens_above_max() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let err = engine
        .native_decoder_greedy_decode(
            &[0],
            NativeDecoderRunOptions {
                max_new_tokens: 1,
                min_new_tokens: 2,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderConfigInvalid { reason } if reason.contains("min_new_tokens"))
    );
}

#[test]
fn engine_native_decoder_sampling_top_k_one_matches_greedy() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let output = engine
        .native_decoder_greedy_decode(
            &[0],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                sampling: NativeDecoderSamplingOptions {
                    temperature: Some(1.0),
                    top_k: Some(1),
                    top_p: Some(1.0),
                    seed: Some(42),
                    ..NativeDecoderSamplingOptions::default()
                },
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(output.generated_token_ids, vec![1, 2]);
}

#[test]
fn engine_native_decoder_sampling_rejects_invalid_temperature() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let err = engine
        .native_decoder_greedy_decode(
            &[0],
            NativeDecoderRunOptions {
                sampling: NativeDecoderSamplingOptions {
                    temperature: Some(0.0),
                    ..NativeDecoderSamplingOptions::default()
                },
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderSamplingInvalid { reason } if reason.contains("temperature"))
    );
}

#[test]
fn engine_native_decoder_reference_logits_match_tiny_fixture() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let norm = 1.4142121f32;

    let report = engine
        .native_decoder_check_reference_logits(NativeDecoderReferenceLogitCheck {
            input_token_ids: vec![0, 1],
            expected_logits: vec![vec![0.0, norm, 0.0, -norm], vec![0.0, 0.0, norm, -norm]],
            tolerance_abs: 1e-5,
            backend: NativeDecoderBackend::CpuReference,
            weight_options: NativeDecoderWeightOptions::default(),
            performance: NativeDecoderPerformanceOptions::default(),
        })
        .unwrap();

    assert_eq!(report.compared_logits, 2);
    assert_eq!(report.compared_values, 8);
    assert!(report.max_abs_diff <= 1e-5);
}

#[test]
fn engine_native_decoder_matches_local_hf_reference_fixture() {
    let fixture: TinyHfNativeDecoderReference = serde_json::from_str(include_str!(
        "../tests/fixtures/tiny_hf_native_decoder_reference.json"
    ))
    .unwrap();
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let tokenizer = engine.native_decoder_tokenizer().unwrap();

    assert_eq!(
        tokenizer.encode(&fixture.prompt).unwrap(),
        fixture.prompt_token_ids
    );
    let report = engine
        .native_decoder_check_reference_logits(NativeDecoderReferenceLogitCheck {
            input_token_ids: fixture
                .prompt_token_ids
                .iter()
                .chain(fixture.expected_generated_token_ids.iter().take(1))
                .copied()
                .collect(),
            expected_logits: fixture.expected_logits.clone(),
            tolerance_abs: fixture.tolerance_abs,
            backend: NativeDecoderBackend::CpuReference,
            weight_options: NativeDecoderWeightOptions::default(),
            performance: NativeDecoderPerformanceOptions::default(),
        })
        .unwrap();
    assert_eq!(report.compared_logits, fixture.expected_logits.len());

    let output = engine
        .native_decoder_generate_text(
            &fixture.prompt,
            NativeDecoderRunOptions {
                max_new_tokens: fixture.max_new_tokens,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();
    assert_eq!(
        output.generated_token_ids,
        fixture.expected_generated_token_ids
    );
    assert_eq!(output.text, fixture.expected_text);
}

#[test]
fn engine_native_decoder_reference_logits_reports_mismatch() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let err = engine
        .native_decoder_check_reference_logits(NativeDecoderReferenceLogitCheck {
            input_token_ids: vec![0],
            expected_logits: vec![vec![0.0, 0.0, 0.0, 0.0]],
            tolerance_abs: 1e-6,
            backend: NativeDecoderBackend::CpuReference,
            weight_options: NativeDecoderWeightOptions::default(),
            performance: NativeDecoderPerformanceOptions::default(),
        })
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderReferenceLogitsMismatch { max_abs_diff, .. } if max_abs_diff > 1.0)
    );
}

#[test]
fn engine_native_decoder_greedy_decode_rejects_empty_prompt() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let err = engine
        .native_decoder_greedy_decode(&[], NativeDecoderRunOptions::default())
        .unwrap_err();

    assert!(matches!(err, RuntimeError::NativeDecoderPromptEmpty));
}

#[test]
fn engine_native_decoder_accelerated_dispatch_uses_best_available_backend() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    let output = engine
        .native_decoder_greedy_decode(
            &[0],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                backend: NativeDecoderBackend::Accelerated,
                performance: NativeDecoderPerformanceOptions {
                    cpu_threads: Some(2),
                    ..NativeDecoderPerformanceOptions::default()
                },
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    #[cfg(all(target_os = "macos", feature = "apple-accelerate"))]
    assert_eq!(output.backend, NativeDecoderBackend::AppleCpuAccelerate);
    #[cfg(not(all(target_os = "macos", feature = "apple-accelerate")))]
    assert_eq!(output.backend, NativeDecoderBackend::CpuReference);
    assert_eq!(output.generated_token_ids, vec![1, 2]);
}

#[test]
fn engine_native_decoder_rejects_reserved_gpu_and_coreml_backends() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));

    for (backend, expected_name) in [
        (NativeDecoderBackend::MetalWgpuLmHead, "metal_wgpu_lm_head"),
        (
            NativeDecoderBackend::MetalWgpuFullDecoder,
            "metal_wgpu_full_decoder",
        ),
        (NativeDecoderBackend::OrtCoreMl, "ort_core_ml"),
    ] {
        let err = engine
            .native_decoder_greedy_decode(
                &[0],
                NativeDecoderRunOptions {
                    backend,
                    ..NativeDecoderRunOptions::default()
                },
            )
            .unwrap_err();

        assert!(
            matches!(err, RuntimeError::NativeDecoderBackendUnavailable { backend, .. } if backend == expected_name)
        );
    }
}

#[test]
fn native_decoder_cpu_rms_norm_matches_reference() {
    let output = native_decoder_cpu_rms_norm(&[3.0, 4.0], 1, 2, &[1.0, 0.5], 1e-6).unwrap();

    assert_close_slice(&output, &[0.8485281, 0.5656854], 1e-5);
}

#[test]
fn native_decoder_cpu_linear_uses_row_major_weight_rows() {
    let output = native_decoder_cpu_linear(&[1.0, 2.0], 1, 2, &[3.0, 4.0, 5.0, 6.0], 2).unwrap();

    assert_close_slice(&output, &[11.0, 17.0], 1e-6);
}

#[test]
fn native_decoder_backend_apple_accelerate_linear_matches_reference() {
    let input = [1.0, 2.0, -1.0, 0.5];
    let weight = [3.0, 4.0, 5.0, 6.0, -2.0, 1.0];
    let performance = NativeDecoderPerformanceOptions::default();
    let reference = native_decoder_cpu_linear(&input, 2, 2, &weight, 3).unwrap();
    let output = native_decoder_backend_linear(
        &input,
        2,
        2,
        &weight,
        3,
        NativeDecoderBackend::AppleCpuAccelerate,
        &performance,
    )
    .unwrap();

    assert_close_slice(&output, &reference, 1e-5);
}

#[test]
fn native_decoder_cpu_rope_rotates_even_odd_pairs() {
    let mut values = vec![1.0, 0.0];
    native_decoder_cpu_apply_llama_rope(&mut values, 1, 1, 2, 1, 10_000.0).unwrap();

    assert_close_slice(&values, &[1.0f32.cos(), 1.0f32.sin()], 1e-6);
}

#[test]
fn native_decoder_cpu_causal_attention_masks_future_tokens() {
    let output =
        native_decoder_cpu_causal_attention(&[1.0, 1.0], &[0.0, 0.0], &[2.0, 4.0], 2, 1, 1, 1)
            .unwrap();

    assert_close_slice(&output, &[2.0, 3.0], 1e-6);
}

#[test]
fn native_decoder_cpu_cached_attention_attends_over_cache() {
    let output =
        native_decoder_cpu_cached_attention(&[1.0], &[0.0, 0.0], &[2.0, 4.0], 2, 1, 1, 1).unwrap();

    assert_close_slice(&output, &[3.0], 1e-6);
}

#[test]
fn native_decoder_paged_kv_cache_tracks_allocated_pages() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let weights = engine.native_decoder_weights().unwrap();
    let mut cache = NativeDecoderKvCache::new_paged(&weights.config, 2).unwrap();
    let performance = NativeDecoderPerformanceOptions {
        kv_cache_page_size_tokens: Some(2),
        ..NativeDecoderPerformanceOptions::default()
    };

    native_decoder_cpu_step(
        &weights,
        &mut cache,
        0,
        NativeDecoderBackend::CpuReference,
        &performance,
    )
    .unwrap();
    assert_eq!(cache.position(), 1);
    assert_eq!(cache.allocated_pages(), 1);
    native_decoder_cpu_step(
        &weights,
        &mut cache,
        1,
        NativeDecoderBackend::CpuReference,
        &performance,
    )
    .unwrap();
    assert_eq!(cache.position(), 2);
    assert_eq!(cache.allocated_pages(), 1);
    native_decoder_cpu_step(
        &weights,
        &mut cache,
        2,
        NativeDecoderBackend::CpuReference,
        &performance,
    )
    .unwrap();
    assert_eq!(cache.position(), 3);
    assert_eq!(cache.allocated_pages(), 2);
}

#[test]
fn engine_native_decoder_paged_generation_matches_flat_cache() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let flat = engine
        .native_decoder_greedy_decode(
            &[0, 1, 0],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                stop_token_ids: vec![3],
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();
    let paged = engine
        .native_decoder_greedy_decode(
            &[0, 1, 0],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                stop_token_ids: vec![3],
                performance: NativeDecoderPerformanceOptions {
                    kv_cache_page_size_tokens: Some(1),
                    prefill_chunk_size: Some(2),
                    ..NativeDecoderPerformanceOptions::default()
                },
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(paged.generated_token_ids, flat.generated_token_ids);
    assert_eq!(paged.logits, flat.logits);
}

#[test]
fn native_decoder_threaded_linear_matches_reference() {
    let input = vec![1.0, 2.0];
    let weight = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let reference = native_decoder_cpu_linear(&input, 1, 2, &weight, 3).unwrap();
    let threaded = native_decoder_cpu_linear_threaded(&input, 1, 2, &weight, 3, 2).unwrap();

    assert_close_slice(&threaded, &reference, 1e-6);
}

#[test]
fn native_decoder_cpu_llama_block_zero_projections_preserve_hidden_states() {
    let config = tiny_native_decoder_cpu_config();
    let hidden_states = vec![1.0, 2.0, 3.0, 4.0];
    let zeros = vec![0.0; 4];
    let output = native_decoder_cpu_llama_block(
        &config,
        NativeDecoderCpuBlockInput {
            hidden_states: &hidden_states,
            sequence_len: 2,
            position_start: 0,
        },
        NativeDecoderCpuLayerWeights {
            input_layernorm: &[1.0, 1.0],
            post_attention_layernorm: &[1.0, 1.0],
            q_proj: &zeros,
            k_proj: &zeros,
            v_proj: &zeros,
            o_proj: &zeros,
            gate_proj: &zeros,
            up_proj: &zeros,
            down_proj: &zeros,
        },
    )
    .unwrap();

    assert_close_slice(&output.hidden_states, &hidden_states, 1e-6);
}

#[test]
fn native_decoder_cpu_llama_block_validates_weight_shape() {
    let config = tiny_native_decoder_cpu_config();
    let hidden_states = vec![1.0, 2.0];
    let zeros = vec![0.0; 4];
    let err = native_decoder_cpu_llama_block(
        &config,
        NativeDecoderCpuBlockInput {
            hidden_states: &hidden_states,
            sequence_len: 1,
            position_start: 0,
        },
        NativeDecoderCpuLayerWeights {
            input_layernorm: &[1.0, 1.0],
            post_attention_layernorm: &[1.0, 1.0],
            q_proj: &[0.0, 0.0, 0.0],
            k_proj: &zeros,
            v_proj: &zeros,
            o_proj: &zeros,
            gate_proj: &zeros,
            up_proj: &zeros,
            down_proj: &zeros,
        },
    )
    .unwrap_err();

    assert!(
        matches!(err, RuntimeError::Shape(message) if message.contains("linear: weight has 3 elements, expected 4"))
    );
}

#[test]
fn session_options_participate_in_cache_key() {
    let default_key = SessionKey::new(0, SessionOptions::default());
    let tuned_key = SessionKey::new(
        0,
        SessionOptions {
            intra_threads: Some(1),
            ..SessionOptions::default()
        },
    );
    assert_ne!(default_key, tuned_key);
}

#[test]
fn tensor_value_shape_mismatch_is_rejected() {
    let err = RuntimeTensor::F32 {
        shape: vec![2, 2],
        data: vec![1.0, 2.0, 3.0],
    }
    .into_ort_value()
    .unwrap_err();
    assert!(matches!(err, RuntimeError::Shape(message) if message.contains("implies 4 elements")));
}

#[test]
fn missing_graph_index_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("no-graph.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "x".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(1.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .write_to_path(&path)
        .unwrap();
    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();

    let err = engine
        .session_handle(0, SessionOptions::default())
        .unwrap_err();

    assert!(matches!(
        err,
        RuntimeError::GraphNotFound {
            graph_idx: 0,
            graph_count: 0
        }
    ));
}

#[test]
fn runs_embedded_onnx_add_graph_from_rsmf() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add.onnx.rsmf");
    let graph = tiny_add_onnx_model();
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(graph.clone()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let handle = engine.session_handle(0, SessionOptions::default()).unwrap();
    assert_eq!(handle.memory_report().graph_payload_bytes, graph.len());
    assert_eq!(handle.memory_report().initializer_count(), 0);
    assert_eq!(handle.memory_report().initializer_materialized_bytes, 0);
    let input_names = handle
        .inputs()
        .iter()
        .map(|input| input.name.as_str())
        .collect::<Vec<_>>();
    let output_names = handle
        .outputs()
        .iter()
        .map(|output| output.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(input_names, vec!["x", "y"]);
    assert_eq!(output_names, vec!["z"]);

    let outputs = engine
        .run_f32(
            0,
            HashMap::from([
                (
                    "x".to_string(),
                    ArrayD::from_shape_vec(vec![2], vec![1.5, -2.0]).unwrap(),
                ),
                (
                    "y".to_string(),
                    ArrayD::from_shape_vec(vec![2], vec![2.5, 3.0]).unwrap(),
                ),
            ]),
        )
        .unwrap();

    let z = outputs.get("z").unwrap();
    assert_eq!(z.shape(), &[2]);
    assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![4.0, 1.0]);
}

#[test]
fn runs_onnx_graph_with_rsmf_external_initializer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add-initializer.onnx.rsmf");
    let graph = tiny_add_external_initializer_onnx_model();
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(graph.clone()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let options = SessionOptions {
        initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
        ..SessionOptions::default()
    };
    let handle = engine.session_handle(0, options.clone()).unwrap();
    let memory_report = handle.memory_report().clone();
    assert_eq!(memory_report.graph_payload_bytes, graph.len());
    assert_eq!(memory_report.initializer_count(), 1);
    assert_eq!(memory_report.initializer_materialized_bytes, 8);
    assert_eq!(
        memory_report.initializers,
        vec![InitializerMemoryReport {
            initializer_name: "bias".to_string(),
            tensor_name: "bias.tensor".to_string(),
            variant_idx: None,
            materialized_bytes: 8,
        }]
    );
    let cached_handle = engine.session_handle(0, options.clone()).unwrap();
    assert_eq!(cached_handle.memory_report(), &memory_report);
    let input_names = handle
        .inputs()
        .iter()
        .map(|input| input.name.as_str())
        .collect::<Vec<_>>();
    let output_names = handle
        .outputs()
        .iter()
        .map(|output| output.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(input_names, vec!["x"]);
    assert_eq!(output_names, vec!["z"]);

    let outputs = engine
        .run_f32_with_options(
            0,
            options,
            HashMap::from([(
                "x".to_string(),
                ArrayD::from_shape_vec(vec![2], vec![1.5, 2.0]).unwrap(),
            )]),
        )
        .unwrap();

    let z = outputs.get("z").unwrap();
    assert_eq!(z.shape(), &[2]);
    assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![11.5, -2.0]);
}

#[test]
fn runs_onnx_graph_with_selected_raw_rsmf_initializer_variant() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add-selected-variant.onnx.rsmf");
    let graph = tiny_add_external_initializer_onnx_model();
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[0.0, 0.0])),
            packed: vec![raw_variant(
                StorageDtype::Logical(LogicalDtype::F32),
                LayoutTag::RowMajor,
                f32_bytes(&[10.0, -4.0]),
            )],
        })
        .with_graph(GraphInput::onnx(graph.clone()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let options = SessionOptions {
        initializers: vec![InitializerBinding::new("bias", "bias.tensor").with_variant(1)],
        ..SessionOptions::default()
    };
    let handle = engine.session_handle(0, options.clone()).unwrap();
    assert_eq!(handle.memory_report().initializer_materialized_bytes, 8);
    assert_eq!(
        handle.memory_report().initializers,
        vec![InitializerMemoryReport {
            initializer_name: "bias".to_string(),
            tensor_name: "bias.tensor".to_string(),
            variant_idx: Some(1),
            materialized_bytes: 8,
        }]
    );

    let outputs = engine
        .run_f32_with_options(
            0,
            options,
            HashMap::from([(
                "x".to_string(),
                ArrayD::from_shape_vec(vec![2], vec![1.5, 2.0]).unwrap(),
            )]),
        )
        .unwrap();

    let z = outputs.get("z").unwrap();
    assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![11.5, -2.0]);
}

#[test]
fn blocked_initializer_variant_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("blocked-initializer-variant.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[0.0, 0.0])),
            packed: vec![raw_variant(
                StorageDtype::Logical(LogicalDtype::F32),
                LayoutTag::Blocked,
                f32_bytes(&[10.0, -4.0]),
            )],
        })
        .with_graph(GraphInput::onnx(tiny_add_external_initializer_onnx_model()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let err = engine
        .session_handle(
            0,
            SessionOptions {
                initializers: vec![InitializerBinding::new("bias", "bias.tensor").with_variant(1)],
                ..SessionOptions::default()
            },
        )
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("row-major"))
    );
}

#[test]
fn runs_onnx_graph_with_i64_rsmf_external_initializer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("add-i64-initializer.onnx.rsmf");
    let graph = tiny_add_external_initializer_onnx_model_with_dtype_shape(7, &[2]);
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::I64,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(i64_bytes(&[10, -4])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(graph.clone()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let options = SessionOptions {
        initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
        ..SessionOptions::default()
    };
    let handle = engine.session_handle(0, options.clone()).unwrap();
    assert_eq!(handle.memory_report().graph_payload_bytes, graph.len());
    assert_eq!(handle.memory_report().initializer_count(), 1);
    assert_eq!(handle.memory_report().initializer_materialized_bytes, 16);

    let outputs = engine
        .run(
            0,
            options,
            HashMap::from([(
                "x".to_string(),
                RuntimeTensor::I64 {
                    shape: vec![2],
                    data: vec![2, 9],
                },
            )]),
        )
        .unwrap();

    assert_eq!(
        outputs.get("z"),
        Some(&RuntimeTensor::I64 {
            shape: vec![2],
            data: vec![12, 5],
        })
    );
}

#[test]
fn missing_initializer_tensor_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing-initializer.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_add_external_initializer_onnx_model()))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let err = engine
        .session_handle(
            0,
            SessionOptions {
                initializers: vec![InitializerBinding::new("bias", "missing.tensor")],
                ..SessionOptions::default()
            },
        )
        .unwrap_err();

    assert!(matches!(
        err,
        RuntimeError::InitializerTensorNotFound {
            initializer_name,
            tensor_name
        } if initializer_name == "bias" && tensor_name == "missing.tensor"
    ));
}

#[test]
fn initializer_shape_mismatch_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shape-mismatch-initializer.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(
            tiny_add_external_initializer_onnx_model_with_dtype_shape(1, &[3]),
        ))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let err = engine
        .session_handle(
            0,
            SessionOptions {
                initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
                ..SessionOptions::default()
            },
        )
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("shape"))
    );
}

#[test]
fn initializer_dtype_mismatch_is_typed_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dtype-mismatch-initializer.rsmf");
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "bias.tensor".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![2],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(f32_bytes(&[10.0, -4.0])),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(
            tiny_add_external_initializer_onnx_model_with_dtype_shape(7, &[2]),
        ))
        .write_to_path(&path)
        .unwrap();

    let engine = Engine::new(RsmfFile::open(path).unwrap()).unwrap();
    let err = engine
        .session_handle(
            0,
            SessionOptions {
                initializers: vec![InitializerBinding::new("bias", "bias.tensor")],
                ..SessionOptions::default()
            },
        )
        .unwrap_err();

    assert!(
        matches!(err, RuntimeError::UnsupportedInitializer { reason, .. } if reason.contains("dtype"))
    );
}

#[test]
fn executor_runs_same_priority_fifo() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-fifo.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(add_request("first", 1.0, 10.0).with_priority(7))
        .unwrap();
    let second = executor
        .submit(add_request("second", 2.0, 20.0).with_priority(7))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let first_response = first.receiver.try_recv().unwrap().unwrap();
    assert_eq!(first_response.request_id, "first");
    assert_eq!(f32_output(&first_response, "z"), vec![11.0, 11.0]);
    assert!(matches!(
        second.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));

    assert!(executor.execute_next().unwrap());
    let second_response = second.wait().unwrap();
    assert_eq!(second_response.request_id, "second");
    assert_eq!(f32_output(&second_response, "z"), vec![22.0, 22.0]);
}

#[test]
fn executor_runs_higher_priority_first() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-priority.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let low = executor
        .submit(add_request("low", 1.0, 10.0).with_priority(1))
        .unwrap();
    let high = executor
        .submit(add_request("high", 2.0, 20.0).with_priority(9))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let high_response = high.receiver.try_recv().unwrap().unwrap();
    assert_eq!(high_response.request_id, "high");
    assert_eq!(f32_output(&high_response, "z"), vec![22.0, 22.0]);
    assert!(matches!(
        low.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
}

#[test]
fn executor_rejects_expired_deadline_before_runtime_dispatch() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-deadline.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let expired = Instant::now() - Duration::from_secs(1);
    let handle = executor
        .submit(RuntimeRequest::new("expired", 99, RuntimeInputs::new()).with_deadline(expired))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let err = handle.wait().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::RequestDeadlineExceeded { request_id } if request_id == "expired"
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 0);
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.deadline_expired, 1);
    assert_eq!(metrics.cancelled, 0);
}

#[test]
fn executor_rejects_zero_timeout_before_runtime_dispatch() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-timeout.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let handle = executor
        .submit(
            RuntimeRequest::new("timeout", 99, RuntimeInputs::new()).with_timeout(Duration::ZERO),
        )
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let err = handle.wait().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::RequestDeadlineExceeded { request_id } if request_id == "timeout"
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 0);
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.deadline_expired, 1);
    assert_eq!(metrics.cancelled, 0);
}

#[test]
fn executor_cancels_queued_request_before_runtime_dispatch() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-cancel.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let handle = executor
        .submit(RuntimeRequest::new("cancelled", 99, RuntimeInputs::new()))
        .unwrap();

    assert_eq!(handle.cancel(), RuntimeCancellationResult::Cancelled);
    assert_eq!(handle.cancel(), RuntimeCancellationResult::AlreadyCancelled);
    assert!(executor.execute_next().unwrap());
    let err = handle.wait().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::RequestCancelled { request_id } if request_id == "cancelled"
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 0);
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.deadline_expired, 0);
    assert_eq!(metrics.cancelled, 1);
}

#[test]
fn cancellation_after_completion_reports_completed() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-completed-cancel.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let handle = executor.submit(add_request("done", 1.0, 10.0)).unwrap();
    let token = handle.cancellation_token();

    assert!(executor.execute_next().unwrap());
    let response = handle.wait().unwrap();
    assert_eq!(response.request_id, "done");
    assert_eq!(token.cancel(), RuntimeCancellationResult::AlreadyCompleted);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 1);
    assert_eq!(metrics.failed, 0);
    assert_eq!(metrics.cancelled, 0);
}

#[test]
fn running_cancellation_requests_ort_termination() {
    let token = RuntimeCancellationToken::new();
    assert!(token.try_mark_running().is_ok());
    let run_options = Arc::new(RunOptions::new().unwrap());
    token.attach_run_options(Arc::clone(&run_options)).unwrap();

    assert_eq!(
        token.cancel(),
        RuntimeCancellationResult::RunningCancellationRequested
    );
    assert!(token.is_cancellation_requested());
}

#[test]
fn pre_requested_running_cancellation_terminates_ort_run() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-preterminated-run.rsmf"));
    let handle = engine.session_handle(0, SessionOptions::default()).unwrap();
    let token = RuntimeCancellationToken::new();
    assert!(token.try_mark_running().is_ok());
    assert_eq!(token.cancel(), RuntimeCancellationResult::AlreadyRunning);

    let err = handle
        .run_with_cancellation(
            HashMap::from([
                (
                    "x".to_string(),
                    RuntimeTensor::F32 {
                        shape: vec![2],
                        data: vec![1.0, 2.0],
                    },
                ),
                (
                    "y".to_string(),
                    RuntimeTensor::F32 {
                        shape: vec![2],
                        data: vec![10.0, 20.0],
                    },
                ),
            ]),
            Some(&token),
        )
        .unwrap_err();

    assert!(matches!(err, RuntimeError::Ort { message, .. } if message.contains("terminate")));
    token.mark_completed();
}

#[test]
fn executor_preserves_runtime_errors() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-error.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let handle = executor
        .submit(RuntimeRequest::new(
            "missing-graph",
            99,
            RuntimeInputs::new(),
        ))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let err = handle.wait().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::GraphNotFound {
            graph_idx: 99,
            graph_count: 1
        }
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 0);
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.deadline_expired, 0);
    assert_eq!(metrics.cancelled, 0);
}

#[test]
fn executor_queue_capacity_is_enforced() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-capacity.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 1,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let _handle = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
    let err = executor
        .submit(add_request("second", 2.0, 20.0))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorQueueFull { capacity: 1 }
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.rejected_by_capacity, 1);
    assert_eq!(metrics.rejected_by_memory, 0);
    assert_eq!(metrics.current_queue_depth, 1);
    assert_eq!(metrics.max_observed_queue_depth, 1);
    assert_eq!(metrics.current_queued_tensor_bytes, 16);
    assert_eq!(metrics.max_observed_queued_tensor_bytes, 16);
}

#[test]
fn executor_memory_budget_is_enforced_and_reported() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-memory-budget.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                max_queued_tensor_bytes: Some(16),
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
    let err = executor
        .submit(add_request("second", 2.0, 20.0))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorQueueBytesExceeded {
            requested_bytes: 16,
            queued_bytes: 16,
            capacity_bytes: 16,
        }
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.rejected_by_capacity, 0);
    assert_eq!(metrics.rejected_by_memory, 1);
    assert_eq!(metrics.current_queue_depth, 1);
    assert_eq!(metrics.current_queued_tensor_bytes, 16);
    assert_eq!(metrics.max_observed_queue_depth, 1);
    assert_eq!(metrics.max_observed_queued_tensor_bytes, 16);

    assert!(executor.execute_next().unwrap());
    let response = first.wait().unwrap();
    assert_eq!(f32_output(&response, "z"), vec![11.0, 11.0]);
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.current_queue_depth, 0);
    assert_eq!(metrics.current_queued_tensor_bytes, 0);
    assert_eq!(metrics.completed, 1);
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.active_requests, 0);
    assert_eq!(metrics.active_runtime_invocations, 0);
    assert_eq!(metrics.active_batch_size, 0);
    assert_eq!(metrics.max_active_requests, 1);
    assert_eq!(metrics.max_active_runtime_invocations, 1);
    assert_eq!(metrics.max_active_batch_size, 1);
}

#[test]
fn executor_hard_memory_pressure_is_enforced_and_reported() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-hard-pressure.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                memory_pressure: RuntimeMemoryPressureConfig {
                    hard_queued_tensor_bytes: Some(16),
                    ..RuntimeMemoryPressureConfig::default()
                },
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
    let err = executor
        .submit(add_request("second", 2.0, 20.0))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorMemoryPressureExceeded {
            requested_bytes: 16,
            queued_bytes: 16,
            hard_limit_bytes: 16,
        }
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.rejected_by_memory, 0);
    assert_eq!(metrics.rejected_by_memory_pressure, 1);
    assert_eq!(metrics.memory_pressure_hard_rejections, 1);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Hard
    );

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 11.0]);
    let metrics = executor.metrics().unwrap();
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Normal
    );
}

#[test]
fn executor_soft_memory_pressure_is_observable_and_released() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-soft-pressure.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                memory_pressure: RuntimeMemoryPressureConfig {
                    soft_queued_tensor_bytes: Some(16),
                    ..RuntimeMemoryPressureConfig::default()
                },
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.memory_pressure_soft_events, 1);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Soft
    );

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 11.0]);
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.memory_pressure_soft_events, 1);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Normal
    );
}

#[test]
fn executor_enforces_tenant_queue_capacity_and_releases_on_dispatch() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-tenant-capacity.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                max_queued_requests_per_tenant: Some(1),
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor
        .submit(add_request("alpha-1", 1.0, 10.0).with_tenant_id("alpha"))
        .unwrap();
    let err = executor
        .submit(add_request("alpha-2", 2.0, 20.0).with_tenant_id("alpha"))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorTenantQueueFull {
            tenant_id,
            capacity: 1,
        } if tenant_id == "alpha"
    ));

    let beta = executor
        .submit(add_request("beta-1", 3.0, 30.0).with_tenant_id("beta"))
        .unwrap();
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 2);
    assert_eq!(metrics.rejected_by_tenant_capacity, 1);
    assert_eq!(tenant_metric(&metrics, "alpha").current_queued_requests, 1);
    assert_eq!(tenant_metric(&metrics, "alpha").rejected_by_capacity, 1);
    assert_eq!(tenant_metric(&metrics, "beta").current_queued_requests, 1);

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 11.0]);
    let second_alpha = executor
        .submit(add_request("alpha-3", 4.0, 40.0).with_tenant_id("alpha"))
        .unwrap();
    let metrics = executor.metrics().unwrap();
    assert_eq!(tenant_metric(&metrics, "alpha").current_queued_requests, 1);
    assert_eq!(
        tenant_metric(&metrics, "alpha").max_observed_queued_requests,
        1
    );

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&beta.wait().unwrap(), "z"), vec![33.0, 33.0]);
    assert!(executor.execute_next().unwrap());
    assert_eq!(
        f32_output(&second_alpha.wait().unwrap(), "z"),
        vec![44.0, 44.0]
    );
}

#[test]
fn executor_enforces_tenant_queued_tensor_byte_budget() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-tenant-memory.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                max_queued_tensor_bytes_per_tenant: Some(16),
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let _alpha = executor
        .submit(add_request("alpha-1", 1.0, 10.0).with_tenant_id("alpha"))
        .unwrap();
    let err = executor
        .submit(add_request("alpha-2", 2.0, 20.0).with_tenant_id("alpha"))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorTenantQueueBytesExceeded {
            tenant_id,
            requested_bytes: 16,
            queued_bytes: 16,
            capacity_bytes: 16,
        } if tenant_id == "alpha"
    ));
    let _beta = executor
        .submit(add_request("beta-1", 3.0, 30.0).with_tenant_id("beta"))
        .unwrap();

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 2);
    assert_eq!(metrics.rejected_by_tenant_memory, 1);
    assert_eq!(
        tenant_metric(&metrics, "alpha").current_queued_tensor_bytes,
        16
    );
    assert_eq!(tenant_metric(&metrics, "alpha").rejected_by_memory, 1);
    assert_eq!(
        tenant_metric(&metrics, "beta").current_queued_tensor_bytes,
        16
    );
}

#[test]
fn executor_batches_compatible_requests() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::ZERO,
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let first_response = first.wait().unwrap();
    let second_response = second.wait().unwrap();
    assert_eq!(f32_output_shape(&first_response, "z"), vec![1, 2]);
    assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output_shape(&second_response, "z"), vec![1, 2]);
    assert_eq!(f32_output(&second_response, "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 2);
    assert_eq!(metrics.completed, 2);
    assert_eq!(metrics.failed, 0);
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batched_requests, 2);
    assert_eq!(metrics.batch_fallbacks, 0);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 0);
    assert_eq!(metrics.batch_flushes_manual, 1);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
    assert_eq!(metrics.active_requests, 0);
    assert_eq!(metrics.active_runtime_invocations, 0);
    assert_eq!(metrics.active_batch_size, 0);
    assert_eq!(metrics.max_active_requests, 2);
    assert_eq!(metrics.max_active_runtime_invocations, 1);
    assert_eq!(metrics.max_active_batch_size, 2);
}

#[test]
fn executor_skips_incompatible_batch_candidates() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-batch-skip.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::ZERO,
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let incompatible = executor
        .submit(dynamic_add_request("incompatible", &[3.0, 4.0], &[30.0, 40.0]).with_priority(-1))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let first_response = first.wait().unwrap();
    assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
    assert!(matches!(
        incompatible.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));

    assert!(executor.execute_next().unwrap());
    let incompatible_response = incompatible.wait().unwrap();
    assert_eq!(f32_output(&incompatible_response, "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 2);
    assert_eq!(metrics.completed, 2);
    assert_eq!(metrics.runtime_invocations, 2);
    assert_eq!(metrics.batches_executed, 0);
    assert_eq!(metrics.batched_requests, 0);
    assert_eq!(metrics.batch_fallbacks, 0);
}

#[test]
fn executor_reports_full_batch_flush_reason() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-full-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 2,
                max_queue_delay: Duration::from_secs(1),
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output(&second.wait().unwrap(), "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batch_flushes_full, 1);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 0);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
}

#[test]
fn background_scheduler_collects_compatible_arrivals_until_delay() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-delay-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 1,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_millis(100),
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert!(matches!(
        first.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    let first_response = first.wait().unwrap();
    let second_response = second.wait().unwrap();
    assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output(&second_response, "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.completed, 2);
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batched_requests, 2);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 1);
    assert_eq!(metrics.batch_flushes_memory_pressure, 0);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
}

#[test]
fn background_scheduler_flushes_open_batch_on_shutdown() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-shutdown-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 1,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_secs(60),
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let handle = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert!(matches!(
        handle.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));

    executor.close().unwrap();
    let response = handle.wait().unwrap();
    assert_eq!(f32_output(&response, "z"), vec![11.0, 22.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.completed, 1);
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 0);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 1);
}

#[test]
fn executor_reports_memory_pressure_batch_flush_reason() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-pressure-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_secs(1),
            }),
            admission: RuntimeAdmissionConfig {
                max_queued_tensor_bytes: Some(32),
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output(&second.wait().unwrap(), "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 1);
    assert_eq!(metrics.memory_pressure_flushes, 1);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
}

#[test]
fn executor_flushes_dynamic_batch_on_soft_memory_pressure() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-soft-pressure-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_secs(1),
            }),
            admission: RuntimeAdmissionConfig {
                memory_pressure: RuntimeMemoryPressureConfig {
                    soft_queued_tensor_bytes: Some(32),
                    flush_dynamic_batches_on_soft_pressure: true,
                    ..RuntimeMemoryPressureConfig::default()
                },
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.memory_pressure_soft_events, 1);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Soft
    );

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output(&second.wait().unwrap(), "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 1);
    assert_eq!(metrics.memory_pressure_flushes, 1);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Normal
    );
}

#[test]
fn network_server_reports_health_and_metrics() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-health.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());

    let (status, body) = http_json(server.local_addr(), "GET", "/health", None);
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["protocol_version"], RUNTIME_NETWORK_PROTOCOL_VERSION);

    let (status, body) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(body["protocol_version"], RUNTIME_NETWORK_PROTOCOL_VERSION);
    assert_eq!(body["submitted"], 0);
    assert_eq!(body["runtime_invocations"], 0);

    server.shutdown().unwrap();
}

#[test]
fn network_server_runs_json_inference_request() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-run.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let request = serde_json::json!({
        "protocol_version": RUNTIME_NETWORK_PROTOCOL_VERSION,
        "request_id": "net-run",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 200);
    assert_eq!(body["protocol_version"], RUNTIME_NETWORK_PROTOCOL_VERSION);
    assert_eq!(body["request_id"], "net-run");
    assert_eq!(body["outputs"]["z"]["dtype"], "f32");
    assert_eq!(body["outputs"]["z"]["shape"], serde_json::json!([2]));
    assert_eq!(
        body["outputs"]["z"]["data"],
        serde_json::json!([11.0, 22.0])
    );

    let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(metrics["submitted"], 1);
    assert_eq!(metrics["completed"], 1);
    assert_eq!(metrics["runtime_invocations"], 1);

    server.shutdown().unwrap();
}

#[test]
fn network_server_rejects_unsupported_protocol_version() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-protocol-version.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let request = serde_json::json!({
        "protocol_version": RUNTIME_NETWORK_PROTOCOL_VERSION + 1,
        "request_id": "net-version",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "unsupported_protocol_version");
    assert_eq!(
        body["error"]["message"],
        "unsupported protocol version 2; supported version is 1"
    );

    let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(metrics["submitted"], 0);

    server.shutdown().unwrap();
}

#[test]
fn network_server_rejects_oversized_request_body() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-body-limit.rsmf"));
    let server = start_test_network_server_with_network_config(
        engine,
        RuntimeExecutorConfig::default(),
        RuntimeNetworkServerConfig {
            max_body_bytes: 8,
            ..RuntimeNetworkServerConfig::default()
        },
    );
    let (status, body) = http_raw_json(
        server.local_addr(),
        "POST /v1/run HTTP/1.1\r\nhost: test\r\ncontent-type: application/json\r\ncontent-length: 9\r\nconnection: close\r\n\r\n123456789",
    );

    assert_eq!(status, 413);
    assert_eq!(body["error"]["code"], "payload_too_large");
    assert_eq!(
        body["error"]["message"],
        "request body is 9 bytes, limit is 8"
    );

    server.shutdown().unwrap();
}

#[test]
fn network_server_rejects_unsupported_content_type() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-content-type.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let body = serde_json::json!({
        "request_id": "net-content-type",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    })
    .to_string();
    let request = format!(
        "POST /v1/run HTTP/1.1\r\nhost: test\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    let (status, body) = http_raw_json(server.local_addr(), &request);

    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "bad_request");
    assert_eq!(
        body["error"]["message"],
        "content-type must be application/json"
    );

    server.shutdown().unwrap();
}

#[test]
fn network_server_sanitizes_runtime_error_response() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-sanitized-error.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let request = serde_json::json!({
        "request_id": "net-runtime-error",
        "graph_idx": 99,
        "inputs": {}
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 500);
    assert_eq!(body["error"]["code"], "runtime_error");
    assert_eq!(body["error"]["message"], "runtime request failed");

    server.shutdown().unwrap();
}

#[test]
fn network_server_enforces_response_body_limit() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-response-limit.rsmf"));
    let server = start_test_network_server_with_network_config(
        engine,
        RuntimeExecutorConfig::default(),
        RuntimeNetworkServerConfig {
            max_response_body_bytes: 128,
            ..RuntimeNetworkServerConfig::default()
        },
    );
    let request = serde_json::json!({
        "request_id": "net-response-limit",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 500);
    assert_eq!(body["error"]["code"], "response_too_large");
    assert_eq!(
        body["error"]["message"],
        "response exceeded configured size limit"
    );

    server.shutdown().unwrap();
}

#[test]
fn network_server_propagates_tenant_id_to_metrics() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-tenant.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let request = serde_json::json!({
        "request_id": "net-tenant",
        "tenant_id": "tenant-a",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 200);
    assert_eq!(body["request_id"], "net-tenant");

    let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(metrics["tenant_metrics"][0]["tenant_id"], "tenant-a");
    assert_eq!(
        metrics["tenant_metrics"][0]["max_observed_queued_requests"],
        1
    );
    assert_eq!(metrics["tenant_metrics"][0]["current_queued_requests"], 0);

    server.shutdown().unwrap();
}

#[test]
fn network_server_cancels_inflight_request() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("network-cancel.rsmf"));
    let server = start_test_network_server(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 1,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_millis(100),
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let addr = server.local_addr();
    let request = serde_json::json!({
        "request_id": "net-cancel",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [1, 2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [1, 2], "data": [10.0, 20.0] }
        }
    });
    let request_thread =
        std::thread::spawn(move || http_json(addr, "POST", "/v1/run", Some(&request)));
    std::thread::sleep(Duration::from_millis(20));

    let (status, body) = http_json(server.local_addr(), "GET", "/v1/requests/net-cancel", None);
    assert_eq!(status, 200);
    assert_eq!(body["status"], "inflight");

    let (status, body) = http_json(
        server.local_addr(),
        "DELETE",
        "/v1/requests/net-cancel",
        None,
    );
    assert_eq!(status, 200);
    assert_eq!(body["cancellation"], "Cancelled");

    let (status, body) = request_thread.join().unwrap();
    assert_eq!(status, 499);
    assert_eq!(body["error"]["code"], "cancelled");

    let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(metrics["cancelled"], 1);

    server.shutdown().unwrap();
}

fn start_test_network_server(
    engine: Engine,
    executor_config: RuntimeExecutorConfig,
) -> RuntimeNetworkServerHandle {
    start_test_network_server_with_network_config(
        engine,
        executor_config,
        RuntimeNetworkServerConfig::default(),
    )
}

fn start_test_network_server_with_network_config(
    engine: Engine,
    executor_config: RuntimeExecutorConfig,
    network_config: RuntimeNetworkServerConfig,
) -> RuntimeNetworkServerHandle {
    RuntimeNetworkServer::new(
        RuntimeExecutor::new(engine, executor_config),
        network_config,
    )
    .start()
    .unwrap()
}

fn http_json(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&serde_json::Value>,
) -> (u16, serde_json::Value) {
    let body = body
        .map(serde_json::to_string)
        .transpose()
        .unwrap()
        .unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nhost: {addr}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    http_raw_json(addr, &request)
}

fn http_raw_json(addr: SocketAddr, request: &str) -> (u16, serde_json::Value) {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    let status = headers
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u16>()
        .unwrap();
    (status, serde_json::from_str(body).unwrap())
}

fn tenant_metric<'a>(
    metrics: &'a RuntimeExecutorMetrics,
    tenant_id: &str,
) -> &'a RuntimeTenantMetrics {
    metrics
        .tenant_metrics
        .iter()
        .find(|metrics| metrics.tenant_id == tenant_id)
        .unwrap()
}

fn tiny_add_onnx_model() -> Vec<u8> {
    let mut model = Vec::new();
    push_i64(&mut model, 1, 7);
    push_string(&mut model, 2, "rsmf-runtime-test");
    push_message(&mut model, 7, tiny_add_graph());
    push_message(&mut model, 8, opset_import("", 13));
    model
}

fn tiny_add_external_initializer_onnx_model() -> Vec<u8> {
    tiny_add_external_initializer_onnx_model_with_dtype_shape(1, &[2])
}

fn tiny_add_external_initializer_onnx_model_with_dtype_shape(
    data_type: i32,
    shape: &[i64],
) -> Vec<u8> {
    let mut model = Vec::new();
    push_i64(&mut model, 1, 7);
    push_string(&mut model, 2, "rsmf-runtime-test");
    push_message(
        &mut model,
        7,
        tiny_add_graph_with_external_initializer(data_type, shape),
    );
    push_message(&mut model, 8, opset_import("", 13));
    model
}

fn tiny_add_graph() -> Vec<u8> {
    let mut graph = Vec::new();
    push_message(&mut graph, 1, add_node());
    push_string(&mut graph, 2, "rsmf_add_graph");
    push_message(&mut graph, 11, value_info("x", &[2]));
    push_message(&mut graph, 11, value_info("y", &[2]));
    push_message(&mut graph, 12, value_info("z", &[2]));
    graph
}

fn tiny_dynamic_add_onnx_model() -> Vec<u8> {
    let mut model = Vec::new();
    push_i64(&mut model, 1, 7);
    push_string(&mut model, 2, "rsmf-runtime-test");
    push_message(&mut model, 7, tiny_dynamic_add_graph());
    push_message(&mut model, 8, opset_import("", 13));
    model
}

fn tiny_dynamic_add_graph() -> Vec<u8> {
    let mut graph = Vec::new();
    push_message(&mut graph, 1, add_node());
    push_string(&mut graph, 2, "rsmf_dynamic_add_graph");
    push_message(&mut graph, 11, dynamic_value_info("x"));
    push_message(&mut graph, 11, dynamic_value_info("y"));
    push_message(&mut graph, 12, dynamic_value_info("z"));
    graph
}

fn tiny_add_graph_with_external_initializer(data_type: i32, shape: &[i64]) -> Vec<u8> {
    let mut graph = Vec::new();
    push_message(&mut graph, 1, add_initializer_node());
    push_string(&mut graph, 2, "rsmf_add_initializer_graph");
    push_message(&mut graph, 5, external_tensor("bias", data_type, shape));
    push_message(&mut graph, 11, value_info_typed("x", &[2], data_type));
    push_message(&mut graph, 12, value_info_typed("z", &[2], data_type));
    graph
}

fn add_node() -> Vec<u8> {
    let mut node = Vec::new();
    push_string(&mut node, 1, "x");
    push_string(&mut node, 1, "y");
    push_string(&mut node, 2, "z");
    push_string(&mut node, 3, "add");
    push_string(&mut node, 4, "Add");
    node
}

fn add_initializer_node() -> Vec<u8> {
    let mut node = Vec::new();
    push_string(&mut node, 1, "x");
    push_string(&mut node, 1, "bias");
    push_string(&mut node, 2, "z");
    push_string(&mut node, 3, "add_initializer");
    push_string(&mut node, 4, "Add");
    node
}

fn external_tensor(name: &str, data_type: i32, shape: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    for &dim in shape {
        push_i64(&mut tensor, 1, dim);
    }
    push_i32(&mut tensor, 2, data_type);
    push_string(&mut tensor, 8, name);
    push_message(&mut tensor, 13, string_string_entry("location", "rsmf"));
    push_i32(&mut tensor, 14, 1);
    tensor
}

fn raw_variant(storage_dtype: StorageDtype, layout: LayoutTag, bytes: Vec<u8>) -> VariantInput {
    VariantInput {
        target: TargetTag::CpuGeneric,
        encoding: EncodingKind::Raw,
        storage_dtype: Some(storage_dtype),
        layout,
        alignment: 64,
        bytes,
        meta: VariantMeta::default(),
    }
}

fn string_string_entry(key: &str, value: &str) -> Vec<u8> {
    let mut entry = Vec::new();
    push_string(&mut entry, 1, key);
    push_string(&mut entry, 2, value);
    entry
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn add_graph_engine(path: std::path::PathBuf) -> Engine {
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_add_onnx_model()))
        .write_to_path(&path)
        .unwrap();
    Engine::new(RsmfFile::open(path).unwrap()).unwrap()
}

struct NativeDecoderFixtureOptions {
    include_tokenizer: bool,
    omit_tensor: Option<String>,
    bad_shape: Option<(String, Vec<u64>)>,
}

impl NativeDecoderFixtureOptions {
    fn default() -> Self {
        Self {
            include_tokenizer: true,
            omit_tensor: None,
            bad_shape: None,
        }
    }
}

fn tiny_native_decoder_engine(
    path: std::path::PathBuf,
    options: NativeDecoderFixtureOptions,
) -> Engine {
    let mut writer = RsmfWriter::new()
        .with_metadata("model.arch", "llama")
        .with_asset(AssetInput::new(
            NATIVE_DECODER_CONFIG_ASSET,
            tiny_native_decoder_config_json().into_bytes(),
        ))
        .with_asset(AssetInput::new(
            NATIVE_DECODER_GENERATION_CONFIG_ASSET,
            br#"{"max_new_tokens": 4}"#.to_vec(),
        ));
    if options.include_tokenizer {
        writer = writer.with_asset(AssetInput::new(
            NATIVE_DECODER_TOKENIZER_ASSET,
            br#"{"model": {"type": "BPE"}}"#.to_vec(),
        ));
    }
    for (name, shape) in tiny_native_decoder_tensor_specs() {
        if options.omit_tensor.as_deref() == Some(name.as_str()) {
            continue;
        }
        let shape = options
            .bad_shape
            .as_ref()
            .and_then(|(bad_name, bad_shape)| (bad_name == &name).then(|| bad_shape.clone()))
            .unwrap_or(shape);
        writer = writer.with_tensor(native_decoder_tensor(&name, shape));
    }
    writer.write_to_path(&path).unwrap();
    Engine::new(RsmfFile::open(path).unwrap()).unwrap()
}

fn tiny_native_decoder_generation_engine(path: std::path::PathBuf) -> Engine {
    let config = tiny_native_decoder_cpu_config();
    let mut writer = RsmfWriter::new()
        .with_metadata("model.arch", "llama")
        .with_asset(AssetInput::new(
            NATIVE_DECODER_CONFIG_ASSET,
            tiny_native_decoder_generation_config_json().into_bytes(),
        ))
        .with_asset(AssetInput::new(
            NATIVE_DECODER_TOKENIZER_ASSET,
            tiny_native_decoder_tokenizer_json().into_bytes(),
        ));
    for expected in expected_native_decoder_tensors(&config).unwrap() {
        let values =
            tiny_native_decoder_generation_tensor_values(&expected.tensor_name, &expected.shape);
        writer = writer.with_tensor(native_decoder_tensor_with_values(
            &expected.tensor_name,
            expected.shape,
            values,
        ));
    }
    writer.write_to_path(&path).unwrap();
    Engine::new(RsmfFile::open(path).unwrap()).unwrap()
}

fn tiny_native_decoder_config_json() -> String {
    serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "max_position_embeddings": 16,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": false
    })
    .to_string()
}

fn tiny_native_decoder_generation_config_json() -> String {
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

fn tiny_native_decoder_cpu_config() -> NativeDecoderConfig {
    NativeDecoderConfig {
        family: NativeDecoderFamily::Llama,
        hidden_size: 2,
        intermediate_size: 2,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        vocab_size: 4,
        max_position_embeddings: 8,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        bos_token_id: Some(1),
        eos_token_ids: vec![2],
        pad_token_id: None,
    }
}

fn tiny_native_decoder_generation_tensor_values(name: &str, shape: &[u64]) -> Vec<f32> {
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

fn tiny_native_decoder_tensor_specs() -> Vec<(String, Vec<u64>)> {
    vec![
        ("model.embed_tokens.weight".to_string(), vec![8, 4]),
        ("model.norm.weight".to_string(), vec![4]),
        ("lm_head.weight".to_string(), vec![8, 4]),
        ("model.layers.0.input_layernorm.weight".to_string(), vec![4]),
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            vec![4],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![4, 4],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            vec![2, 4],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            vec![2, 4],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            vec![4, 4],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![6, 4],
        ),
        ("model.layers.0.mlp.up_proj.weight".to_string(), vec![6, 4]),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![4, 6],
        ),
    ]
}

fn native_decoder_tensor(name: &str, shape: Vec<u64>) -> TensorInput {
    let elements = shape.iter().product::<u64>() as usize;
    TensorInput {
        name: name.to_string(),
        dtype: LogicalDtype::F32,
        shape,
        shard_id: 0,
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(vec![0u8; elements * 4]),
        packed: Vec::new(),
    }
}

fn native_decoder_tensor_with_values(name: &str, shape: Vec<u64>, values: Vec<f32>) -> TensorInput {
    assert_eq!(values.len(), shape.iter().product::<u64>() as usize);
    TensorInput {
        name: name.to_string(),
        dtype: LogicalDtype::F32,
        shape,
        shard_id: 0,
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(f32_bytes(&values)),
        packed: Vec::new(),
    }
}

fn dynamic_add_graph_engine(path: std::path::PathBuf) -> Engine {
    RsmfWriter::new()
        .with_tensor(TensorInput {
            name: "fixture.weight".to_string(),
            dtype: LogicalDtype::F32,
            shape: vec![1],
            shard_id: 0,
            metadata: Vec::new(),
            canonical: VariantInput::canonical_raw(0.0f32.to_le_bytes().to_vec()),
            packed: Vec::new(),
        })
        .with_graph(GraphInput::onnx(tiny_dynamic_add_onnx_model()))
        .write_to_path(&path)
        .unwrap();
    Engine::new(RsmfFile::open(path).unwrap()).unwrap()
}

fn add_request(request_id: &str, x: f32, y: f32) -> RuntimeRequest {
    RuntimeRequest::new(
        request_id,
        0,
        HashMap::from([
            (
                "x".to_string(),
                RuntimeTensor::F32 {
                    shape: vec![2],
                    data: vec![x, x],
                },
            ),
            (
                "y".to_string(),
                RuntimeTensor::F32 {
                    shape: vec![2],
                    data: vec![y, y],
                },
            ),
        ]),
    )
}

fn dynamic_add_request(request_id: &str, x: &[f32], y: &[f32]) -> RuntimeRequest {
    RuntimeRequest::new(
        request_id,
        0,
        HashMap::from([
            (
                "x".to_string(),
                RuntimeTensor::F32 {
                    shape: vec![1, x.len()],
                    data: x.to_vec(),
                },
            ),
            (
                "y".to_string(),
                RuntimeTensor::F32 {
                    shape: vec![1, y.len()],
                    data: y.to_vec(),
                },
            ),
        ]),
    )
}

fn f32_output(response: &RuntimeResponse, name: &str) -> Vec<f32> {
    match response.outputs.get(name).unwrap() {
        RuntimeTensor::F32 { data, .. } => data.clone(),
        other => panic!("expected F32 output, got {other:?}"),
    }
}

fn assert_close_slice(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "index {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn f32_output_shape(response: &RuntimeResponse, name: &str) -> Vec<usize> {
    match response.outputs.get(name).unwrap() {
        RuntimeTensor::F32 { shape, .. } => shape.clone(),
        other => panic!("expected F32 output, got {other:?}"),
    }
}

fn value_info(name: &str, shape: &[i64]) -> Vec<u8> {
    value_info_typed(name, shape, 1)
}

fn value_info_typed(name: &str, shape: &[i64], data_type: i32) -> Vec<u8> {
    let mut value = Vec::new();
    push_string(&mut value, 1, name);
    push_message(&mut value, 2, type_proto(data_type, shape));
    value
}

fn dynamic_value_info(name: &str) -> Vec<u8> {
    let mut value = Vec::new();
    push_string(&mut value, 1, name);
    push_message(&mut value, 2, dynamic_type_proto());
    value
}

fn type_proto(data_type: i32, shape: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    push_i32(&mut tensor, 1, data_type);
    push_message(&mut tensor, 2, tensor_shape(shape));

    let mut type_proto = Vec::new();
    push_message(&mut type_proto, 1, tensor);
    type_proto
}

fn dynamic_type_proto() -> Vec<u8> {
    let mut tensor = Vec::new();
    push_i32(&mut tensor, 1, 1);
    push_message(&mut tensor, 2, dynamic_tensor_shape());

    let mut type_proto = Vec::new();
    push_message(&mut type_proto, 1, tensor);
    type_proto
}

fn tensor_shape(shape: &[i64]) -> Vec<u8> {
    let mut tensor_shape = Vec::new();
    for &dim in shape {
        let mut dimension = Vec::new();
        push_i64(&mut dimension, 1, dim);
        push_message(&mut tensor_shape, 1, dimension);
    }
    tensor_shape
}

fn dynamic_tensor_shape() -> Vec<u8> {
    let mut tensor_shape = Vec::new();
    let mut batch = Vec::new();
    push_string(&mut batch, 2, "batch");
    push_message(&mut tensor_shape, 1, batch);

    let mut width = Vec::new();
    push_i64(&mut width, 1, 2);
    push_message(&mut tensor_shape, 1, width);
    tensor_shape
}

fn opset_import(domain: &str, version: i64) -> Vec<u8> {
    let mut opset = Vec::new();
    if !domain.is_empty() {
        push_string(&mut opset, 1, domain);
    }
    push_i64(&mut opset, 2, version);
    opset
}

fn push_i32(out: &mut Vec<u8>, field: u32, value: i32) {
    push_varint_field(out, field, value as u64);
}

fn push_i64(out: &mut Vec<u8>, field: u32, value: i64) {
    push_varint_field(out, field, value as u64);
}

fn push_string(out: &mut Vec<u8>, field: u32, value: &str) {
    push_bytes(out, field, value.as_bytes());
}

fn push_message(out: &mut Vec<u8>, field: u32, message: Vec<u8>) {
    push_bytes(out, field, &message);
}

fn push_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    push_tag(out, field, 2);
    push_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn push_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
    push_tag(out, field, 0);
    push_varint(out, value);
}

fn push_tag(out: &mut Vec<u8>, field: u32, wire_type: u64) {
    push_varint(out, ((field as u64) << 3) | wire_type);
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}
