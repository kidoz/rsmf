use super::super::native_decoder::*;
use super::super::*;
use super::assert_close_slice;

use rsmf_core::RsmfFile;
use rsmf_core::writer::{
    AssetInput, RsmfWriter, TensorInput, VariantInput, convert_f32_to_q4_0_bytes,
    convert_f32_to_q8_0_bytes,
};
use rsmf_core::{LogicalDtype, TargetTag};
use serde::Deserialize;
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
fn engine_native_decoder_tokenizer_loads_sentencepiece_model_asset() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_engine(
        dir.path().join("native-decoder-tokenizer-model.rsmf"),
        NativeDecoderFixtureOptions {
            include_tokenizer: false,
            include_sentencepiece_model: true,
            ..NativeDecoderFixtureOptions::default()
        },
    );

    let contract = engine.native_decoder_contract().unwrap();
    assert_eq!(
        contract.assets.tokenizer_asset,
        NATIVE_DECODER_SENTENCEPIECE_MODEL_ASSET
    );
    let tokenizer = engine.native_decoder_tokenizer().unwrap();
    assert_eq!(tokenizer.encode("hello world").unwrap(), vec![1, 2]);
}

#[test]
fn engine_native_decoder_tokenizer_rejects_unsupported_model_type() {
    let err =
        NativeDecoderTokenizer::from_json(br#"{"model": {"type": "SentencePiece"}}"#).unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderTokenizerInvalid { reason } if reason.contains("only WordLevel, BPE, Unigram, and WordPiece"))
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
fn native_decoder_bpe_tokenizer_supports_byte_fallback() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<0xC3>": 0,
                    "<0xA9>": 1
                },
                "merges": [],
                "byte_fallback": true
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("é").unwrap(), vec![0, 1]);
    assert_eq!(tokenizer.decode(&[0, 1]).unwrap(), "é");
}

#[test]
fn native_decoder_unigram_tokenizer_segments_with_scores_and_metaspace() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "Unigram",
                "vocab": [
                    ["<unk>", 0.0],
                    ["▁hello", -0.1],
                    ["▁world", -0.1],
                    ["▁", -3.0],
                    ["hello", -1.0],
                    ["world", -1.0],
                    ["wor", -2.0],
                    ["ld", -2.0]
                ],
                "unk_id": 0
            },
            "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "add_prefix_space": true }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.model_type, "Unigram");
    assert_eq!(tokenizer.encode("hello world").unwrap(), vec![1, 2]);
    assert_eq!(tokenizer.decode(&[1, 2]).unwrap(), "hello world");
}

#[test]
fn native_decoder_unigram_tokenizer_uses_unk_for_unmatched_chars() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "Unigram",
                "vocab": [
                    ["<unk>", 0.0],
                    ["▁hello", -0.1]
                ],
                "unk_id": 0
            },
            "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "add_prefix_space": true }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("hello!").unwrap(), vec![1, 0]);
}

#[test]
fn native_decoder_sentencepiece_model_protobuf_encodes_unigram() {
    let tokenizer =
        NativeDecoderTokenizer::from_sentencepiece_model(&tiny_sentencepiece_unigram_model())
            .unwrap();

    assert_eq!(tokenizer.model_type, "SentencePieceUnigram");
    assert_eq!(tokenizer.encode("hello world").unwrap(), vec![1, 2]);
    assert_eq!(tokenizer.encode("hello!").unwrap(), vec![1, 0]);
    assert_eq!(tokenizer.decode(&[1, 2]).unwrap(), "hello world");
}

#[test]
fn native_decoder_sentencepiece_model_rejects_precompiled_normalizer() {
    let err = NativeDecoderTokenizer::from_sentencepiece_model(
        &tiny_sentencepiece_unigram_model_with_precompiled_normalizer(),
    )
    .unwrap_err();

    assert!(
        matches!(err, RuntimeError::NativeDecoderTokenizerInvalid { reason } if reason.contains("precompiled normalizer"))
    );
}

#[test]
fn native_decoder_wordpiece_tokenizer_matches_bert_style_flow() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "normalizer": {
                "type": "BertNormalizer",
                "lowercase": true,
                "strip_accents": true,
                "handle_chinese_chars": true,
                "clean_text": true
            },
            "pre_tokenizer": { "type": "Whitespace" },
            "model": {
                "type": "WordPiece",
                "vocab": {
                    "[UNK]": 0,
                    "[CLS]": 1,
                    "[SEP]": 2,
                    "hello": 3,
                    "world": 4,
                    "!": 5,
                    "un": 6,
                    "##aff": 7,
                    "##able": 8
                },
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 12
            },
            "post_processor": {
                "type": "BertProcessing",
                "sep": ["[SEP]", 2],
                "cls": ["[CLS]", 1]
            },
            "decoder": {
                "type": "WordPiece",
                "prefix": "##",
                "cleanup": true
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(
        tokenizer.encode("Héllo unaffable world!").unwrap(),
        vec![1, 3, 6, 7, 8, 4, 5, 2]
    );
    assert_eq!(
        tokenizer.encode_pair("hello", "world").unwrap(),
        vec![1, 3, 2, 4, 2]
    );
    assert_eq!(
        tokenizer.decode(&[3, 6, 7, 8, 4, 5]).unwrap(),
        "hello unaffable world!"
    );
}

#[test]
fn native_decoder_wordpiece_tokenizer_uses_unk_for_overlong_or_unmatched_words() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "WordPiece",
                "vocab": { "[UNK]": 0, "ok": 1 },
                "unk_token": "[UNK]",
                "max_input_chars_per_word": 2
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("missing").unwrap(), vec![0]);
    assert_eq!(tokenizer.encode("toolong").unwrap(), vec![0]);
}

#[test]
fn native_decoder_tokenizer_applies_lowercase_normalizer() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "normalizer": { "type": "Lowercase" },
            "model": {
                "type": "BPE",
                "vocab": { "hello": 0, "world": 1 },
                "merges": []
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("HELLO WORLD").unwrap(), vec![0, 1]);
}

#[test]
fn native_decoder_tokenizer_applies_sequence_normalizer() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    { "type": "NFKC" },
                    { "type": "Lowercase" }
                ]
            },
            "model": {
                "type": "WordLevel",
                "vocab": { "hello": 0 }
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("ＨＥＬＬＯ").unwrap(), vec![0]);
}

#[test]
fn native_decoder_tokenizer_applies_nfc_normalizer() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "normalizer": { "type": "NFC" },
            "model": {
                "type": "WordLevel",
                "vocab": { "é": 0 }
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("e\u{301}").unwrap(), vec![0]);
}

#[test]
fn native_decoder_tokenizer_rejects_unsupported_normalizer() {
    let err = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "normalizer": { "type": "Precompiled" },
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
        matches!(err, RuntimeError::NativeDecoderTokenizerInvalid { reason } if reason.contains("unsupported normalizer"))
    );
}

#[test]
fn native_decoder_tokenizer_applies_strip_replace_and_split_components() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    { "type": "Strip", "strip_left": true, "strip_right": true },
                    { "type": "Replace", "pattern": { "String": "-" }, "content": " " },
                    { "type": "Lowercase" }
                ]
            },
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    { "type": "WhitespaceSplit" },
                    { "type": "Punctuation" },
                    { "type": "Digits", "individual_digits": true }
                ]
            },
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "!": 2,
                    "2": 3,
                    "0": 4
                }
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(
        tokenizer.encode("  HELLO-world!20  ").unwrap(),
        vec![0, 1, 2, 3, 4]
    );
}

#[test]
fn native_decoder_tokenizer_applies_roberta_processing_and_bpe_decoder() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<s>": 0,
                    "</s>": 1,
                    "low</w>": 2,
                    "er</w>": 3
                },
                "merges": []
            },
            "pre_tokenizer": { "type": "WhitespaceSplit" },
            "post_processor": {
                "type": "RobertaProcessing",
                "sep": ["</s>", 1],
                "cls": ["<s>", 0]
            },
            "decoder": {
                "type": "BPEDecoder",
                "suffix": "</w>"
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(
        tokenizer.encode_pair("low</w>", "er</w>").unwrap(),
        vec![0, 2, 1, 1, 3, 1]
    );
    assert_eq!(tokenizer.decode(&[2, 3]).unwrap(), "low er");
}

#[test]
fn native_decoder_tokenizer_applies_template_processing() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<s>": 1,
                    "</s>": 2,
                    "hello": 3,
                    "world": 4,
                    "<sep>": 5
                }
            },
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    { "SpecialToken": { "id": "<s>", "type_id": 0 } },
                    { "Sequence": { "id": "A", "type_id": 0 } },
                    { "SpecialToken": { "id": "</s>", "type_id": 0 } }
                ],
                "pair": [
                    { "Sequence": { "id": "A", "type_id": 0 } },
                    { "SpecialToken": { "id": "<sep>", "type_id": 0 } },
                    { "Sequence": { "id": "B", "type_id": 1 } },
                    { "SpecialToken": { "id": "</s>", "type_id": 1 } }
                ],
                "special_tokens": {
                    "<s>": { "id": "<s>", "ids": [1], "tokens": ["<s>"] },
                    "</s>": { "id": "</s>", "ids": [2], "tokens": ["</s>"] },
                    "<sep>": { "id": "<sep>", "ids": [5], "tokens": ["<sep>"] }
                }
            }
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("hello").unwrap(), vec![1, 3, 2]);
    assert_eq!(
        tokenizer.encode_pair("hello", "world").unwrap(),
        vec![3, 5, 4, 2]
    );
}

#[test]
fn native_decoder_tokenizer_isolates_adjacent_special_tokens() {
    let tokenizer = NativeDecoderTokenizer::from_json(
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<s>": 0,
                    "hello": 1
                },
                "merges": []
            },
            "added_tokens": [
                { "id": 0, "content": "<s>", "special": true }
            ]
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();

    assert_eq!(tokenizer.encode("<s>hello").unwrap(), vec![0, 1]);
}

#[test]
fn native_decoder_tokenizer_renders_chat_template_from_tokenizer_config() {
    let template = concat!(
        "{% for message in messages %}",
        "{{ '<|' + message['role'] + '|> ' + message['content'] + '\\n' }}",
        "{% endfor %}",
        "{% if add_generation_prompt %}",
        "{{ '<|assistant|> ' }}",
        "{% endif %}"
    );
    let tokenizer_config = serde_json::json!({ "chat_template": template }).to_string();
    let tokenizer = NativeDecoderTokenizer::from_json_with_assets(
        serde_json::json!({
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<|user|>": 0,
                    "hello": 1,
                    "<|assistant|>": 2
                }
            },
            "added_tokens": [
                { "id": 0, "content": "<|user|>", "special": true },
                { "id": 2, "content": "<|assistant|>", "special": true }
            ]
        })
        .to_string()
        .as_bytes(),
        Some(tokenizer_config.as_bytes()),
        None,
    )
    .unwrap();

    let messages = vec![NativeDecoderChatMessage::new("user", "hello")];
    assert_eq!(
        tokenizer.apply_chat_template(&messages, true).unwrap(),
        "<|user|> hello\n<|assistant|> "
    );
    assert_eq!(
        tokenizer.encode_chat(&messages, true).unwrap(),
        vec![0, 1, 2]
    );
}

#[test]
fn native_decoder_tokenizer_renders_chat_template_role_conditionals() {
    let template = concat!(
        "{% for message in messages %}",
        "{% if message['role'] == 'user' %}",
        "{{ '[INST] ' + message['content'] + ' [/INST] ' }}",
        "{% elif message['role'] == 'assistant' %}",
        "{{ message['content'] + ' ' }}",
        "{% else %}",
        "{{ message['content'] + ' ' }}",
        "{% endif %}",
        "{% endfor %}"
    );
    let tokenizer_config = serde_json::json!({ "chat_template": template }).to_string();
    let tokenizer = NativeDecoderTokenizer::from_json_with_assets(
        serde_json::json!({
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[INST]": 0,
                    "hello": 1,
                    "[/INST]": 2,
                    "hi": 3
                }
            }
        })
        .to_string()
        .as_bytes(),
        Some(tokenizer_config.as_bytes()),
        None,
    )
    .unwrap();

    let messages = vec![
        NativeDecoderChatMessage::new("user", "hello"),
        NativeDecoderChatMessage::new("assistant", "hi"),
    ];
    assert_eq!(
        tokenizer.apply_chat_template(&messages, false).unwrap(),
        "[INST] hello [/INST] hi "
    );
    assert_eq!(
        tokenizer.encode_chat(&messages, false).unwrap(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn native_decoder_tokenizer_renders_nested_minijinja_chat_template() {
    let template = concat!(
        "{{ bos_token }}",
        "{% for message in messages %}",
        "{% if message.role == 'user' %}",
        "{{ '<|user|>' + message.content|trim }}",
        "{% elif message.role == 'assistant' %}",
        "{{ '<|assistant|>' + message.content }}",
        "{% endif %}",
        "{% endfor %}",
        "{% for tool in tools %}{{ tool.name }}{% endfor %}",
        "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
    );
    let tokenizer_config = serde_json::json!({
        "bos_token": "<s>",
        "chat_template": template
    })
    .to_string();
    let tokenizer = NativeDecoderTokenizer::from_json_with_assets(
        serde_json::json!({
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<s><|user|>hello<|assistant|>hi<|assistant|>": 0
                }
            }
        })
        .to_string()
        .as_bytes(),
        Some(tokenizer_config.as_bytes()),
        None,
    )
    .unwrap();

    let messages = vec![
        NativeDecoderChatMessage::new("user", " hello "),
        NativeDecoderChatMessage::new("assistant", "hi"),
    ];

    assert_eq!(
        tokenizer.apply_chat_template(&messages, true).unwrap(),
        "<s><|user|>hello<|assistant|>hi<|assistant|>"
    );
    assert_eq!(tokenizer.encode_chat(&messages, true).unwrap(), vec![0]);
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
    let residency = session.residency_report();
    assert_eq!(
        residency.resident_weight_bytes,
        session.weights.resident_bytes()
    );
    assert!(residency.resident_weight_bytes > 0);
    assert_eq!(residency.resident_kv_cache_bytes, 0);

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
fn engine_native_decoder_session_prefix_cache_reuses_prompt_kv() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let session = engine.native_decoder_session().unwrap();
    let options = NativeDecoderRunOptions {
        max_new_tokens: 1,
        performance: NativeDecoderPerformanceOptions {
            prefix_cache_max_entries: Some(4),
            ..NativeDecoderPerformanceOptions::default()
        },
        ..NativeDecoderRunOptions::default()
    };

    let first = session
        .generate_token_ids(&[0, 1], options.clone())
        .unwrap();
    let second = session.generate_token_ids(&[0, 1], options).unwrap();
    let report = session.prefix_cache_report();

    assert_eq!(first.performance.prefix_cache_hit_tokens, 0);
    assert_eq!(first.performance.prefix_cache_miss_tokens, 2);
    assert_eq!(second.performance.prefix_cache_hit_tokens, 2);
    assert_eq!(second.performance.prefix_cache_miss_tokens, 0);
    assert_eq!(first.generated_token_ids, second.generated_token_ids);
    assert_eq!(report.entries, 1);
    assert_eq!(report.lookups, 2);
    assert_eq!(report.hits, 1);
    assert!(session.residency_report().resident_kv_cache_bytes > 0);
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
        "../../tests/fixtures/tiny_hf_native_decoder_reference.json"
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
    assert_eq!(cache.resident_bytes(), 0);
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
    let one_page_bytes = cache.resident_bytes();
    assert!(one_page_bytes > 0);
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
    assert_eq!(cache.resident_bytes(), one_page_bytes);
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
    assert!(cache.resident_bytes() > one_page_bytes);
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
fn engine_native_decoder_cpu_tiled_attention_matches_scalar() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let scalar = engine
        .native_decoder_greedy_decode(
            &[0, 1, 0],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                stop_token_ids: vec![3],
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();
    let tiled = engine
        .native_decoder_greedy_decode(
            &[0, 1, 0],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                stop_token_ids: vec![3],
                performance: NativeDecoderPerformanceOptions {
                    attention: NativeDecoderAttentionImplementation::CpuTiled,
                    kv_cache_page_size_tokens: Some(1),
                    ..NativeDecoderPerformanceOptions::default()
                },
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(tiled.generated_token_ids, scalar.generated_token_ids);
    assert_eq!(tiled.logits, scalar.logits);
}

#[test]
fn engine_native_decoder_loads_raw_f16_weights_into_f32_path() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine_with_dtype(
        dir.path().join("native-decode-f16.rsmf"),
        LogicalDtype::F16,
    );

    let output = engine
        .native_decoder_greedy_decode(
            &[0],
            NativeDecoderRunOptions {
                max_new_tokens: 2,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(output.generated_token_ids, vec![1, 2]);
}

#[test]
fn engine_native_decoder_raw_i8_weights_require_lossy_quantized_opt_in() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine_with_dtype(
        dir.path().join("native-decode-i8.rsmf"),
        LogicalDtype::I8,
    );

    let err = engine.native_decoder_weights().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::NativeDecoderTensorDtypeUnsupported { dtype, .. } if dtype == "I8"
    ));

    let weights = engine
        .native_decoder_weights_with_options(&NativeDecoderWeightOptions {
            allow_lossy_quantized: true,
            ..NativeDecoderWeightOptions::default()
        })
        .unwrap();
    assert_eq!(weights.token_embedding[0], 1.0);
    assert_eq!(weights.lm_head.as_ref().unwrap()[6], -1.0);
    assert!(matches!(
        weights.lm_head_quantized,
        Some(NativeDecoderQuantizedMatrix::RawI8 { .. })
    ));
    assert!(matches!(
        weights.layers[0].q_proj_quantized,
        Some(NativeDecoderQuantizedMatrix::RawI8 { .. })
    ));
    assert!(matches!(
        weights.layers[0].down_proj_quantized,
        Some(NativeDecoderQuantizedMatrix::RawI8 { .. })
    ));

    let output = native_decoder_generate_with_backend(
        &weights,
        &[0],
        NativeDecoderRunOptions {
            max_new_tokens: 2,
            ..NativeDecoderRunOptions::default()
        },
        NativeDecoderBackend::CpuReference,
    )
    .unwrap();
    assert_eq!(output.generated_token_ids, vec![1, 2]);
}

#[test]
fn engine_native_decoder_q8_lm_head_variant_uses_direct_quantized_kernel() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("native-decode-q8-head.rsmf");
    let engine = tiny_native_decoder_generation_engine_with_q8_lm_head_variant(path.clone());
    let file = RsmfFile::open(path).unwrap();
    let lm_head_variant = file
        .manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == "lm_head.weight")
        .and_then(|tensor| tensor.packed_variants.first().copied())
        .unwrap();
    let mut variants = std::collections::HashMap::new();
    variants.insert("lm_head.weight".to_string(), lm_head_variant);
    let weights = engine
        .native_decoder_weights_with_options(&NativeDecoderWeightOptions {
            tensor_variants: variants,
            allow_lossy_quantized: true,
        })
        .unwrap();
    assert!(matches!(
        weights.lm_head_quantized,
        Some(NativeDecoderQuantizedMatrix::Q8_0 { .. })
    ));

    let reference = engine
        .native_decoder_greedy_decode(
            &[0],
            NativeDecoderRunOptions {
                max_new_tokens: 1,
                ..NativeDecoderRunOptions::default()
            },
        )
        .unwrap();
    let quantized = native_decoder_generate_with_backend(
        &weights,
        &[0],
        NativeDecoderRunOptions {
            max_new_tokens: 1,
            ..NativeDecoderRunOptions::default()
        },
        NativeDecoderBackend::CpuReference,
    )
    .unwrap();

    assert_eq!(quantized.generated_token_ids, reference.generated_token_ids);
    assert_close_slice(&quantized.logits[0], &reference.logits[0], 1e-2);
}

#[test]
fn engine_native_decoder_q4_projection_variants_use_direct_quantized_kernels() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("native-decode-q4-projections.rsmf");
    let engine = tiny_native_decoder_generation_engine_with_q4_projection_variants(path.clone());
    let file = RsmfFile::open(path).unwrap();
    let mut variants = std::collections::HashMap::new();
    for tensor in &file.manifest().tensors {
        if let Some(variant) = tensor.packed_variants.first().copied() {
            variants.insert(tensor.name.clone(), variant);
        }
    }
    let weights = engine
        .native_decoder_weights_with_options(&NativeDecoderWeightOptions {
            tensor_variants: variants,
            allow_lossy_quantized: true,
        })
        .unwrap();

    assert!(matches!(
        weights.layers[0].q_proj_quantized,
        Some(NativeDecoderQuantizedMatrix::Q4_0 { .. })
    ));
    assert!(matches!(
        weights.layers[0].gate_proj_quantized,
        Some(NativeDecoderQuantizedMatrix::Q4_0 { .. })
    ));
    assert!(matches!(
        weights.lm_head_quantized,
        Some(NativeDecoderQuantizedMatrix::Q4_0 { .. })
    ));

    let output = native_decoder_generate_with_backend(
        &weights,
        &[0],
        NativeDecoderRunOptions {
            max_new_tokens: 2,
            ..NativeDecoderRunOptions::default()
        },
        NativeDecoderBackend::CpuReference,
    )
    .unwrap();
    assert_eq!(output.generated_token_ids, vec![1, 2]);
}

#[test]
fn engine_native_decoder_continuous_batch_interleaves_decode_steps() {
    let dir = tempdir().unwrap();
    let engine = tiny_native_decoder_generation_engine(dir.path().join("native-decode.rsmf"));
    let session = engine.native_decoder_session().unwrap();
    let options = NativeDecoderRunOptions {
        max_new_tokens: 2,
        performance: NativeDecoderPerformanceOptions {
            continuous_batch_max_requests: Some(2),
            ..NativeDecoderPerformanceOptions::default()
        },
        ..NativeDecoderRunOptions::default()
    };

    let (outputs, report) = session
        .generate_token_ids_continuous_batch(vec![
            NativeDecoderContinuousBatchRequest {
                request_id: "first".to_string(),
                input_token_ids: vec![0],
                options: options.clone(),
                deadline: None,
                cancelled: false,
            },
            NativeDecoderContinuousBatchRequest {
                request_id: "second".to_string(),
                input_token_ids: vec![0],
                options,
                deadline: None,
                cancelled: false,
            },
        ])
        .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].request_id, "first");
    assert_eq!(outputs[1].request_id, "second");
    assert_eq!(outputs[0].output.generated_token_ids, vec![1, 2]);
    assert_eq!(outputs[1].output.generated_token_ids, vec![1, 2]);
    assert_eq!(report.admitted_requests, 2);
    assert_eq!(report.decode_rounds, 2);
    assert_eq!(report.scheduled_decode_steps, 4);
    assert_eq!(report.fused_lm_head_batches, 1);
    assert_eq!(report.fused_lm_head_rows, 2);
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
            q_proj_quantized: None,
            k_proj: &zeros,
            k_proj_quantized: None,
            v_proj: &zeros,
            v_proj_quantized: None,
            o_proj: &zeros,
            o_proj_quantized: None,
            gate_proj: &zeros,
            gate_proj_quantized: None,
            up_proj: &zeros,
            up_proj_quantized: None,
            down_proj: &zeros,
            down_proj_quantized: None,
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
            q_proj_quantized: None,
            k_proj: &zeros,
            k_proj_quantized: None,
            v_proj: &zeros,
            v_proj_quantized: None,
            o_proj: &zeros,
            o_proj_quantized: None,
            gate_proj: &zeros,
            gate_proj_quantized: None,
            up_proj: &zeros,
            up_proj_quantized: None,
            down_proj: &zeros,
            down_proj_quantized: None,
        },
    )
    .unwrap_err();

    assert!(
        matches!(err, RuntimeError::Shape(message) if message.contains("linear: weight has 3 elements, expected 4"))
    );
}

struct NativeDecoderFixtureOptions {
    include_tokenizer: bool,
    include_sentencepiece_model: bool,
    omit_tensor: Option<String>,
    bad_shape: Option<(String, Vec<u64>)>,
}

impl NativeDecoderFixtureOptions {
    fn default() -> Self {
        Self {
            include_tokenizer: true,
            include_sentencepiece_model: false,
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
    if options.include_sentencepiece_model {
        writer = writer.with_asset(AssetInput::new(
            NATIVE_DECODER_SENTENCEPIECE_MODEL_ASSET,
            tiny_sentencepiece_unigram_model(),
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
    tiny_native_decoder_generation_engine_with_dtype(path, LogicalDtype::F32)
}

fn tiny_native_decoder_generation_engine_with_dtype(
    path: std::path::PathBuf,
    dtype: LogicalDtype,
) -> Engine {
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
        writer = writer.with_tensor(native_decoder_tensor_with_dtype_values(
            &expected.tensor_name,
            expected.shape,
            dtype,
            values,
        ));
    }
    writer.write_to_path(&path).unwrap();
    Engine::new(RsmfFile::open(path).unwrap()).unwrap()
}

fn tiny_native_decoder_generation_engine_with_q8_lm_head_variant(
    path: std::path::PathBuf,
) -> Engine {
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
        let mut tensor = native_decoder_tensor_with_dtype_values(
            &expected.tensor_name,
            expected.shape,
            LogicalDtype::F32,
            values,
        );
        if expected.tensor_name == "lm_head.weight" {
            tensor.packed.push(VariantInput::packed_q8_0(
                TargetTag::CpuGeneric,
                convert_f32_to_q8_0_bytes(&tensor.canonical.bytes),
            ));
        }
        writer = writer.with_tensor(tensor);
    }
    writer.write_to_path(&path).unwrap();
    Engine::new(RsmfFile::open(path).unwrap()).unwrap()
}

fn tiny_native_decoder_generation_engine_with_q4_projection_variants(
    path: std::path::PathBuf,
) -> Engine {
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
        let mut tensor = native_decoder_tensor_with_dtype_values(
            &expected.tensor_name,
            expected.shape,
            LogicalDtype::F32,
            values,
        );
        if expected.tensor_name == "lm_head.weight"
            || expected.tensor_name.contains(".self_attn.")
            || expected.tensor_name.contains(".mlp.")
        {
            tensor.packed.push(VariantInput::packed_q4_0(
                TargetTag::CpuGeneric,
                convert_f32_to_q4_0_bytes(&tensor.canonical.bytes),
            ));
        }
        writer = writer.with_tensor(tensor);
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

fn tiny_sentencepiece_unigram_model() -> Vec<u8> {
    tiny_sentencepiece_unigram_model_with_normalizer(None)
}

fn tiny_sentencepiece_unigram_model_with_precompiled_normalizer() -> Vec<u8> {
    tiny_sentencepiece_unigram_model_with_normalizer(Some(vec![1, 2, 3]))
}

fn tiny_sentencepiece_unigram_model_with_normalizer(
    precompiled_normalizer: Option<Vec<u8>>,
) -> Vec<u8> {
    let mut model = Vec::new();
    sp_push_message(&mut model, 1, sp_piece("<unk>", 0.0, 2));
    sp_push_message(&mut model, 1, sp_piece("▁hello", -0.1, 1));
    sp_push_message(&mut model, 1, sp_piece("▁world", -0.2, 1));
    let mut trainer = Vec::new();
    sp_push_varint(&mut trainer, 3, 1);
    sp_push_message(&mut model, 2, trainer);
    let mut normalizer = Vec::new();
    if let Some(precompiled_normalizer) = precompiled_normalizer {
        sp_push_bytes(&mut normalizer, 2, &precompiled_normalizer);
    }
    sp_push_varint(&mut normalizer, 3, 1);
    sp_push_message(&mut model, 3, normalizer);
    model
}

fn sp_piece(piece: &str, score: f32, kind: u64) -> Vec<u8> {
    let mut out = Vec::new();
    sp_push_bytes(&mut out, 1, piece.as_bytes());
    sp_push_fixed32(&mut out, 2, score.to_bits());
    sp_push_varint(&mut out, 3, kind);
    out
}

fn sp_push_message(out: &mut Vec<u8>, field: u64, bytes: Vec<u8>) {
    sp_push_bytes(out, field, &bytes);
}

fn sp_push_bytes(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    sp_push_raw_varint(out, (field << 3) | 2);
    sp_push_raw_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn sp_push_varint(out: &mut Vec<u8>, field: u64, value: u64) {
    sp_push_raw_varint(out, field << 3);
    sp_push_raw_varint(out, value);
}

fn sp_push_fixed32(out: &mut Vec<u8>, field: u64, value: u32) {
    sp_push_raw_varint(out, (field << 3) | 5);
    out.extend_from_slice(&value.to_le_bytes());
}

fn sp_push_raw_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
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

fn native_decoder_tensor_with_dtype_values(
    name: &str,
    shape: Vec<u64>,
    dtype: LogicalDtype,
    values: Vec<f32>,
) -> TensorInput {
    assert_eq!(values.len(), shape.iter().product::<u64>() as usize);
    TensorInput {
        name: name.to_string(),
        dtype,
        shape,
        shard_id: 0,
        metadata: Vec::new(),
        canonical: VariantInput::canonical_raw(native_decoder_value_bytes(dtype, &values)),
        packed: Vec::new(),
    }
}

fn native_decoder_value_bytes(dtype: LogicalDtype, values: &[f32]) -> Vec<u8> {
    match dtype {
        LogicalDtype::F32 => f32_bytes(values),
        LogicalDtype::F16 => values
            .iter()
            .flat_map(|value| match *value {
                0.0 => [0x00, 0x00],
                1.0 => [0x00, 0x3c],
                -1.0 => [0x00, 0xbc],
                other => panic!("test F16 helper only supports -1/0/1, got {other}"),
            })
            .collect(),
        LogicalDtype::BF16 => values
            .iter()
            .flat_map(|value| match *value {
                0.0 => [0x00, 0x00],
                1.0 => [0x80, 0x3f],
                -1.0 => [0x80, 0xbf],
                other => panic!("test BF16 helper only supports -1/0/1, got {other}"),
            })
            .collect(),
        LogicalDtype::I8 => values
            .iter()
            .map(|value| match *value {
                0.0 => 0u8,
                1.0 => 1u8,
                -1.0 => 255u8,
                other => panic!("test I8 helper only supports -1/0/1, got {other}"),
            })
            .collect(),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
