use super::NativeDecoderTokenizerMode;
use crate::{Result, RuntimeError};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub(crate) struct NativeDecoderTokenizerJson {
    pub(crate) model: NativeDecoderTokenizerModelJson,
    #[serde(default)]
    pub(crate) normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) post_processor: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) decoder: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) added_tokens: Vec<NativeDecoderAddedTokenJson>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NativeDecoderTokenizerModelJson {
    #[serde(rename = "type")]
    pub(crate) tokenizer_type: String,
    #[serde(default)]
    pub(crate) vocab: NativeDecoderTokenizerVocabJson,
    #[serde(default)]
    pub(crate) unk_token: Option<String>,
    #[serde(default)]
    pub(crate) unk_id: Option<usize>,
    #[serde(default)]
    pub(crate) merges: Vec<NativeDecoderBpeMergeJson>,
    #[serde(default)]
    pub(crate) byte_fallback: bool,
    #[serde(default)]
    pub(crate) continuing_subword_prefix: Option<String>,
    #[serde(default)]
    pub(crate) max_input_chars_per_word: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum NativeDecoderTokenizerVocabJson {
    Map(HashMap<String, i64>),
    Unigram(Vec<(String, f64)>),
}

impl Default for NativeDecoderTokenizerVocabJson {
    fn default() -> Self {
        Self::Map(HashMap::new())
    }
}

impl NativeDecoderTokenizerVocabJson {
    pub(crate) fn to_vocab_and_scores(
        &self,
        mode: NativeDecoderTokenizerMode,
    ) -> Result<(HashMap<String, i64>, HashMap<String, i64>)> {
        match (mode, self) {
            (_, Self::Map(vocab)) => Ok((vocab.clone(), HashMap::new())),
            (NativeDecoderTokenizerMode::Unigram, Self::Unigram(entries)) => {
                let mut vocab = HashMap::with_capacity(entries.len());
                let mut scores = HashMap::with_capacity(entries.len());
                for (idx, (token, score)) in entries.iter().enumerate() {
                    let token_id = i64::try_from(idx).map_err(|_| {
                        RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("Unigram token index {idx} cannot convert to i64"),
                        }
                    })?;
                    vocab.insert(token.clone(), token_id);
                    scores.insert(token.clone(), (score * 1_000_000.0).round() as i64);
                }
                Ok((vocab, scores))
            }
            (_, Self::Unigram(_)) => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "array vocab is supported only for Unigram tokenizers".to_string(),
            }),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct NativeDecoderAddedTokenJson {
    pub(crate) id: i64,
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) special: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum NativeDecoderBpeMergeJson {
    Text(String),
    Pair([String; 2]),
}
