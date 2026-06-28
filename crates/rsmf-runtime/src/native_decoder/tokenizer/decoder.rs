use super::{
    decode_bpe_tokens, decode_sentencepiece_tokens, decode_wordpiece_tokens,
    replace_pattern_from_json,
};
use crate::{Result, RuntimeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderDecoder {
    None,
    WordPiece {
        prefix: String,
        cleanup: bool,
    },
    ByteLevel,
    Metaspace {
        replacement: char,
        add_prefix_space: bool,
    },
    Bpe {
        suffix: String,
    },
    Fuse,
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
    Replace {
        pattern: String,
        content: String,
    },
    Sequence(Vec<NativeDecoderDecoder>),
}

impl NativeDecoderDecoder {
    pub(crate) fn from_json(value: Option<&serde_json::Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::None);
        };
        if value.is_null() {
            return Ok(Self::None);
        }
        let Some(kind) = value.get("type").and_then(serde_json::Value::as_str) else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "decoder.type is required when decoder is present".to_string(),
            });
        };
        match kind {
            "WordPiece" => Ok(Self::WordPiece {
                prefix: value
                    .get("prefix")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("##")
                    .to_string(),
                cleanup: value
                    .get("cleanup")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
            }),
            "ByteLevel" => Ok(Self::ByteLevel),
            "Metaspace" => {
                let replacement = value
                    .get("replacement")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("▁");
                let mut chars = replacement.chars();
                let Some(replacement) = chars.next() else {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "decoder Metaspace replacement must not be empty".to_string(),
                    });
                };
                if chars.next().is_some() {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "decoder Metaspace replacement must be one character".to_string(),
                    });
                }
                Ok(Self::Metaspace {
                    replacement,
                    add_prefix_space: value
                        .get("add_prefix_space")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(true),
                })
            }
            "BPEDecoder" => Ok(Self::Bpe {
                suffix: value
                    .get("suffix")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("</w>")
                    .to_string(),
            }),
            "Fuse" => Ok(Self::Fuse),
            "Strip" => {
                let content = value
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(" ");
                let mut chars = content.chars();
                let Some(content) = chars.next() else {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "decoder Strip content must not be empty".to_string(),
                    });
                };
                if chars.next().is_some() {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "decoder Strip content must be one character".to_string(),
                    });
                }
                Ok(Self::Strip {
                    content,
                    start: value
                        .get("start")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(0),
                    stop: value
                        .get("stop")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(0),
                })
            }
            "Replace" => {
                let (pattern, regex) = replace_pattern_from_json(value.get("pattern"))?;
                if regex {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "decoder Replace regex patterns are not supported".to_string(),
                    });
                }
                Ok(Self::Replace {
                    pattern,
                    content: value
                        .get("content")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: "decoder Replace requires string content".to_string(),
                        })?
                        .to_string(),
                })
            }
            "Sequence" => {
                let decoders = value
                    .get("decoders")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "decoder Sequence requires a decoders array".to_string(),
                    })?
                    .iter()
                    .map(|decoder| Self::from_json(Some(decoder)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self::Sequence(decoders))
            }
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported decoder {other}"),
            }),
        }
    }

    pub(crate) fn decode(&self, tokens: &[String], special_tokens: &[String]) -> Result<String> {
        match self {
            Self::None => Ok(tokens.join("")),
            Self::WordPiece { prefix, cleanup } => {
                Ok(decode_wordpiece_tokens(tokens, prefix, *cleanup))
            }
            Self::ByteLevel => Ok(decode_bpe_tokens(tokens)),
            Self::Metaspace { replacement, .. } => {
                Ok(decode_sentencepiece_tokens(tokens, *replacement))
            }
            Self::Bpe { suffix } => Ok(tokens
                .iter()
                .map(|token| token.strip_suffix(suffix).unwrap_or(token))
                .collect::<Vec<_>>()
                .join(" ")),
            Self::Fuse => Ok(tokens
                .iter()
                .filter(|token| !special_tokens.contains(token))
                .cloned()
                .collect::<Vec<_>>()
                .join("")),
            Self::Strip {
                content,
                start,
                stop,
            } => {
                let mut decoded = tokens.join("");
                for _ in 0..*start {
                    decoded = decoded
                        .strip_prefix(*content)
                        .unwrap_or(&decoded)
                        .to_string();
                }
                for _ in 0..*stop {
                    decoded = decoded
                        .strip_suffix(*content)
                        .unwrap_or(&decoded)
                        .to_string();
                }
                Ok(decoded)
            }
            Self::Replace { pattern, content } => Ok(tokens.join("").replace(pattern, content)),
            Self::Sequence(decoders) => {
                let mut pieces = tokens.to_vec();
                for decoder in decoders {
                    pieces = vec![decoder.decode(&pieces, special_tokens)?];
                }
                Ok(pieces.join(""))
            }
        }
    }
}
