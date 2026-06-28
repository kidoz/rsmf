use crate::{Result, RuntimeError};
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderNormalizer {
    None,
    Lowercase,
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Strip {
        strip_left: bool,
        strip_right: bool,
    },
    StripAccents,
    Replace {
        pattern: String,
        content: String,
        regex: bool,
    },
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    },
    Sequence(Vec<NativeDecoderNormalizer>),
}

impl NativeDecoderNormalizer {
    pub(crate) fn from_json(value: Option<&serde_json::Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::None);
        };
        if value.is_null() {
            return Ok(Self::None);
        }
        let Some(kind) = value.get("type").and_then(serde_json::Value::as_str) else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "normalizer.type is required when normalizer is present".to_string(),
            });
        };
        match kind {
            "Lowercase" => Ok(Self::Lowercase),
            "NFC" => Ok(Self::Nfc),
            "NFD" => Ok(Self::Nfd),
            "NFKC" => Ok(Self::Nfkc),
            "NFKD" => Ok(Self::Nfkd),
            "Strip" => Ok(Self::Strip {
                strip_left: value
                    .get("strip_left")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
                strip_right: value
                    .get("strip_right")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
            }),
            "StripAccents" => Ok(Self::StripAccents),
            "Replace" => {
                let (pattern, regex) = replace_pattern_from_json(value.get("pattern"))?;
                let content = value
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "normalizer Replace requires string content".to_string(),
                    })?
                    .to_string();
                if regex {
                    Regex::new(&pattern).map_err(|error| {
                        RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("normalizer Replace regex is unsupported: {error}"),
                        }
                    })?;
                }
                Ok(Self::Replace {
                    pattern,
                    content,
                    regex,
                })
            }
            "BertNormalizer" | "Bert" => Ok(Self::Bert {
                clean_text: value
                    .get("clean_text")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
                handle_chinese_chars: value
                    .get("handle_chinese_chars")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
                strip_accents: value
                    .get("strip_accents")
                    .and_then(serde_json::Value::as_bool),
                lowercase: value
                    .get("lowercase")
                    .or_else(|| value.get("lower_case"))
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
            }),
            "Sequence" => {
                let normalizers = value
                    .get("normalizers")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "normalizer Sequence requires a normalizers array".to_string(),
                    })?
                    .iter()
                    .map(|normalizer| Self::from_json(Some(normalizer)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self::Sequence(normalizers))
            }
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported normalizer {other}"),
            }),
        }
    }

    pub(crate) fn normalize(&self, text: &str) -> String {
        match self {
            Self::None => text.to_string(),
            Self::Lowercase => text.to_lowercase(),
            Self::Nfc => text.nfc().collect(),
            Self::Nfd => text.nfd().collect(),
            Self::Nfkc => text.nfkc().collect(),
            Self::Nfkd => text.nfkd().collect(),
            Self::Strip {
                strip_left,
                strip_right,
            } => match (*strip_left, *strip_right) {
                (true, true) => text.trim().to_string(),
                (true, false) => text.trim_start().to_string(),
                (false, true) => text.trim_end().to_string(),
                (false, false) => text.to_string(),
            },
            Self::StripAccents => strip_accents(text),
            Self::Replace {
                pattern,
                content,
                regex,
            } => {
                if *regex {
                    Regex::new(pattern)
                        .map(|regex| regex.replace_all(text, content.as_str()).to_string())
                        .unwrap_or_else(|_| text.to_string())
                } else {
                    text.replace(pattern, content)
                }
            }
            Self::Bert {
                clean_text,
                handle_chinese_chars,
                strip_accents: should_strip_accents,
                lowercase,
            } => normalize_bert(
                text,
                *clean_text,
                *handle_chinese_chars,
                should_strip_accents.unwrap_or(*lowercase),
                *lowercase,
            ),
            Self::Sequence(normalizers) => normalizers
                .iter()
                .fold(text.to_string(), |text, next| next.normalize(&text)),
        }
    }
}

pub(crate) fn replace_pattern_from_json(
    value: Option<&serde_json::Value>,
) -> Result<(String, bool)> {
    let Some(value) = value else {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "normalizer Replace requires pattern".to_string(),
        });
    };
    if let Some(pattern) = value.as_str() {
        return Ok((pattern.to_string(), false));
    }
    if let Some(pattern) = value.get("String").and_then(serde_json::Value::as_str) {
        return Ok((pattern.to_string(), false));
    }
    if let Some(pattern) = value.get("Regex").and_then(serde_json::Value::as_str) {
        return Ok((pattern.to_string(), true));
    }
    Err(RuntimeError::NativeDecoderTokenizerInvalid {
        reason: "normalizer Replace pattern must be a string, String, or Regex".to_string(),
    })
}

pub(crate) fn normalize_bert(
    text: &str,
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents_flag: bool,
    lowercase: bool,
) -> String {
    let mut output = String::new();
    for ch in text.chars() {
        if clean_text && (ch == '\u{0}' || ch == '\u{fffd}' || is_bert_control(ch)) {
            continue;
        }
        let ch = if clean_text && ch.is_whitespace() {
            ' '
        } else {
            ch
        };
        if handle_chinese_chars && is_cjk_character(ch) {
            output.push(' ');
            output.push(ch);
            output.push(' ');
        } else {
            output.push(ch);
        }
    }
    if lowercase {
        output = output.to_lowercase();
    }
    if strip_accents_flag {
        output = strip_accents(&output);
    }
    output
}

pub(crate) fn strip_accents(text: &str) -> String {
    text.nfd().filter(|ch| !is_combining_mark(*ch)).collect()
}

pub(crate) fn is_bert_control(ch: char) -> bool {
    ch.is_control() && ch != '\t' && ch != '\n' && ch != '\r'
}

pub(crate) fn is_cjk_character(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}
