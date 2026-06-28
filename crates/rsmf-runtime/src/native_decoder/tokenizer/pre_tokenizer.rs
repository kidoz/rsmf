use super::replace_pattern_from_json;
use crate::{Result, RuntimeError};
use regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderPreTokenizer {
    Whitespace,
    WhitespaceSplit,
    ByteLevel {
        add_prefix_space: bool,
        use_regex: bool,
    },
    Metaspace {
        replacement: char,
        add_prefix_space: bool,
    },
    Punctuation,
    Digits {
        individual_digits: bool,
    },
    CharDelimiterSplit(char),
    Split {
        pattern: String,
        invert: bool,
    },
    Sequence(Vec<NativeDecoderPreTokenizer>),
}

impl NativeDecoderPreTokenizer {
    pub(crate) fn from_json(value: Option<&serde_json::Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::WhitespaceSplit);
        };
        if value.is_null() {
            return Ok(Self::WhitespaceSplit);
        }
        let Some(kind) = value.get("type").and_then(serde_json::Value::as_str) else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "pre_tokenizer.type is required when pre_tokenizer is present".to_string(),
            });
        };
        match kind {
            "Whitespace" => Ok(Self::Whitespace),
            "WhitespaceSplit" => Ok(Self::WhitespaceSplit),
            "ByteLevel" => Ok(Self::ByteLevel {
                add_prefix_space: value
                    .get("add_prefix_space")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
                use_regex: value
                    .get("use_regex")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
            }),
            "Metaspace" => {
                let replacement = value
                    .get("replacement")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("▁");
                let mut chars = replacement.chars();
                let Some(replacement) = chars.next() else {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "pre_tokenizer Metaspace replacement must not be empty".to_string(),
                    });
                };
                if chars.next().is_some() {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "pre_tokenizer Metaspace replacement must be one character"
                            .to_string(),
                    });
                }
                let add_prefix_space = if let Some(add_prefix_space) = value
                    .get("add_prefix_space")
                    .and_then(serde_json::Value::as_bool)
                {
                    add_prefix_space
                } else {
                    match value
                        .get("prepend_scheme")
                        .and_then(serde_json::Value::as_str)
                    {
                        Some("always") => true,
                        Some("never") => false,
                        _ => true,
                    }
                };
                Ok(Self::Metaspace {
                    replacement,
                    add_prefix_space,
                })
            }
            "Punctuation" => Ok(Self::Punctuation),
            "Digits" => Ok(Self::Digits {
                individual_digits: value
                    .get("individual_digits")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            }),
            "CharDelimiterSplit" => {
                let delimiter = value
                    .get("delimiter")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "pre_tokenizer CharDelimiterSplit requires delimiter".to_string(),
                    })?;
                let mut chars = delimiter.chars();
                let Some(delimiter) = chars.next() else {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "pre_tokenizer CharDelimiterSplit delimiter must not be empty"
                            .to_string(),
                    });
                };
                if chars.next().is_some() {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "pre_tokenizer CharDelimiterSplit delimiter must be one character"
                            .to_string(),
                    });
                }
                Ok(Self::CharDelimiterSplit(delimiter))
            }
            "Split" => {
                let (pattern, regex) = replace_pattern_from_json(value.get("pattern"))?;
                if regex {
                    Regex::new(&pattern).map_err(|error| {
                        RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("pre_tokenizer Split regex is unsupported: {error}"),
                        }
                    })?;
                }
                Ok(Self::Split {
                    pattern,
                    invert: value
                        .get("invert")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                })
            }
            "Sequence" => {
                let pre_tokenizers = value
                    .get("pretokenizers")
                    .or_else(|| value.get("pre_tokenizers"))
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "pre_tokenizer Sequence requires a pretokenizers array".to_string(),
                    })?
                    .iter()
                    .map(|pre_tokenizer| Self::from_json(Some(pre_tokenizer)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self::Sequence(pre_tokenizers))
            }
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported pre_tokenizer {other}"),
            }),
        }
    }

    pub(crate) fn pieces(&self, text: &str) -> Vec<String> {
        match self {
            Self::Whitespace => split_whitespace_and_punctuation(text),
            Self::WhitespaceSplit => text.split_whitespace().map(ToString::to_string).collect(),
            Self::ByteLevel {
                add_prefix_space, ..
            } => {
                let mut pieces = Vec::new();
                for (index, piece) in text.split_whitespace().enumerate() {
                    if index == 0 && !*add_prefix_space {
                        pieces.push(piece.to_string());
                    } else {
                        pieces.push(format!("Ġ{piece}"));
                    }
                }
                pieces
            }
            Self::Metaspace {
                replacement,
                add_prefix_space,
            } => {
                let mut piece = text
                    .chars()
                    .map(|ch| if ch.is_whitespace() { *replacement } else { ch })
                    .collect::<String>();
                if *add_prefix_space && !piece.starts_with(*replacement) {
                    piece.insert(0, *replacement);
                }
                vec![piece]
            }
            Self::Punctuation => split_punctuation(text),
            Self::Digits { individual_digits } => split_digits(text, *individual_digits),
            Self::CharDelimiterSplit(delimiter) => text
                .split(*delimiter)
                .filter(|piece| !piece.is_empty())
                .map(ToString::to_string)
                .collect(),
            Self::Split { pattern, invert } => split_by_pattern(text, pattern, *invert),
            Self::Sequence(pre_tokenizers) => {
                let mut pieces = vec![text.to_string()];
                for pre_tokenizer in pre_tokenizers {
                    pieces = pieces
                        .into_iter()
                        .flat_map(|piece| pre_tokenizer.pieces(&piece))
                        .collect();
                }
                pieces
            }
        }
    }
}

pub(crate) fn split_whitespace_and_punctuation(text: &str) -> Vec<String> {
    text.split_whitespace()
        .flat_map(split_punctuation)
        .collect()
}

pub(crate) fn split_punctuation(text: &str) -> Vec<String> {
    split_by_char_class(text, is_tokenizer_punctuation)
}

pub(crate) fn split_digits(text: &str, individual_digits: bool) -> Vec<String> {
    if individual_digits {
        split_by_char_class(text, |ch| ch.is_ascii_digit())
    } else {
        split_runs_by_class(text, |ch| ch.is_ascii_digit())
    }
}

pub(crate) fn split_by_pattern(text: &str, pattern: &str, invert: bool) -> Vec<String> {
    let Ok(regex) = Regex::new(pattern) else {
        if pattern.is_empty() {
            return vec![text.to_string()];
        }
        return text
            .split(pattern)
            .filter(|piece| !piece.is_empty())
            .map(ToString::to_string)
            .collect();
    };
    if invert {
        regex
            .find_iter(text)
            .map(|matched| matched.as_str().to_string())
            .collect()
    } else {
        regex
            .split(text)
            .filter(|piece| !piece.is_empty())
            .map(ToString::to_string)
            .collect()
    }
}

pub(crate) fn split_by_char_class(
    text: &str,
    mut is_boundary_token: impl FnMut(char) -> bool,
) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut buffer = String::new();
    for ch in text.chars() {
        if is_boundary_token(ch) {
            if !buffer.is_empty() {
                pieces.push(std::mem::take(&mut buffer));
            }
            pieces.push(ch.to_string());
        } else {
            buffer.push(ch);
        }
    }
    if !buffer.is_empty() {
        pieces.push(buffer);
    }
    pieces
}

pub(crate) fn split_runs_by_class(
    text: &str,
    mut classify: impl FnMut(char) -> bool,
) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut buffer = String::new();
    let mut active_class = None;
    for ch in text.chars() {
        let class = classify(ch);
        if active_class.is_some_and(|active| active != class) && !buffer.is_empty() {
            pieces.push(std::mem::take(&mut buffer));
        }
        active_class = Some(class);
        buffer.push(ch);
    }
    if !buffer.is_empty() {
        pieces.push(buffer);
    }
    pieces
}

pub(crate) fn is_tokenizer_punctuation(ch: char) -> bool {
    let code = ch as u32;
    (33..=47).contains(&code)
        || (58..=64).contains(&code)
        || (91..=96).contains(&code)
        || (123..=126).contains(&code)
        || ch.is_ascii_punctuation()
}
