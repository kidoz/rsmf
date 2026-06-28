use super::NativeDecoderEncoding;
use crate::{Result, RuntimeError};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderPostProcessor {
    None,
    TemplateProcessing {
        single: Vec<NativeDecoderTemplatePiece>,
        pair: Vec<NativeDecoderTemplatePiece>,
    },
    BertProcessing {
        sep: NativeDecoderTemplatePiece,
        cls: NativeDecoderTemplatePiece,
    },
    RobertaProcessing {
        sep: NativeDecoderTemplatePiece,
        cls: NativeDecoderTemplatePiece,
    },
    ByteLevel,
    Sequence(Vec<NativeDecoderPostProcessor>),
}

impl NativeDecoderPostProcessor {
    pub(crate) fn from_json(value: Option<&serde_json::Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::None);
        };
        if value.is_null() {
            return Ok(Self::None);
        }
        let Some(kind) = value.get("type").and_then(serde_json::Value::as_str) else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "post_processor.type is required when post_processor is present"
                    .to_string(),
            });
        };
        match kind {
            "ByteLevel" => Ok(Self::ByteLevel),
            "Sequence" => {
                let processors = value
                    .get("processors")
                    .or_else(|| value.get("post_processors"))
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "post_processor Sequence requires a processors array".to_string(),
                    })?
                    .iter()
                    .map(|processor| Self::from_json(Some(processor)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self::Sequence(processors))
            }
            "BertProcessing" => Ok(Self::BertProcessing {
                sep: post_processor_special_pair(value, "sep")?,
                cls: post_processor_special_pair(value, "cls")?,
            }),
            "RobertaProcessing" => Ok(Self::RobertaProcessing {
                sep: post_processor_special_pair(value, "sep")?,
                cls: post_processor_special_pair(value, "cls")?,
            }),
            "TemplateProcessing" => {
                let special_tokens =
                    template_special_tokens_from_json(value.get("special_tokens"))?;
                Ok(Self::TemplateProcessing {
                    single: template_pieces_from_json(
                        value.get("single"),
                        "single",
                        &special_tokens,
                    )?,
                    pair: template_pieces_from_json(value.get("pair"), "pair", &special_tokens)?,
                })
            }
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported post_processor {other}"),
            }),
        }
    }

    pub(crate) fn apply_single(
        &self,
        encoding: NativeDecoderEncoding,
    ) -> Result<NativeDecoderEncoding> {
        match self {
            Self::None | Self::ByteLevel => Ok(encoding),
            Self::TemplateProcessing { single, .. } => {
                apply_template_pieces(single, &encoding, &NativeDecoderEncoding::new())
            }
            Self::BertProcessing { sep, cls } => apply_template_pieces(
                &[
                    cls.clone(),
                    NativeDecoderTemplatePiece::SequenceA,
                    sep.clone(),
                ],
                &encoding,
                &NativeDecoderEncoding::new(),
            ),
            Self::RobertaProcessing { sep, cls } => apply_template_pieces(
                &[
                    cls.clone(),
                    NativeDecoderTemplatePiece::SequenceA,
                    sep.clone(),
                ],
                &encoding,
                &NativeDecoderEncoding::new(),
            ),
            Self::Sequence(processors) => {
                processors.iter().try_fold(encoding, |encoding, processor| {
                    processor.apply_single(encoding)
                })
            }
        }
    }

    pub(crate) fn apply_pair(
        &self,
        first: NativeDecoderEncoding,
        second: NativeDecoderEncoding,
    ) -> Result<NativeDecoderEncoding> {
        match self {
            Self::None | Self::ByteLevel => {
                let mut output = first;
                output.extend(&second);
                Ok(output)
            }
            Self::TemplateProcessing { pair, .. } => apply_template_pieces(pair, &first, &second),
            Self::BertProcessing { sep, cls } => apply_template_pieces(
                &[
                    cls.clone(),
                    NativeDecoderTemplatePiece::SequenceA,
                    sep.clone(),
                    NativeDecoderTemplatePiece::SequenceB,
                    sep.clone(),
                ],
                &first,
                &second,
            ),
            Self::RobertaProcessing { sep, cls } => apply_template_pieces(
                &[
                    cls.clone(),
                    NativeDecoderTemplatePiece::SequenceA,
                    sep.clone(),
                    sep.clone(),
                    NativeDecoderTemplatePiece::SequenceB,
                    sep.clone(),
                ],
                &first,
                &second,
            ),
            Self::Sequence(processors) => {
                let Some((first_processor, rest)) = processors.split_first() else {
                    let mut output = first;
                    output.extend(&second);
                    return Ok(output);
                };
                let mut output = first_processor.apply_pair(first, second)?;
                for processor in rest {
                    output = processor.apply_single(output)?;
                }
                Ok(output)
            }
        }
    }

    pub(crate) fn special_token_strings(&self) -> Vec<String> {
        match self {
            Self::None => Vec::new(),
            Self::TemplateProcessing { single, pair } => single
                .iter()
                .chain(pair.iter())
                .filter_map(|piece| match piece {
                    NativeDecoderTemplatePiece::SpecialToken { token, .. } => Some(token.clone()),
                    NativeDecoderTemplatePiece::SequenceA
                    | NativeDecoderTemplatePiece::SequenceB => None,
                })
                .collect(),
            Self::BertProcessing { sep, cls } | Self::RobertaProcessing { sep, cls } => [sep, cls]
                .into_iter()
                .filter_map(|piece| match piece {
                    NativeDecoderTemplatePiece::SpecialToken { token, .. } => Some(token.clone()),
                    NativeDecoderTemplatePiece::SequenceA
                    | NativeDecoderTemplatePiece::SequenceB => None,
                })
                .collect(),
            Self::ByteLevel => Vec::new(),
            Self::Sequence(processors) => processors
                .iter()
                .flat_map(Self::special_token_strings)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderTemplatePiece {
    SequenceA,
    SequenceB,
    SpecialToken { token: String, ids: Vec<i64> },
}

pub(crate) fn template_pieces_from_json(
    value: Option<&serde_json::Value>,
    field_name: &str,
    special_tokens: &HashMap<String, Vec<i64>>,
) -> Result<Vec<NativeDecoderTemplatePiece>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Some(items) = value.as_array() else {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("TemplateProcessing.{field_name} must be an array"),
        });
    };
    items
        .iter()
        .map(|item| template_piece_from_json(item, special_tokens))
        .collect::<Result<Vec<_>>>()
}

pub(crate) fn template_special_tokens_from_json(
    value: Option<&serde_json::Value>,
) -> Result<HashMap<String, Vec<i64>>> {
    let Some(value) = value else {
        return Ok(HashMap::new());
    };
    let Some(entries) = value.as_object() else {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "TemplateProcessing.special_tokens must be an object".to_string(),
        });
    };
    let mut special_tokens = HashMap::with_capacity(entries.len());
    for (token, entry) in entries {
        let ids = entry
            .get("ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("TemplateProcessing special token {token} requires ids"),
            })?
            .iter()
            .map(|id| {
                id.as_i64()
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: format!(
                            "TemplateProcessing special token {token} ids must be integers"
                        ),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if ids.is_empty() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("TemplateProcessing special token {token} ids must not be empty"),
            });
        }
        special_tokens.insert(token.clone(), ids);
    }
    Ok(special_tokens)
}

pub(crate) fn template_piece_from_json(
    value: &serde_json::Value,
    special_tokens: &HashMap<String, Vec<i64>>,
) -> Result<NativeDecoderTemplatePiece> {
    if let Some(piece) = value.as_str() {
        return template_piece_from_string(piece, special_tokens);
    }
    if let Some(sequence) = value.get("Sequence") {
        let id = sequence
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("A");
        return match id {
            "A" => Ok(NativeDecoderTemplatePiece::SequenceA),
            "B" => Ok(NativeDecoderTemplatePiece::SequenceB),
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported TemplateProcessing sequence id {other}"),
            }),
        };
    }
    if let Some(special) = value.get("SpecialToken") {
        let token = special
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "TemplateProcessing SpecialToken requires string id".to_string(),
            })?
            .to_string();
        let ids = special
            .get("ids")
            .and_then(serde_json::Value::as_array)
            .map(|ids| {
                ids.iter()
                    .map(|id| {
                        id.as_i64()
                            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                                reason: format!(
                                    "TemplateProcessing SpecialToken {token} ids must be integers"
                                ),
                            })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .or_else(|| special_tokens.get(&token).cloned())
            .unwrap_or_default();
        if ids.is_empty() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("TemplateProcessing SpecialToken {token} requires ids"),
            });
        }
        return Ok(NativeDecoderTemplatePiece::SpecialToken { token, ids });
    }
    Err(RuntimeError::NativeDecoderTokenizerInvalid {
        reason: "unsupported TemplateProcessing piece".to_string(),
    })
}

pub(crate) fn template_piece_from_string(
    value: &str,
    special_tokens: &HashMap<String, Vec<i64>>,
) -> Result<NativeDecoderTemplatePiece> {
    let piece = value.trim();
    if piece.starts_with("$A") || piece.starts_with("$0") {
        return Ok(NativeDecoderTemplatePiece::SequenceA);
    }
    if piece.starts_with("$B") || piece.starts_with("$1") {
        return Ok(NativeDecoderTemplatePiece::SequenceB);
    }
    let token = piece
        .split_once(':')
        .map_or(piece, |(token, _)| token)
        .trim()
        .to_string();
    let ids = special_tokens.get(&token).cloned().unwrap_or_default();
    if ids.is_empty() {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("TemplateProcessing string piece {value:?} requires special token ids"),
        });
    }
    Ok(NativeDecoderTemplatePiece::SpecialToken { token, ids })
}

pub(crate) fn post_processor_special_pair(
    value: &serde_json::Value,
    field_name: &str,
) -> Result<NativeDecoderTemplatePiece> {
    let pair = value
        .get(field_name)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("post_processor {field_name} must be [token, id]"),
        })?;
    if pair.len() != 2 {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("post_processor {field_name} must be [token, id]"),
        });
    }
    let token = pair[0]
        .as_str()
        .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("post_processor {field_name} token must be a string"),
        })?
        .to_string();
    let id = pair[1]
        .as_i64()
        .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("post_processor {field_name} id must be an integer"),
        })?;
    Ok(NativeDecoderTemplatePiece::SpecialToken {
        token,
        ids: vec![id],
    })
}

pub(crate) fn apply_template_pieces(
    pieces: &[NativeDecoderTemplatePiece],
    first: &NativeDecoderEncoding,
    second: &NativeDecoderEncoding,
) -> Result<NativeDecoderEncoding> {
    if pieces.is_empty() {
        let mut output = first.clone();
        output.extend(second);
        return Ok(output);
    }
    let mut output = NativeDecoderEncoding::new();
    for piece in pieces {
        match piece {
            NativeDecoderTemplatePiece::SequenceA => output.extend(first),
            NativeDecoderTemplatePiece::SequenceB => output.extend(second),
            NativeDecoderTemplatePiece::SpecialToken { token, ids } => {
                for token_id in ids {
                    output.push(*token_id, token.clone(), None, 0, true, None);
                }
            }
        }
    }
    Ok(output)
}
