use super::*;

/// Minimal tokenizer contract supported by the native decoder text API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDecoderTokenizer {
    /// Tokenizer model kind from `tokenizer.json`.
    pub model_type: String,
    /// Token string to token id map.
    pub vocab: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    unk_token: Option<String>,
    mode: NativeDecoderTokenizerMode,
    normalizer: NativeDecoderNormalizer,
    pre_tokenizer: NativeDecoderPreTokenizer,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_fallback: bool,
}

impl NativeDecoderTokenizer {
    pub(crate) fn from_json(bytes: &[u8]) -> Result<Self> {
        let raw: NativeDecoderTokenizerJson = serde_json::from_slice(bytes).map_err(|error| {
            RuntimeError::NativeDecoderTokenizerInvalid {
                reason: error.to_string(),
            }
        })?;
        reject_supported_tokenizer_component("post_processor", raw.post_processor.as_ref())?;
        let normalizer = NativeDecoderNormalizer::from_json(raw.normalizer.as_ref())?;
        let pre_tokenizer = NativeDecoderPreTokenizer::from_json(raw.pre_tokenizer.as_ref())?;
        let mode = match raw.model.tokenizer_type.as_str() {
            "WordLevel" => NativeDecoderTokenizerMode::WordLevel,
            "BPE" => NativeDecoderTokenizerMode::Bpe,
            other => {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!(
                        "only WordLevel and BPE tokenizer.json assets are supported, got {other}"
                    ),
                });
            }
        };
        let bpe_ranks = if mode == NativeDecoderTokenizerMode::Bpe {
            bpe_ranks_from_merges(&raw.model.merges)?
        } else {
            HashMap::new()
        };
        let unk_token = raw.model.unk_token;
        let mut vocab = raw.model.vocab;
        for added in raw.added_tokens {
            vocab.entry(added.content).or_insert(added.id);
        }
        if vocab.is_empty() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "tokenizer vocab must not be empty".to_string(),
            });
        }
        let mut id_to_token = HashMap::with_capacity(vocab.len());
        for (token, token_id) in &vocab {
            if id_to_token.insert(*token_id, token.clone()).is_some() {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!("duplicate tokenizer id {token_id}"),
                });
            }
        }
        if let Some(unk_token) = &unk_token {
            if !vocab.contains_key(unk_token) {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!("unk_token {unk_token} is not present in vocab"),
                });
            }
        }
        Ok(Self {
            model_type: raw.model.tokenizer_type,
            vocab,
            id_to_token,
            unk_token,
            mode,
            normalizer,
            pre_tokenizer,
            bpe_ranks,
            byte_fallback: raw.model.byte_fallback,
        })
    }

    /// Encode text to token ids.
    ///
    /// WordLevel tokenizers use whitespace token lookup. BPE tokenizers support
    /// simple whitespace or ByteLevel-style pre-tokenization, vocab/merges, and
    /// exact special-token lookup. Unsupported tokenizer components fail at
    /// load time with typed errors.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let normalized = self.normalizer.normalize(text);
        let mut token_ids = Vec::new();
        for token in self.pre_tokenizer.pieces(&normalized) {
            match self.mode {
                NativeDecoderTokenizerMode::WordLevel => {
                    token_ids.push(self.lookup_token_id(&token)?);
                }
                NativeDecoderTokenizerMode::Bpe => {
                    token_ids.extend(self.encode_bpe_piece(&token)?);
                }
            }
        }
        if token_ids.is_empty() {
            return Err(RuntimeError::NativeDecoderPromptEmpty);
        }
        Ok(token_ids)
    }

    /// Decode token ids back to text.
    pub fn decode(&self, token_ids: &[i64]) -> Result<String> {
        let tokens = token_ids
            .iter()
            .map(|token_id| {
                self.id_to_token.get(token_id).cloned().ok_or_else(|| {
                    RuntimeError::NativeDecoderTokenizerTokenUnknown {
                        token: token_id.to_string(),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        match self.mode {
            NativeDecoderTokenizerMode::WordLevel => Ok(tokens.join(" ")),
            NativeDecoderTokenizerMode::Bpe => Ok(self.decode_bpe_tokens(&tokens)?),
        }
    }

    pub(crate) fn lookup_token_id(&self, token: &str) -> Result<i64> {
        if let Some(token_id) = self.vocab.get(token) {
            Ok(*token_id)
        } else if let Some(unk_token) = &self.unk_token {
            self.vocab.get(unk_token).copied().ok_or_else(|| {
                RuntimeError::NativeDecoderTokenizerTokenUnknown {
                    token: token.to_string(),
                }
            })
        } else {
            Err(RuntimeError::NativeDecoderTokenizerTokenUnknown {
                token: token.to_string(),
            })
        }
    }

    pub(crate) fn encode_bpe_piece(&self, piece: &str) -> Result<Vec<i64>> {
        if self.vocab.contains_key(piece) {
            return self.lookup_token_id(piece).map(|token_id| vec![token_id]);
        }
        let mut symbols = piece.chars().map(|c| c.to_string()).collect::<Vec<_>>();
        while symbols.len() > 1 {
            let Some((best_index, _)) = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    self.bpe_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (index, *rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let merged = format!("{}{}", symbols[best_index], symbols[best_index + 1]);
            symbols.splice(best_index..=best_index + 1, [merged]);
        }
        symbols
            .into_iter()
            .map(|symbol| self.lookup_bpe_symbol_ids(&symbol))
            .collect::<Result<Vec<_>>>()
            .map(|parts| parts.into_iter().flatten().collect())
    }

    pub(crate) fn lookup_bpe_symbol_ids(&self, symbol: &str) -> Result<Vec<i64>> {
        if let Some(token_id) = self.vocab.get(symbol) {
            return Ok(vec![*token_id]);
        }
        if self.byte_fallback {
            return symbol
                .as_bytes()
                .iter()
                .map(|byte| {
                    self.lookup_token_id(&format!("<0x{byte:02X}>"))
                        .map_err(|_| RuntimeError::NativeDecoderTokenizerTokenUnknown {
                            token: symbol.to_string(),
                        })
                })
                .collect();
        }
        self.lookup_token_id(symbol).map(|token_id| vec![token_id])
    }

    pub(crate) fn decode_bpe_tokens(&self, tokens: &[String]) -> Result<String> {
        if !self.byte_fallback {
            return Ok(decode_bpe_tokens(tokens));
        }
        let mut decoded_tokens = Vec::new();
        let mut byte_buffer = Vec::new();
        for token in tokens {
            if let Some(byte) = parse_byte_fallback_token(token) {
                byte_buffer.push(byte);
            } else {
                flush_byte_fallback_buffer(&mut byte_buffer, &mut decoded_tokens)?;
                decoded_tokens.push(token.clone());
            }
        }
        flush_byte_fallback_buffer(&mut byte_buffer, &mut decoded_tokens)?;
        Ok(decode_bpe_tokens(&decoded_tokens))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderTokenizerMode {
    WordLevel,
    Bpe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderNormalizer {
    None,
    Lowercase,
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
            Self::Sequence(normalizers) => normalizers
                .iter()
                .fold(text.to_string(), |text, next| next.normalize(&text)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderPreTokenizer {
    Whitespace,
    ByteLevel { add_prefix_space: bool },
}

impl NativeDecoderPreTokenizer {
    pub(crate) fn from_json(value: Option<&serde_json::Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::Whitespace);
        };
        if value.is_null() {
            return Ok(Self::Whitespace);
        }
        let Some(kind) = value.get("type").and_then(serde_json::Value::as_str) else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "pre_tokenizer.type is required when pre_tokenizer is present".to_string(),
            });
        };
        match kind {
            "Whitespace" | "WhitespaceSplit" => Ok(Self::Whitespace),
            "ByteLevel" => Ok(Self::ByteLevel {
                add_prefix_space: value
                    .get("add_prefix_space")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            }),
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported pre_tokenizer {other}"),
            }),
        }
    }

    pub(crate) fn pieces(self, text: &str) -> Vec<String> {
        match self {
            Self::Whitespace => text.split_whitespace().map(ToString::to_string).collect(),
            Self::ByteLevel { add_prefix_space } => {
                let mut pieces = Vec::new();
                for (index, piece) in text.split_whitespace().enumerate() {
                    if index == 0 && !add_prefix_space {
                        pieces.push(piece.to_string());
                    } else {
                        pieces.push(format!("Ġ{piece}"));
                    }
                }
                pieces
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct NativeDecoderTokenizerJson {
    model: NativeDecoderTokenizerModelJson,
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    post_processor: Option<serde_json::Value>,
    #[serde(default)]
    added_tokens: Vec<NativeDecoderAddedTokenJson>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NativeDecoderTokenizerModelJson {
    #[serde(rename = "type")]
    tokenizer_type: String,
    #[serde(default)]
    vocab: HashMap<String, i64>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    merges: Vec<NativeDecoderBpeMergeJson>,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NativeDecoderAddedTokenJson {
    id: i64,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum NativeDecoderBpeMergeJson {
    Text(String),
    Pair([String; 2]),
}

pub(crate) fn reject_supported_tokenizer_component(
    name: &str,
    value: Option<&serde_json::Value>,
) -> Result<()> {
    if value.is_some_and(|value| !value.is_null()) {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("{name} is not supported by the native tokenizer yet"),
        });
    }
    Ok(())
}

pub(crate) fn bpe_ranks_from_merges(
    merges: &[NativeDecoderBpeMergeJson],
) -> Result<HashMap<(String, String), usize>> {
    let mut ranks = HashMap::with_capacity(merges.len());
    for (rank, merge) in merges.iter().enumerate() {
        let (left, right) =
            match merge {
                NativeDecoderBpeMergeJson::Text(value) => {
                    let mut parts = value.split_whitespace();
                    let left = parts.next().ok_or_else(|| {
                        RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("invalid BPE merge {value:?}"),
                        }
                    })?;
                    let right = parts.next().ok_or_else(|| {
                        RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("invalid BPE merge {value:?}"),
                        }
                    })?;
                    if parts.next().is_some() {
                        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("invalid BPE merge {value:?}"),
                        });
                    }
                    (left.to_string(), right.to_string())
                }
                NativeDecoderBpeMergeJson::Pair([left, right]) => (left.clone(), right.clone()),
            };
        ranks.insert((left, right), rank);
    }
    Ok(ranks)
}

pub(crate) fn decode_bpe_tokens(tokens: &[String]) -> String {
    let mut text = String::new();
    for token in tokens {
        if let Some(rest) = token.strip_prefix('Ġ') {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(rest);
        } else {
            text.push_str(token);
        }
    }
    text
}

pub(crate) fn parse_byte_fallback_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

pub(crate) fn flush_byte_fallback_buffer(
    byte_buffer: &mut Vec<u8>,
    decoded_tokens: &mut Vec<String>,
) -> Result<()> {
    if byte_buffer.is_empty() {
        return Ok(());
    }
    let text = String::from_utf8(std::mem::take(byte_buffer)).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("byte_fallback emitted invalid UTF-8: {error}"),
        }
    })?;
    decoded_tokens.push(text);
    Ok(())
}
