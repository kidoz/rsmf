use super::*;
use minijinja::{Environment, ErrorKind as JinjaErrorKind, value::Value as JinjaValue};
use regex::Regex;
use serde_json::json;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

const MAX_TOKENIZER_JSON_BYTES: usize = 64 * 1024 * 1024;
const MAX_TOKENIZER_VOCAB_ENTRIES: usize = 2_000_000;
const MAX_TOKENIZER_TOKEN_CHARS: usize = 16_384;
const MAX_CHAT_TEMPLATE_BYTES: usize = 1024 * 1024;
const MAX_CHAT_RENDERED_BYTES: usize = 8 * 1024 * 1024;
const CHAT_TEMPLATE_FUEL: u64 = 500_000;

/// Role/content message consumed by native decoder chat templates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeDecoderChatMessage {
    /// Chat role, for example `system`, `user`, or `assistant`.
    pub role: String,
    /// Message content.
    pub content: String,
}

impl NativeDecoderChatMessage {
    /// Build a chat message from a role and content string.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

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
    post_processor: NativeDecoderPostProcessor,
    bpe_ranks: HashMap<(String, String), usize>,
    unigram_scores: HashMap<String, i64>,
    wordpiece_prefix: String,
    max_input_chars_per_word: usize,
    byte_fallback: bool,
    decoder: NativeDecoderDecoder,
    special_tokens: Vec<String>,
    chat_template: Option<NativeDecoderChatTemplate>,
}

impl NativeDecoderTokenizer {
    /// Parse a supported Hugging Face `tokenizer.json` payload.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        Self::from_json_with_assets(bytes, None, None)
    }

    pub(crate) fn from_json_with_assets(
        bytes: &[u8],
        tokenizer_config: Option<&[u8]>,
        chat_template_asset: Option<&[u8]>,
    ) -> Result<Self> {
        if bytes.len() > MAX_TOKENIZER_JSON_BYTES {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!(
                    "tokenizer.json is too large: {} bytes exceeds {}",
                    bytes.len(),
                    MAX_TOKENIZER_JSON_BYTES
                ),
            });
        }
        let raw: NativeDecoderTokenizerJson = serde_json::from_slice(bytes).map_err(|error| {
            RuntimeError::NativeDecoderTokenizerInvalid {
                reason: error.to_string(),
            }
        })?;
        let normalizer = NativeDecoderNormalizer::from_json(raw.normalizer.as_ref())?;
        let pre_tokenizer = NativeDecoderPreTokenizer::from_json(raw.pre_tokenizer.as_ref())?;
        let post_processor = NativeDecoderPostProcessor::from_json(raw.post_processor.as_ref())?;
        let mode = match raw.model.tokenizer_type.as_str() {
            "WordLevel" => NativeDecoderTokenizerMode::WordLevel,
            "BPE" => NativeDecoderTokenizerMode::Bpe,
            "Unigram" => NativeDecoderTokenizerMode::Unigram,
            "WordPiece" => NativeDecoderTokenizerMode::WordPiece,
            other => {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!(
                        "only WordLevel, BPE, Unigram, and WordPiece tokenizer.json assets are supported, got {other}"
                    ),
                });
            }
        };
        let bpe_ranks = if mode == NativeDecoderTokenizerMode::Bpe {
            bpe_ranks_from_merges(&raw.model.merges)?
        } else {
            HashMap::new()
        };
        let (mut vocab, unigram_scores) = raw.model.vocab.to_vocab_and_scores(mode)?;
        let mut unk_token = raw.model.unk_token;
        if unk_token.is_none() {
            if let Some(unk_id) = raw.model.unk_id {
                let token_id = i64::try_from(unk_id).map_err(|_| {
                    RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: format!("unk_id {unk_id} cannot convert to i64"),
                    }
                })?;
                unk_token = vocab
                    .iter()
                    .find_map(|(token, id)| (*id == token_id).then(|| token.clone()));
            }
        }
        let decoder = NativeDecoderDecoder::from_json(raw.decoder.as_ref())?;
        let mut special_tokens = Vec::new();
        for added in raw.added_tokens {
            validate_tokenizer_token(&added.content)?;
            if added.special {
                special_tokens.push(added.content.clone());
            }
            vocab.entry(added.content).or_insert(added.id);
        }
        special_tokens.extend(post_processor.special_token_strings());
        special_tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        special_tokens.dedup();
        if vocab.is_empty() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "tokenizer vocab must not be empty".to_string(),
            });
        }
        if vocab.len() > MAX_TOKENIZER_VOCAB_ENTRIES {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!(
                    "tokenizer vocab has {} entries, maximum supported is {}",
                    vocab.len(),
                    MAX_TOKENIZER_VOCAB_ENTRIES
                ),
            });
        }
        let mut id_to_token = HashMap::with_capacity(vocab.len());
        for (token, token_id) in &vocab {
            validate_tokenizer_token(token)?;
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
            post_processor,
            bpe_ranks,
            unigram_scores,
            wordpiece_prefix: raw
                .model
                .continuing_subword_prefix
                .unwrap_or_else(|| "##".to_string()),
            max_input_chars_per_word: raw.model.max_input_chars_per_word.unwrap_or(100),
            byte_fallback: raw.model.byte_fallback,
            decoder,
            special_tokens,
            chat_template: NativeDecoderChatTemplate::from_assets(
                tokenizer_config,
                chat_template_asset,
            )?,
        })
    }

    /// Encode text to token ids.
    ///
    /// WordLevel tokenizers use whitespace token lookup. BPE tokenizers support
    /// simple whitespace or ByteLevel-style pre-tokenization, vocab/merges, and
    /// exact special-token lookup. Unsupported tokenizer components fail at
    /// load time with typed errors.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        Ok(self.encode_to_encoding(text)?.ids)
    }

    /// Encode a pair of text sequences and apply a pair post-processor when
    /// the tokenizer defines one.
    pub fn encode_pair(&self, first: &str, second: &str) -> Result<Vec<i64>> {
        let first = self.encode_without_post_processor(first, Some(0))?;
        let second = self.encode_without_post_processor(second, Some(1))?;
        Ok(self.post_processor.apply_pair(first, second)?.ids)
    }

    pub(crate) fn encode_to_encoding(&self, text: &str) -> Result<NativeDecoderEncoding> {
        let model_encoding = self.encode_without_post_processor(text, Some(0))?;
        self.post_processor.apply_single(model_encoding)
    }

    /// Render the configured chat template to prompt text.
    pub fn apply_chat_template(
        &self,
        messages: &[NativeDecoderChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let Some(template) = &self.chat_template else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "tokenizer has no supported chat template".to_string(),
            });
        };
        template.render(messages, add_generation_prompt)
    }

    /// Render and encode chat messages with the configured chat template.
    pub fn encode_chat(
        &self,
        messages: &[NativeDecoderChatMessage],
        add_generation_prompt: bool,
    ) -> Result<Vec<i64>> {
        let prompt = self.apply_chat_template(messages, add_generation_prompt)?;
        self.encode(&prompt)
    }

    fn encode_without_post_processor(
        &self,
        text: &str,
        sequence_id: Option<u8>,
    ) -> Result<NativeDecoderEncoding> {
        let normalized = self.normalizer.normalize(text);
        let mut encoding = NativeDecoderEncoding::new();
        for segment in isolate_special_token_segments(&normalized, &self.special_tokens) {
            match segment {
                NativeDecoderTokenSegment::Special(token) => {
                    encoding.push(
                        self.lookup_token_id(&token)?,
                        token,
                        None,
                        0,
                        true,
                        sequence_id,
                    );
                }
                NativeDecoderTokenSegment::Text(text) => {
                    for token in self.pre_tokenizer.pieces(&text) {
                        match self.mode {
                            NativeDecoderTokenizerMode::WordLevel => {
                                encoding.push(
                                    self.lookup_token_id(&token)?,
                                    token,
                                    None,
                                    0,
                                    false,
                                    sequence_id,
                                );
                            }
                            NativeDecoderTokenizerMode::Bpe => {
                                self.extend_encoding_from_ids(
                                    &mut encoding,
                                    self.encode_bpe_piece(&token)?,
                                    sequence_id,
                                )?;
                            }
                            NativeDecoderTokenizerMode::Unigram => {
                                self.extend_encoding_from_ids(
                                    &mut encoding,
                                    self.encode_unigram_piece(&token)?,
                                    sequence_id,
                                )?;
                            }
                            NativeDecoderTokenizerMode::WordPiece => {
                                self.extend_encoding_from_ids(
                                    &mut encoding,
                                    self.encode_wordpiece_piece(&token)?,
                                    sequence_id,
                                )?;
                            }
                        }
                    }
                }
            }
        }
        if encoding.ids.is_empty() {
            return Err(RuntimeError::NativeDecoderPromptEmpty);
        }
        Ok(encoding)
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
            _ if self.decoder != NativeDecoderDecoder::None => {
                self.decoder.decode(&tokens, &self.special_tokens)
            }
            NativeDecoderTokenizerMode::WordLevel => Ok(tokens.join(" ")),
            NativeDecoderTokenizerMode::Bpe => Ok(self.decode_bpe_tokens(&tokens)?),
            NativeDecoderTokenizerMode::Unigram => Ok(decode_sentencepiece_tokens(&tokens, '▁')),
            NativeDecoderTokenizerMode::WordPiece => Ok(decode_wordpiece_tokens(
                &tokens,
                &self.wordpiece_prefix,
                true,
            )),
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

    pub(crate) fn encode_unigram_piece(&self, piece: &str) -> Result<Vec<i64>> {
        if self.vocab.contains_key(piece) {
            return self.lookup_token_id(piece).map(|token_id| vec![token_id]);
        }
        let mut boundaries = piece.char_indices().map(|(idx, _)| idx).collect::<Vec<_>>();
        if boundaries.first().copied() != Some(0) {
            boundaries.insert(0, 0);
        }
        if boundaries.last().copied() != Some(piece.len()) {
            boundaries.push(piece.len());
        }
        let mut boundary_index = HashMap::with_capacity(boundaries.len());
        for (index, boundary) in boundaries.iter().enumerate() {
            boundary_index.insert(*boundary, index);
        }
        let mut dp: Vec<Option<NativeDecoderUnigramPath>> = vec![None; boundaries.len()];
        dp[0] = Some(NativeDecoderUnigramPath {
            score: 0,
            previous: 0,
            token: String::new(),
        });
        for start_index in 0..boundaries.len().saturating_sub(1) {
            let Some(start_path) = dp[start_index].clone() else {
                continue;
            };
            let start = boundaries[start_index];
            let mut matched = false;
            for end_index in start_index + 1..boundaries.len() {
                let end = boundaries[end_index];
                let candidate = &piece[start..end];
                let Some(score) = self.unigram_scores.get(candidate).copied() else {
                    continue;
                };
                matched = true;
                update_unigram_path(
                    &mut dp[end_index],
                    NativeDecoderUnigramPath {
                        score: start_path.score + score,
                        previous: start_index,
                        token: candidate.to_string(),
                    },
                );
            }
            if !matched {
                if let Some(unk_token) = &self.unk_token {
                    let end = boundaries[start_index + 1];
                    let end_index = *boundary_index.get(&end).ok_or_else(|| {
                        RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: "internal Unigram boundary lookup failed".to_string(),
                        }
                    })?;
                    update_unigram_path(
                        &mut dp[end_index],
                        NativeDecoderUnigramPath {
                            score: start_path.score - 1_000_000_000,
                            previous: start_index,
                            token: unk_token.clone(),
                        },
                    );
                }
            }
        }
        let Some(mut cursor) = dp.len().checked_sub(1) else {
            return Err(RuntimeError::NativeDecoderPromptEmpty);
        };
        if dp[cursor].is_none() {
            return Err(RuntimeError::NativeDecoderTokenizerTokenUnknown {
                token: piece.to_string(),
            });
        }
        let mut tokens = Vec::new();
        while cursor > 0 {
            let Some(path) = &dp[cursor] else {
                return Err(RuntimeError::NativeDecoderTokenizerTokenUnknown {
                    token: piece.to_string(),
                });
            };
            tokens.push(path.token.clone());
            cursor = path.previous;
        }
        tokens.reverse();
        tokens
            .into_iter()
            .map(|token| self.lookup_token_id(&token))
            .collect()
    }

    pub(crate) fn encode_wordpiece_piece(&self, piece: &str) -> Result<Vec<i64>> {
        if piece.chars().count() > self.max_input_chars_per_word {
            return self.lookup_unk_or_unknown(piece);
        }
        let boundaries = char_boundaries_with_end(piece);
        let mut start_index = 0usize;
        let mut output = Vec::new();
        while start_index + 1 < boundaries.len() {
            let mut end_index = boundaries.len() - 1;
            let mut matched: Option<(usize, i64)> = None;
            while end_index > start_index {
                let start = boundaries[start_index];
                let end = boundaries[end_index];
                let candidate = if start_index == 0 {
                    piece[start..end].to_string()
                } else {
                    format!("{}{}", self.wordpiece_prefix, &piece[start..end])
                };
                if let Some(token_id) = self.vocab.get(&candidate) {
                    matched = Some((end_index, *token_id));
                    break;
                }
                end_index -= 1;
            }
            let Some((next_start, token_id)) = matched else {
                return self.lookup_unk_or_unknown(piece);
            };
            output.push(token_id);
            start_index = next_start;
        }
        Ok(output)
    }

    pub(crate) fn lookup_unk_or_unknown(&self, piece: &str) -> Result<Vec<i64>> {
        let Some(unk_token) = &self.unk_token else {
            return Err(RuntimeError::NativeDecoderTokenizerTokenUnknown {
                token: piece.to_string(),
            });
        };
        self.lookup_token_id(unk_token)
            .map(|token_id| vec![token_id])
    }

    pub(crate) fn extend_encoding_from_ids(
        &self,
        encoding: &mut NativeDecoderEncoding,
        token_ids: Vec<i64>,
        sequence_id: Option<u8>,
    ) -> Result<()> {
        for token_id in token_ids {
            let token = self.id_to_token.get(&token_id).cloned().ok_or_else(|| {
                RuntimeError::NativeDecoderTokenizerTokenUnknown {
                    token: token_id.to_string(),
                }
            })?;
            encoding.push(token_id, token, None, 0, false, sequence_id);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderTokenizerMode {
    WordLevel,
    Bpe,
    Unigram,
    WordPiece,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeDecoderEncoding {
    ids: Vec<i64>,
    tokens: Vec<String>,
    offsets: Vec<Option<(usize, usize)>>,
    type_ids: Vec<u32>,
    special_tokens_mask: Vec<bool>,
    sequence_ids: Vec<Option<u8>>,
}

impl NativeDecoderEncoding {
    pub(crate) fn new() -> Self {
        Self {
            ids: Vec::new(),
            tokens: Vec::new(),
            offsets: Vec::new(),
            type_ids: Vec::new(),
            special_tokens_mask: Vec::new(),
            sequence_ids: Vec::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        id: i64,
        token: String,
        offset: Option<(usize, usize)>,
        type_id: u32,
        special: bool,
        sequence_id: Option<u8>,
    ) {
        self.ids.push(id);
        self.tokens.push(token);
        self.offsets.push(offset);
        self.type_ids.push(type_id);
        self.special_tokens_mask.push(special);
        self.sequence_ids.push(sequence_id);
    }

    pub(crate) fn extend(&mut self, other: &Self) {
        self.ids.extend_from_slice(&other.ids);
        self.tokens.extend_from_slice(&other.tokens);
        self.offsets.extend_from_slice(&other.offsets);
        self.type_ids.extend_from_slice(&other.type_ids);
        self.special_tokens_mask
            .extend_from_slice(&other.special_tokens_mask);
        self.sequence_ids.extend_from_slice(&other.sequence_ids);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeDecoderUnigramPath {
    score: i64,
    previous: usize,
    token: String,
}

pub(crate) fn update_unigram_path(
    slot: &mut Option<NativeDecoderUnigramPath>,
    candidate: NativeDecoderUnigramPath,
) {
    let replace = slot.as_ref().is_none_or(|current| {
        candidate.score > current.score
            || (candidate.score == current.score && candidate.token.len() > current.token.len())
    });
    if replace {
        *slot = Some(candidate);
    }
}

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
    decoder: Option<serde_json::Value>,
    #[serde(default)]
    added_tokens: Vec<NativeDecoderAddedTokenJson>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NativeDecoderTokenizerModelJson {
    #[serde(rename = "type")]
    tokenizer_type: String,
    #[serde(default)]
    vocab: NativeDecoderTokenizerVocabJson,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    unk_id: Option<usize>,
    #[serde(default)]
    merges: Vec<NativeDecoderBpeMergeJson>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    max_input_chars_per_word: Option<usize>,
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
    id: i64,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum NativeDecoderBpeMergeJson {
    Text(String),
    Pair([String; 2]),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderTokenSegment {
    Text(String),
    Special(String),
}

pub(crate) fn isolate_special_token_segments(
    text: &str,
    special_tokens: &[String],
) -> Vec<NativeDecoderTokenSegment> {
    if special_tokens.is_empty() {
        return vec![NativeDecoderTokenSegment::Text(text.to_string())];
    }
    let mut segments = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let next = special_tokens
            .iter()
            .filter_map(|token| rest.find(token).map(|index| (index, token)))
            .min_by_key(|(index, token)| (*index, std::cmp::Reverse(token.len())));
        let Some((index, token)) = next else {
            segments.push(NativeDecoderTokenSegment::Text(rest.to_string()));
            break;
        };
        if index > 0 {
            segments.push(NativeDecoderTokenSegment::Text(rest[..index].to_string()));
        }
        segments.push(NativeDecoderTokenSegment::Special(token.clone()));
        rest = &rest[index + token.len()..];
    }
    segments
        .into_iter()
        .filter(|segment| match segment {
            NativeDecoderTokenSegment::Text(text) => !text.is_empty(),
            NativeDecoderTokenSegment::Special(_) => true,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeDecoderChatTemplate {
    template: String,
    globals: HashMap<String, serde_json::Value>,
}

impl NativeDecoderChatTemplate {
    pub(crate) fn from_assets(
        tokenizer_config: Option<&[u8]>,
        chat_template_asset: Option<&[u8]>,
    ) -> Result<Option<Self>> {
        let config = if let Some(bytes) = chat_template_asset {
            Some(NativeDecoderChatTemplateConfig {
                template: chat_template_from_asset(bytes)?,
                globals: tokenizer_config
                    .map(chat_template_globals_from_tokenizer_config)
                    .transpose()?
                    .unwrap_or_default(),
            })
        } else if let Some(bytes) = tokenizer_config {
            chat_template_from_tokenizer_config(bytes)?
        } else {
            None
        };
        Ok(config.map(|config| Self {
            template: config.template,
            globals: config.globals,
        }))
    }

    pub(crate) fn render(
        &self,
        messages: &[NativeDecoderChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template requires at least one message".to_string(),
            });
        }
        render_chat_template(
            &self.template,
            &self.globals,
            messages,
            add_generation_prompt,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeDecoderChatTemplateConfig {
    template: String,
    globals: HashMap<String, serde_json::Value>,
}

pub(crate) fn chat_template_from_tokenizer_config(
    bytes: &[u8],
) -> Result<Option<NativeDecoderChatTemplateConfig>> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("tokenizer_config.json is not valid JSON: {error}"),
        }
    })?;
    let Some(template) = value.get("chat_template") else {
        return Ok(None);
    };
    let Some(template) = template.as_str() else {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "tokenizer_config.json chat_template must be a string".to_string(),
        });
    };
    validate_chat_template_size(template)?;
    Ok(Some(NativeDecoderChatTemplateConfig {
        template: template.to_string(),
        globals: chat_template_globals_from_value(&value),
    }))
}

pub(crate) fn chat_template_from_asset(bytes: &[u8]) -> Result<String> {
    if bytes.len() > MAX_CHAT_TEMPLATE_BYTES {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!(
                "chat template asset is too large: {} bytes exceeds {}",
                bytes.len(),
                MAX_CHAT_TEMPLATE_BYTES
            ),
        });
    }
    let text = std::str::from_utf8(bytes).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("chat_template.json is not UTF-8: {error}"),
        }
    })?;
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(template) = value.as_str() {
            validate_chat_template_size(template)?;
            return Ok(template.to_string());
        }
        if let Some(template) = value
            .get("chat_template")
            .and_then(serde_json::Value::as_str)
        {
            validate_chat_template_size(template)?;
            return Ok(template.to_string());
        }
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "chat_template.json must be a JSON string or object with chat_template"
                .to_string(),
        });
    }
    validate_chat_template_size(text)?;
    Ok(text.to_string())
}

pub(crate) fn render_chat_template(
    template: &str,
    globals: &HashMap<String, serde_json::Value>,
    messages: &[NativeDecoderChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    match render_chat_template_jinja(template, globals, messages, add_generation_prompt) {
        Ok(rendered) => Ok(rendered),
        Err(jinja_error) => {
            if let Ok(rendered) =
                render_chat_template_legacy(template, messages, add_generation_prompt)
            {
                return Ok(rendered);
            }
            Err(jinja_error)
        }
    }
}

pub(crate) fn render_chat_template_jinja(
    template: &str,
    globals: &HashMap<String, serde_json::Value>,
    messages: &[NativeDecoderChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    validate_chat_template_size(template)?;
    let mut env = Environment::new();
    env.set_fuel(Some(CHAT_TEMPLATE_FUEL));
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                JinjaErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    let messages = messages
        .iter()
        .map(|message| json!({ "role": message.role, "content": message.content }))
        .collect::<Vec<_>>();
    let mut context = serde_json::Map::new();
    context.insert("messages".to_string(), serde_json::Value::Array(messages));
    context.insert(
        "add_generation_prompt".to_string(),
        serde_json::Value::Bool(add_generation_prompt),
    );
    context.insert("tools".to_string(), serde_json::Value::Array(Vec::new()));
    context.insert(
        "documents".to_string(),
        serde_json::Value::Array(Vec::new()),
    );
    for (key, value) in globals {
        context.insert(key.clone(), value.clone());
    }
    let rendered = env
        .template_from_str(template)
        .and_then(|template| template.render(JinjaValue::from_serialize(&context)))
        .map_err(|error| RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("chat template render failed: {error}"),
        })?;
    if rendered.len() > MAX_CHAT_RENDERED_BYTES {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!(
                "chat template rendered {} bytes, maximum supported is {}",
                rendered.len(),
                MAX_CHAT_RENDERED_BYTES
            ),
        });
    }
    Ok(rendered)
}

pub(crate) fn render_chat_template_legacy(
    template: &str,
    messages: &[NativeDecoderChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    let Some((before, for_body, after)) = split_jinja_for_messages(template)? else {
        return render_chat_template_segment(template, None, add_generation_prompt);
    };
    let mut rendered = render_chat_template_segment(before, None, add_generation_prompt)?;
    for message in messages {
        rendered.push_str(&render_chat_template_segment(
            for_body,
            Some(message),
            add_generation_prompt,
        )?);
    }
    rendered.push_str(&render_chat_template_segment(
        after,
        None,
        add_generation_prompt,
    )?);
    Ok(rendered)
}

pub(crate) fn validate_chat_template_size(template: &str) -> Result<()> {
    if template.len() > MAX_CHAT_TEMPLATE_BYTES {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!(
                "chat template is too large: {} bytes exceeds {}",
                template.len(),
                MAX_CHAT_TEMPLATE_BYTES
            ),
        });
    }
    Ok(())
}

pub(crate) fn chat_template_globals_from_tokenizer_config(
    bytes: &[u8],
) -> Result<HashMap<String, serde_json::Value>> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("tokenizer_config.json is not valid JSON: {error}"),
        }
    })?;
    Ok(chat_template_globals_from_value(&value))
}

pub(crate) fn chat_template_globals_from_value(
    value: &serde_json::Value,
) -> HashMap<String, serde_json::Value> {
    let mut globals = HashMap::new();
    for key in [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ] {
        if let Some(value) = value.get(key) {
            globals.insert(key.to_string(), value.clone());
        }
    }
    globals
}

pub(crate) fn split_jinja_for_messages(template: &str) -> Result<Option<(&str, &str, &str)>> {
    let Some((for_start, for_end, for_tag)) = find_jinja_tag(template, 0, "for") else {
        return Ok(None);
    };
    if for_tag != "for message in messages" {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("unsupported chat template for block {for_tag:?}"),
        });
    }
    let Some((end_start, end_end, end_tag)) = find_jinja_tag(template, for_end, "endfor") else {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "chat template for block is missing endfor".to_string(),
        });
    };
    if end_tag != "endfor" {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("unsupported chat template end block {end_tag:?}"),
        });
    }
    Ok(Some((
        &template[..for_start],
        &template[for_end..end_start],
        &template[end_end..],
    )))
}

pub(crate) fn render_chat_template_segment(
    segment: &str,
    message: Option<&NativeDecoderChatMessage>,
    add_generation_prompt: bool,
) -> Result<String> {
    let mut rendered = String::new();
    let mut rest = segment;
    while let Some(index) = rest.find('{') {
        rendered.push_str(&rest[..index]);
        rest = &rest[index..];
        if rest.starts_with("{{") {
            let end =
                rest.find("}}")
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "chat template expression is missing }}".to_string(),
                    })?;
            let expression = rest[2..end].trim().trim_matches('-').trim();
            rendered.push_str(&eval_chat_expression(expression, message)?);
            rest = &rest[end + 2..];
        } else if rest.starts_with("{%") {
            let end =
                rest.find("%}")
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "chat template block is missing %}".to_string(),
                    })?;
            let tag = rest[2..end].trim().trim_matches('-').trim();
            if let Some(condition) = tag.strip_prefix("if ") {
                let body_start = end + 2;
                let (branch, next_index) = render_chat_if_block(
                    rest,
                    body_start,
                    condition,
                    message,
                    add_generation_prompt,
                )?;
                rendered.push_str(&branch);
                rest = &rest[next_index..];
            } else {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!("unsupported chat template block {tag:?}"),
                });
            }
        } else {
            rendered.push('{');
            rest = &rest[1..];
        }
    }
    rendered.push_str(rest);
    Ok(rendered)
}

pub(crate) fn render_chat_if_block(
    rest: &str,
    body_start: usize,
    first_condition: &str,
    message: Option<&NativeDecoderChatMessage>,
    add_generation_prompt: bool,
) -> Result<(String, usize)> {
    let mut branches = Vec::new();
    let mut current_condition = Some(first_condition.trim());
    let mut current_body_start = body_start;
    loop {
        let Some((tag_start, tag_end, tag)) = find_next_jinja_control_tag(rest, current_body_start)
        else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template if block is missing endif".to_string(),
            });
        };
        if let Some(condition) = tag.strip_prefix("elif ") {
            branches.push((current_condition, current_body_start, tag_start));
            current_condition = Some(condition.trim());
            current_body_start = tag_end;
        } else if tag == "else" {
            branches.push((current_condition, current_body_start, tag_start));
            current_condition = None;
            current_body_start = tag_end;
        } else if tag == "endif" {
            branches.push((current_condition, current_body_start, tag_start));
            for (condition, start, end) in branches {
                let selected = match condition {
                    Some(condition) => {
                        eval_chat_condition(condition, message, add_generation_prompt)?
                    }
                    None => true,
                };
                if selected {
                    return Ok((
                        render_chat_template_segment(
                            &rest[start..end],
                            message,
                            add_generation_prompt,
                        )?,
                        tag_end,
                    ));
                }
            }
            return Ok((String::new(), tag_end));
        } else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported chat template if control block {tag:?}"),
            });
        }
    }
}

pub(crate) fn find_next_jinja_control_tag(
    template: &str,
    start: usize,
) -> Option<(usize, usize, &str)> {
    let mut search_start = start;
    while let Some(relative_start) = template[search_start..].find("{%") {
        let tag_start = search_start + relative_start;
        let tag_body_start = tag_start + 2;
        let relative_end = template[tag_body_start..].find("%}")?;
        let tag_end = tag_body_start + relative_end + 2;
        let tag = template[tag_body_start..tag_body_start + relative_end]
            .trim()
            .trim_matches('-')
            .trim();
        if tag.starts_with("elif ") || tag == "else" || tag == "endif" {
            return Some((tag_start, tag_end, tag));
        }
        search_start = tag_end;
    }
    None
}

pub(crate) fn eval_chat_condition(
    condition: &str,
    message: Option<&NativeDecoderChatMessage>,
    add_generation_prompt: bool,
) -> Result<bool> {
    match condition {
        "add_generation_prompt" => return Ok(add_generation_prompt),
        "not add_generation_prompt" => return Ok(!add_generation_prompt),
        _ => {}
    }
    if let Some((left, right)) = condition.split_once("==") {
        return Ok(eval_chat_condition_value(left.trim(), message)?
            == eval_chat_condition_literal(right.trim())?);
    }
    if let Some((left, right)) = condition.split_once("!=") {
        return Ok(eval_chat_condition_value(left.trim(), message)?
            != eval_chat_condition_literal(right.trim())?);
    }
    Err(RuntimeError::NativeDecoderTokenizerInvalid {
        reason: format!("unsupported chat template condition {condition:?}"),
    })
}

pub(crate) fn eval_chat_condition_value(
    expression: &str,
    message: Option<&NativeDecoderChatMessage>,
) -> Result<String> {
    match expression {
        "message['role']" | "message[\"role\"]" | "message.role" => message
            .map(|message| message.role.clone())
            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template message role used outside message loop".to_string(),
            }),
        "message['content']" | "message[\"content\"]" | "message.content" => message
            .map(|message| message.content.clone())
            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template message content used outside message loop".to_string(),
            }),
        other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("unsupported chat template condition value {other:?}"),
        }),
    }
}

pub(crate) fn eval_chat_condition_literal(value: &str) -> Result<String> {
    if (value.starts_with('\'') && value.ends_with('\''))
        || (value.starts_with('"') && value.ends_with('"'))
    {
        Ok(unescape_chat_literal(&value[1..value.len() - 1]))
    } else {
        Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("unsupported chat template condition literal {value:?}"),
        })
    }
}

pub(crate) fn find_jinja_tag<'a>(
    template: &'a str,
    start: usize,
    expected_prefix: &str,
) -> Option<(usize, usize, &'a str)> {
    let mut search_start = start;
    while let Some(relative_start) = template[search_start..].find("{%") {
        let tag_start = search_start + relative_start;
        let tag_body_start = tag_start + 2;
        let relative_end = template[tag_body_start..].find("%}")?;
        let tag_end = tag_body_start + relative_end + 2;
        let tag = template[tag_body_start..tag_body_start + relative_end]
            .trim()
            .trim_matches('-')
            .trim();
        if tag.starts_with(expected_prefix) {
            return Some((tag_start, tag_end, tag));
        }
        search_start = tag_end;
    }
    None
}

pub(crate) fn eval_chat_expression(
    expression: &str,
    message: Option<&NativeDecoderChatMessage>,
) -> Result<String> {
    let mut output = String::new();
    for part in split_chat_concat(expression) {
        output.push_str(&eval_chat_expression_part(part.trim(), message)?);
    }
    Ok(output)
}

pub(crate) fn split_chat_concat(expression: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut quote = None;
    let mut escape = false;
    for (index, ch) in expression.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        if let Some(active_quote) = quote {
            if ch == active_quote {
                quote = None;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
        } else if ch == '+' {
            parts.push(&expression[start..index]);
            start = index + ch.len_utf8();
        }
    }
    parts.push(&expression[start..]);
    parts
}

pub(crate) fn eval_chat_expression_part(
    part: &str,
    message: Option<&NativeDecoderChatMessage>,
) -> Result<String> {
    if (part.starts_with('\'') && part.ends_with('\''))
        || (part.starts_with('"') && part.ends_with('"'))
    {
        return Ok(unescape_chat_literal(&part[1..part.len() - 1]));
    }
    if matches!(
        part,
        "message['role']" | "message[\"role\"]" | "message.role"
    ) {
        return message.map(|message| message.role.clone()).ok_or_else(|| {
            RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template message role used outside message loop".to_string(),
            }
        });
    }
    if matches!(
        part,
        "message['content']" | "message[\"content\"]" | "message.content"
    ) {
        return message
            .map(|message| message.content.clone())
            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template message content used outside message loop".to_string(),
            });
    }
    Err(RuntimeError::NativeDecoderTokenizerInvalid {
        reason: format!("unsupported chat template expression part {part:?}"),
    })
}

pub(crate) fn unescape_chat_literal(literal: &str) -> String {
    let mut output = String::new();
    let mut chars = literal.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => output.push('\n'),
            Some('r') => output.push('\r'),
            Some('t') => output.push('\t'),
            Some('\\') => output.push('\\'),
            Some('\'') => output.push('\''),
            Some('"') => output.push('"'),
            Some(other) => {
                output.push('\\');
                output.push(other);
            }
            None => output.push('\\'),
        }
    }
    output
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

pub(crate) fn decode_sentencepiece_tokens(tokens: &[String], replacement: char) -> String {
    let text = tokens.join("");
    text.replace(replacement, " ").trim_start().to_string()
}

pub(crate) fn decode_wordpiece_tokens(tokens: &[String], prefix: &str, cleanup: bool) -> String {
    let mut text = String::new();
    for token in tokens {
        if let Some(rest) = token.strip_prefix(prefix) {
            text.push_str(rest);
        } else {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(token);
        }
    }
    if cleanup {
        cleanup_wordpiece_text(&text)
    } else {
        text
    }
}

pub(crate) fn cleanup_wordpiece_text(text: &str) -> String {
    text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

pub(crate) fn char_boundaries_with_end(text: &str) -> Vec<usize> {
    let mut boundaries = text.char_indices().map(|(idx, _)| idx).collect::<Vec<_>>();
    if boundaries.first().copied() != Some(0) {
        boundaries.insert(0, 0);
    }
    if boundaries.last().copied() != Some(text.len()) {
        boundaries.push(text.len());
    }
    boundaries
}

pub(crate) fn validate_tokenizer_token(token: &str) -> Result<()> {
    if token.chars().count() > MAX_TOKENIZER_TOKEN_CHARS {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!(
                "tokenizer token has {} chars, maximum supported is {}",
                token.chars().count(),
                MAX_TOKENIZER_TOKEN_CHARS
            ),
        });
    }
    Ok(())
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
