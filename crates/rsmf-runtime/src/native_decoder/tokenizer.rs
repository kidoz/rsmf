use super::*;

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

    /// Parse a supported SentencePiece `tokenizer.model` protobuf payload.
    ///
    /// This accepts the subset the native tokenizer can execute exactly today:
    /// Unigram models with whitespace escaping through the standard `▁`
    /// metaspace convention and no embedded precompiled normalizer map.
    pub fn from_sentencepiece_model(bytes: &[u8]) -> Result<Self> {
        Self::from_sentencepiece_model_with_assets(bytes, None, None)
    }

    pub(crate) fn from_sentencepiece_model_with_assets(
        bytes: &[u8],
        tokenizer_config: Option<&[u8]>,
        chat_template_asset: Option<&[u8]>,
    ) -> Result<Self> {
        if bytes.len() > MAX_TOKENIZER_JSON_BYTES {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!(
                    "tokenizer.model is too large: {} bytes exceeds {}",
                    bytes.len(),
                    MAX_TOKENIZER_JSON_BYTES
                ),
            });
        }
        let model = NativeDecoderSentencePieceModel::from_bytes(bytes)?;
        if model.has_precompiled_normalizer {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "SentencePiece precompiled normalizer maps are not supported".to_string(),
            });
        }
        if model.model_type != NativeDecoderSentencePieceModelType::Unigram {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!(
                    "SentencePiece model type {} is not supported by the direct protobuf path",
                    model.model_type.name()
                ),
            });
        }
        let mut vocab = HashMap::with_capacity(model.pieces.len());
        let mut unigram_scores = HashMap::with_capacity(model.pieces.len());
        let mut special_tokens = Vec::new();
        let mut unk_token = None;
        for (idx, piece) in model.pieces.iter().enumerate() {
            validate_tokenizer_token(&piece.piece)?;
            let token_id =
                i64::try_from(idx).map_err(|_| RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!("SentencePiece token index {idx} cannot convert to i64"),
                })?;
            if piece.kind == NativeDecoderSentencePiecePieceType::Unknown {
                unk_token = Some(piece.piece.clone());
            }
            if piece.kind.is_special() {
                special_tokens.push(piece.piece.clone());
            }
            vocab.insert(piece.piece.clone(), token_id);
            unigram_scores.insert(
                piece.piece.clone(),
                (piece.score * 1_000_000.0).round() as i64,
            );
        }
        if vocab.is_empty() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "SentencePiece model has no pieces".to_string(),
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
        special_tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        special_tokens.dedup();
        Ok(Self {
            model_type: "SentencePieceUnigram".to_string(),
            vocab,
            id_to_token,
            unk_token,
            mode: NativeDecoderTokenizerMode::Unigram,
            normalizer: NativeDecoderNormalizer::None,
            pre_tokenizer: NativeDecoderPreTokenizer::Metaspace {
                replacement: '▁',
                add_prefix_space: model.add_dummy_prefix,
            },
            post_processor: NativeDecoderPostProcessor::None,
            bpe_ranks: HashMap::new(),
            unigram_scores,
            wordpiece_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
            byte_fallback: false,
            decoder: NativeDecoderDecoder::Metaspace {
                replacement: '▁',
                add_prefix_space: model.add_dummy_prefix,
            },
            special_tokens,
            chat_template: NativeDecoderChatTemplate::from_assets(
                tokenizer_config,
                chat_template_asset,
            )?,
        })
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

mod encoding;

pub(crate) use encoding::NativeDecoderEncoding;

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

mod normalizer;

pub(crate) use normalizer::{NativeDecoderNormalizer, replace_pattern_from_json};

mod post_processor;

pub(crate) use post_processor::NativeDecoderPostProcessor;

mod decoder;

pub(crate) use decoder::NativeDecoderDecoder;

mod pre_tokenizer;

pub(crate) use pre_tokenizer::NativeDecoderPreTokenizer;

mod sentencepiece_model;

pub(crate) use sentencepiece_model::{
    NativeDecoderSentencePieceModel, NativeDecoderSentencePieceModelType,
    NativeDecoderSentencePiecePieceType,
};

mod json;

pub(crate) use json::{NativeDecoderBpeMergeJson, NativeDecoderTokenizerJson};

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

mod chat_template;

pub(crate) use chat_template::NativeDecoderChatTemplate;

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
