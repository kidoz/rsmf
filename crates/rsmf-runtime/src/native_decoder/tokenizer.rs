use super::*;
use unicode_normalization::UnicodeNormalization;

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
    byte_fallback: bool,
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
            other => {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!(
                        "only WordLevel, BPE, and Unigram tokenizer.json assets are supported, got {other}"
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
        let mut special_tokens = Vec::new();
        for added in raw.added_tokens {
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
            post_processor,
            bpe_ranks,
            unigram_scores,
            byte_fallback: raw.model.byte_fallback,
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
        let model_token_ids = self.encode_without_post_processor(text)?;
        self.post_processor.apply_single(model_token_ids)
    }

    /// Encode a pair of text sequences and apply a pair post-processor when
    /// the tokenizer defines one.
    pub fn encode_pair(&self, first: &str, second: &str) -> Result<Vec<i64>> {
        let first_ids = self.encode_without_post_processor(first)?;
        let second_ids = self.encode_without_post_processor(second)?;
        self.post_processor.apply_pair(first_ids, second_ids)
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

    fn encode_without_post_processor(&self, text: &str) -> Result<Vec<i64>> {
        let normalized = self.normalizer.normalize(text);
        let mut token_ids = Vec::new();
        for segment in isolate_special_token_segments(&normalized, &self.special_tokens) {
            match segment {
                NativeDecoderTokenSegment::Special(token) => {
                    token_ids.push(self.lookup_token_id(&token)?);
                }
                NativeDecoderTokenSegment::Text(text) => {
                    for token in self.pre_tokenizer.pieces(&text) {
                        match self.mode {
                            NativeDecoderTokenizerMode::WordLevel => {
                                token_ids.push(self.lookup_token_id(&token)?);
                            }
                            NativeDecoderTokenizerMode::Bpe => {
                                token_ids.extend(self.encode_bpe_piece(&token)?);
                            }
                            NativeDecoderTokenizerMode::Unigram => {
                                token_ids.extend(self.encode_unigram_piece(&token)?);
                            }
                        }
                    }
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
            NativeDecoderTokenizerMode::Unigram => Ok(decode_sentencepiece_tokens(&tokens)),
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderTokenizerMode {
    WordLevel,
    Bpe,
    Unigram,
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
            Self::Sequence(normalizers) => normalizers
                .iter()
                .fold(text.to_string(), |text, next| next.normalize(&text)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderPostProcessor {
    None,
    TemplateProcessing {
        single: Vec<NativeDecoderTemplatePiece>,
        pair: Vec<NativeDecoderTemplatePiece>,
    },
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
            "RobertaProcessing" => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "unsupported post_processor RobertaProcessing; use TemplateProcessing"
                    .to_string(),
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

    pub(crate) fn apply_single(&self, ids: Vec<i64>) -> Result<Vec<i64>> {
        match self {
            Self::None => Ok(ids),
            Self::TemplateProcessing { single, .. } => apply_template_pieces(single, &ids, &[]),
        }
    }

    pub(crate) fn apply_pair(&self, first: Vec<i64>, second: Vec<i64>) -> Result<Vec<i64>> {
        match self {
            Self::None => {
                let mut ids = first;
                ids.extend(second);
                Ok(ids)
            }
            Self::TemplateProcessing { pair, .. } => apply_template_pieces(pair, &first, &second),
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeDecoderTemplatePiece {
    SequenceA,
    SequenceB,
    SpecialToken { token: String, ids: Vec<i64> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderPreTokenizer {
    Whitespace,
    ByteLevel {
        add_prefix_space: bool,
    },
    Metaspace {
        replacement: char,
        add_prefix_space: bool,
    },
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
                Ok(Self::Metaspace {
                    replacement,
                    add_prefix_space: value
                        .get("add_prefix_space")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(true),
                })
            }
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
            Self::Metaspace {
                replacement,
                add_prefix_space,
            } => {
                let mut piece = text
                    .chars()
                    .map(|ch| if ch.is_whitespace() { replacement } else { ch })
                    .collect::<String>();
                if add_prefix_space && !piece.starts_with(replacement) {
                    piece.insert(0, replacement);
                }
                vec![piece]
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
    vocab: NativeDecoderTokenizerVocabJson,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    unk_id: Option<usize>,
    #[serde(default)]
    merges: Vec<NativeDecoderBpeMergeJson>,
    #[serde(default)]
    byte_fallback: bool,
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

pub(crate) fn apply_template_pieces(
    pieces: &[NativeDecoderTemplatePiece],
    first: &[i64],
    second: &[i64],
) -> Result<Vec<i64>> {
    if pieces.is_empty() {
        let mut ids = first.to_vec();
        ids.extend_from_slice(second);
        return Ok(ids);
    }
    let mut output = Vec::new();
    for piece in pieces {
        match piece {
            NativeDecoderTemplatePiece::SequenceA => output.extend_from_slice(first),
            NativeDecoderTemplatePiece::SequenceB => output.extend_from_slice(second),
            NativeDecoderTemplatePiece::SpecialToken { ids, .. } => output.extend_from_slice(ids),
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
}

impl NativeDecoderChatTemplate {
    pub(crate) fn from_assets(
        tokenizer_config: Option<&[u8]>,
        chat_template_asset: Option<&[u8]>,
    ) -> Result<Option<Self>> {
        let template = if let Some(bytes) = chat_template_asset {
            Some(chat_template_from_asset(bytes)?)
        } else if let Some(bytes) = tokenizer_config {
            chat_template_from_tokenizer_config(bytes)?
        } else {
            None
        };
        Ok(template.map(|template| Self { template }))
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
        render_chat_template(&self.template, messages, add_generation_prompt)
    }
}

pub(crate) fn chat_template_from_tokenizer_config(bytes: &[u8]) -> Result<Option<String>> {
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
    Ok(Some(template.to_string()))
}

pub(crate) fn chat_template_from_asset(bytes: &[u8]) -> Result<String> {
    let text = std::str::from_utf8(bytes).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("chat_template.json is not UTF-8: {error}"),
        }
    })?;
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(template) = value.as_str() {
            return Ok(template.to_string());
        }
        if let Some(template) = value
            .get("chat_template")
            .and_then(serde_json::Value::as_str)
        {
            return Ok(template.to_string());
        }
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "chat_template.json must be a JSON string or object with chat_template"
                .to_string(),
        });
    }
    Ok(text.to_string())
}

pub(crate) fn render_chat_template(
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

pub(crate) fn decode_sentencepiece_tokens(tokens: &[String]) -> String {
    let text = tokens.join("");
    text.replace('▁', " ").trim_start().to_string()
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
