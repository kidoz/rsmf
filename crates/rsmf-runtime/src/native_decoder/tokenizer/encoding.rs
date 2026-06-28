#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeDecoderEncoding {
    pub(crate) ids: Vec<i64>,
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
