use crate::{Result, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderSentencePieceModelType {
    Unigram,
    Bpe,
    Word,
    Char,
    Unknown(i32),
}

impl NativeDecoderSentencePieceModelType {
    pub(crate) fn name(self) -> String {
        match self {
            Self::Unigram => "unigram".to_string(),
            Self::Bpe => "bpe".to_string(),
            Self::Word => "word".to_string(),
            Self::Char => "char".to_string(),
            Self::Unknown(value) => format!("unknown({value})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecoderSentencePiecePieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
    UnknownRaw(i32),
}

impl NativeDecoderSentencePiecePieceType {
    pub(crate) fn is_special(self) -> bool {
        matches!(self, Self::Unknown | Self::Control | Self::UserDefined)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeDecoderSentencePiecePiece {
    pub(crate) piece: String,
    pub(crate) score: f32,
    pub(crate) kind: NativeDecoderSentencePiecePieceType,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeDecoderSentencePieceModel {
    pub(crate) pieces: Vec<NativeDecoderSentencePiecePiece>,
    pub(crate) model_type: NativeDecoderSentencePieceModelType,
    pub(crate) add_dummy_prefix: bool,
    pub(crate) has_precompiled_normalizer: bool,
}

impl NativeDecoderSentencePieceModel {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = SentencePieceProtoReader::new(bytes);
        let mut pieces = Vec::new();
        let mut model_type = NativeDecoderSentencePieceModelType::Unigram;
        let mut add_dummy_prefix = true;
        let mut has_precompiled_normalizer = false;
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 if field.wire_type == SENTENCEPIECE_WIRE_LEN => {
                    pieces.push(parse_sentencepiece_piece(
                        field.bytes.ok_or_else(sentencepiece_missing_bytes)?,
                    )?);
                }
                2 if field.wire_type == SENTENCEPIECE_WIRE_LEN => {
                    model_type = parse_sentencepiece_trainer_spec(
                        field.bytes.ok_or_else(sentencepiece_missing_bytes)?,
                    )?;
                }
                3 if field.wire_type == SENTENCEPIECE_WIRE_LEN => {
                    let normalizer = parse_sentencepiece_normalizer_spec(
                        field.bytes.ok_or_else(sentencepiece_missing_bytes)?,
                    )?;
                    add_dummy_prefix = normalizer.add_dummy_prefix;
                    has_precompiled_normalizer = normalizer.has_precompiled_normalizer;
                }
                _ => {}
            }
        }
        Ok(Self {
            pieces,
            model_type,
            add_dummy_prefix,
            has_precompiled_normalizer,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SentencePieceProtoField<'a> {
    number: u32,
    wire_type: u8,
    varint: Option<u64>,
    fixed32: Option<u32>,
    bytes: Option<&'a [u8]>,
}

struct SentencePieceProtoReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SentencePieceProtoReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn next_field(&mut self) -> Result<Option<SentencePieceProtoField<'a>>> {
        if self.offset == self.bytes.len() {
            return Ok(None);
        }
        let tag = self.read_varint()?;
        let number =
            u32::try_from(tag >> 3).map_err(|_| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "SentencePiece protobuf field number exceeds u32::MAX".to_string(),
            })?;
        if number == 0 {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "SentencePiece protobuf field number 0 is invalid".to_string(),
            });
        }
        let wire_type = (tag & 0x07) as u8;
        match wire_type {
            SENTENCEPIECE_WIRE_VARINT => Ok(Some(SentencePieceProtoField {
                number,
                wire_type,
                varint: Some(self.read_varint()?),
                fixed32: None,
                bytes: None,
            })),
            SENTENCEPIECE_WIRE_LEN => {
                let len = usize::try_from(self.read_varint()?).map_err(|_| {
                    RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "SentencePiece protobuf length exceeds usize::MAX".to_string(),
                    }
                })?;
                let end = self.offset.checked_add(len).ok_or_else(|| {
                    RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "SentencePiece protobuf length overflow".to_string(),
                    }
                })?;
                if end > self.bytes.len() {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "SentencePiece protobuf length-delimited field extends past input"
                            .to_string(),
                    });
                }
                let bytes = &self.bytes[self.offset..end];
                self.offset = end;
                Ok(Some(SentencePieceProtoField {
                    number,
                    wire_type,
                    varint: None,
                    fixed32: None,
                    bytes: Some(bytes),
                }))
            }
            SENTENCEPIECE_WIRE_FIXED32 => {
                let end = self.offset.checked_add(4).ok_or_else(|| {
                    RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "SentencePiece protobuf fixed32 overflow".to_string(),
                    }
                })?;
                if end > self.bytes.len() {
                    return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "SentencePiece protobuf fixed32 extends past input".to_string(),
                    });
                }
                let raw = u32::from_le_bytes([
                    self.bytes[self.offset],
                    self.bytes[self.offset + 1],
                    self.bytes[self.offset + 2],
                    self.bytes[self.offset + 3],
                ]);
                self.offset = end;
                Ok(Some(SentencePieceProtoField {
                    number,
                    wire_type,
                    varint: None,
                    fixed32: Some(raw),
                    bytes: None,
                }))
            }
            SENTENCEPIECE_WIRE_FIXED64 => {
                self.skip_bytes(8)?;
                Ok(Some(SentencePieceProtoField {
                    number,
                    wire_type,
                    varint: None,
                    fixed32: None,
                    bytes: None,
                }))
            }
            other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported SentencePiece protobuf wire type {other}"),
            }),
        }
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            if self.offset >= self.bytes.len() {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: "unterminated SentencePiece protobuf varint".to_string(),
                });
            }
            let byte = self.bytes[self.offset];
            self.offset += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "SentencePiece protobuf varint exceeds 64 bits".to_string(),
        })
    }

    fn skip_bytes(&mut self, len: usize) -> Result<()> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "SentencePiece protobuf skip overflow".to_string(),
            }
        })?;
        if end > self.bytes.len() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "SentencePiece protobuf fixed-width field extends past input".to_string(),
            });
        }
        self.offset = end;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct SentencePieceNormalizerSpec {
    add_dummy_prefix: bool,
    has_precompiled_normalizer: bool,
}

fn parse_sentencepiece_piece(bytes: &[u8]) -> Result<NativeDecoderSentencePiecePiece> {
    let mut reader = SentencePieceProtoReader::new(bytes);
    let mut piece = None;
    let mut score = 0.0f32;
    let mut kind = NativeDecoderSentencePiecePieceType::Normal;
    while let Some(field) = reader.next_field()? {
        match field.number {
            1 if field.wire_type == SENTENCEPIECE_WIRE_LEN => {
                piece = Some(
                    std::str::from_utf8(field.bytes.ok_or_else(sentencepiece_missing_bytes)?)
                        .map_err(|error| RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: format!("SentencePiece piece is not UTF-8: {error}"),
                        })?
                        .to_string(),
                );
            }
            2 if field.wire_type == SENTENCEPIECE_WIRE_FIXED32 => {
                score = f32::from_bits(field.fixed32.ok_or_else(sentencepiece_missing_fixed32)?);
            }
            3 if field.wire_type == SENTENCEPIECE_WIRE_VARINT => {
                kind = sentencepiece_piece_type(
                    i32::try_from(field.varint.ok_or_else(sentencepiece_missing_varint)?).map_err(
                        |_| RuntimeError::NativeDecoderTokenizerInvalid {
                            reason: "SentencePiece piece type exceeds i32::MAX".to_string(),
                        },
                    )?,
                );
            }
            _ => {}
        }
    }
    let piece = piece.ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
        reason: "SentencePiece piece entry is missing piece text".to_string(),
    })?;
    Ok(NativeDecoderSentencePiecePiece { piece, score, kind })
}

fn parse_sentencepiece_trainer_spec(bytes: &[u8]) -> Result<NativeDecoderSentencePieceModelType> {
    let mut reader = SentencePieceProtoReader::new(bytes);
    let mut model_type = NativeDecoderSentencePieceModelType::Unigram;
    while let Some(field) = reader.next_field()? {
        if field.number == 3 && field.wire_type == SENTENCEPIECE_WIRE_VARINT {
            model_type = sentencepiece_model_type(
                i32::try_from(field.varint.ok_or_else(sentencepiece_missing_varint)?).map_err(
                    |_| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "SentencePiece model type exceeds i32::MAX".to_string(),
                    },
                )?,
            );
        }
    }
    Ok(model_type)
}

fn parse_sentencepiece_normalizer_spec(bytes: &[u8]) -> Result<SentencePieceNormalizerSpec> {
    let mut reader = SentencePieceProtoReader::new(bytes);
    let mut add_dummy_prefix = true;
    let mut has_precompiled_normalizer = false;
    while let Some(field) = reader.next_field()? {
        match field.number {
            2 if field.wire_type == SENTENCEPIECE_WIRE_LEN => {
                has_precompiled_normalizer = !field
                    .bytes
                    .ok_or_else(sentencepiece_missing_bytes)?
                    .is_empty();
            }
            3 if field.wire_type == SENTENCEPIECE_WIRE_VARINT => {
                add_dummy_prefix = field.varint.ok_or_else(sentencepiece_missing_varint)? != 0;
            }
            _ => {}
        }
    }
    Ok(SentencePieceNormalizerSpec {
        add_dummy_prefix,
        has_precompiled_normalizer,
    })
}

fn sentencepiece_model_type(raw: i32) -> NativeDecoderSentencePieceModelType {
    match raw {
        1 => NativeDecoderSentencePieceModelType::Unigram,
        2 => NativeDecoderSentencePieceModelType::Bpe,
        3 => NativeDecoderSentencePieceModelType::Word,
        4 => NativeDecoderSentencePieceModelType::Char,
        other => NativeDecoderSentencePieceModelType::Unknown(other),
    }
}

fn sentencepiece_piece_type(raw: i32) -> NativeDecoderSentencePiecePieceType {
    match raw {
        1 => NativeDecoderSentencePiecePieceType::Normal,
        2 => NativeDecoderSentencePiecePieceType::Unknown,
        3 => NativeDecoderSentencePiecePieceType::Control,
        4 => NativeDecoderSentencePiecePieceType::UserDefined,
        5 => NativeDecoderSentencePiecePieceType::Unused,
        6 => NativeDecoderSentencePiecePieceType::Byte,
        other => NativeDecoderSentencePiecePieceType::UnknownRaw(other),
    }
}

fn sentencepiece_missing_bytes() -> RuntimeError {
    RuntimeError::NativeDecoderTokenizerInvalid {
        reason: "internal SentencePiece protobuf parser expected bytes".to_string(),
    }
}

fn sentencepiece_missing_fixed32() -> RuntimeError {
    RuntimeError::NativeDecoderTokenizerInvalid {
        reason: "internal SentencePiece protobuf parser expected fixed32".to_string(),
    }
}

fn sentencepiece_missing_varint() -> RuntimeError {
    RuntimeError::NativeDecoderTokenizerInvalid {
        reason: "internal SentencePiece protobuf parser expected varint".to_string(),
    }
}

const SENTENCEPIECE_WIRE_VARINT: u8 = 0;
const SENTENCEPIECE_WIRE_FIXED64: u8 = 1;
const SENTENCEPIECE_WIRE_LEN: u8 = 2;
const SENTENCEPIECE_WIRE_FIXED32: u8 = 5;
