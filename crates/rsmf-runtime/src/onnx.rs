use std::collections::HashMap;

use rsmf_core::LogicalDtype;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OnnxInitializerInfo {
    pub(crate) data_type: Option<OnnxTensorDataType>,
    pub(crate) shape: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnnxTensorDataType {
    Float,
    Uint8,
    Int8,
    Int32,
    Int64,
    Bool,
    Double,
    Other(i32),
}

impl OnnxTensorDataType {
    fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            6 => Self::Int32,
            7 => Self::Int64,
            9 => Self::Bool,
            11 => Self::Double,
            other => Self::Other(other),
        }
    }

    pub(crate) fn from_logical_dtype(dtype: LogicalDtype) -> Option<Self> {
        match dtype {
            LogicalDtype::F32 => Some(Self::Float),
            LogicalDtype::F64 => Some(Self::Double),
            LogicalDtype::I64 => Some(Self::Int64),
            LogicalDtype::I32 => Some(Self::Int32),
            LogicalDtype::U8 => Some(Self::Uint8),
            LogicalDtype::I8 => Some(Self::Int8),
            LogicalDtype::Bool => Some(Self::Bool),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Float => "Float/F32",
            Self::Uint8 => "Uint8",
            Self::Int8 => "Int8",
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::Bool => "Bool",
            Self::Double => "Double/F64",
            Self::Other(4) => "Uint16",
            Self::Other(5) => "Int16",
            Self::Other(10) => "Float16",
            Self::Other(16) => "BFloat16",
            Self::Other(_) => "unknown",
        }
    }
}

pub(crate) fn onnx_initializers(
    graph_bytes: &[u8],
) -> std::result::Result<HashMap<String, OnnxInitializerInfo>, String> {
    let graph = onnx_graph(graph_bytes)?
        .ok_or_else(|| "ONNX model does not contain a graph".to_string())?;
    graph_initializers(graph)
}

fn onnx_graph(model: &[u8]) -> std::result::Result<Option<&[u8]>, String> {
    let mut reader = ProtoReader::new(model);
    let mut graph = None;
    while let Some(field) = reader.next_field()? {
        if field.number == 7 && field.wire_type == PROTO_LEN {
            graph = Some(field.bytes);
        }
    }
    Ok(graph)
}

fn graph_initializers(
    graph: &[u8],
) -> std::result::Result<HashMap<String, OnnxInitializerInfo>, String> {
    let mut reader = ProtoReader::new(graph);
    let mut initializers = HashMap::new();
    while let Some(field) = reader.next_field()? {
        if field.number == 5 && field.wire_type == PROTO_LEN {
            let Some((name, info)) = tensor_proto_initializer(field.bytes)? else {
                continue;
            };
            initializers.insert(name, info);
        }
    }
    Ok(initializers)
}

fn tensor_proto_initializer(
    tensor: &[u8],
) -> std::result::Result<Option<(String, OnnxInitializerInfo)>, String> {
    let mut reader = ProtoReader::new(tensor);
    let mut name = None;
    let mut shape = Vec::new();
    let mut data_type = None;

    while let Some(field) = reader.next_field()? {
        match (field.number, field.wire_type) {
            (1, PROTO_VARINT) => {
                let dim = i64::try_from(field.varint).map_err(|_| {
                    format!(
                        "ONNX initializer dimension {} exceeds i64::MAX",
                        field.varint
                    )
                })?;
                if dim < 0 {
                    return Err(format!("ONNX initializer dimension {dim} is negative"));
                }
                shape.push(dim);
            }
            (2, PROTO_VARINT) => {
                let raw = i32::try_from(field.varint).map_err(|_| {
                    format!("ONNX initializer dtype {} exceeds i32::MAX", field.varint)
                })?;
                data_type = Some(OnnxTensorDataType::from_raw(raw));
            }
            (8, PROTO_LEN) => {
                name = Some(
                    std::str::from_utf8(field.bytes)
                        .map_err(|e| format!("ONNX initializer name is not UTF-8: {e}"))?
                        .to_string(),
                );
            }
            _ => {}
        }
    }

    Ok(name.map(|name| (name, OnnxInitializerInfo { data_type, shape })))
}

const PROTO_VARINT: u8 = 0;
const PROTO_LEN: u8 = 2;

#[derive(Debug)]
struct ProtoField<'a> {
    number: u32,
    wire_type: u8,
    varint: u64,
    bytes: &'a [u8],
}

struct ProtoReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn next_field(&mut self) -> std::result::Result<Option<ProtoField<'a>>, String> {
        if self.pos == self.bytes.len() {
            return Ok(None);
        }

        let tag = self.read_varint()?;
        let number = u32::try_from(tag >> 3)
            .map_err(|_| format!("protobuf field number {} exceeds u32::MAX", tag >> 3))?;
        let wire_type = (tag & 0x07) as u8;
        if number == 0 {
            return Err("protobuf field number 0 is invalid".to_string());
        }

        match wire_type {
            PROTO_VARINT => {
                let varint = self.read_varint()?;
                Ok(Some(ProtoField {
                    number,
                    wire_type,
                    varint,
                    bytes: &[],
                }))
            }
            PROTO_LEN => {
                let len = usize::try_from(self.read_varint()?)
                    .map_err(|_| "protobuf length exceeds usize::MAX".to_string())?;
                let end = self
                    .pos
                    .checked_add(len)
                    .ok_or_else(|| "protobuf length overflow".to_string())?;
                if end > self.bytes.len() {
                    return Err("protobuf length-delimited field extends past input".to_string());
                }
                let bytes = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(Some(ProtoField {
                    number,
                    wire_type,
                    varint: 0,
                    bytes,
                }))
            }
            1 => {
                self.skip(8)?;
                Ok(Some(ProtoField {
                    number,
                    wire_type,
                    varint: 0,
                    bytes: &[],
                }))
            }
            5 => {
                self.skip(4)?;
                Ok(Some(ProtoField {
                    number,
                    wire_type,
                    varint: 0,
                    bytes: &[],
                }))
            }
            other => Err(format!("unsupported protobuf wire type {other}")),
        }
    }

    fn read_varint(&mut self) -> std::result::Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let Some(&byte) = self.bytes.get(self.pos) else {
                return Err("unterminated protobuf varint".to_string());
            };
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("protobuf varint exceeds 64 bits".to_string())
    }

    fn skip(&mut self, len: usize) -> std::result::Result<(), String> {
        self.pos = self
            .pos
            .checked_add(len)
            .ok_or_else(|| "protobuf skip overflow".to_string())?;
        if self.pos > self.bytes.len() {
            return Err("protobuf fixed-width field extends past input".to_string());
        }
        Ok(())
    }
}
