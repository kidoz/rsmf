//! Versioned custom binary manifest codec.
//!
//! The manifest is a deterministic little-endian byte stream. Encoding never
//! uses `serde` or FlatBuffers — see `docs/FORMAT_DECISIONS.md` for the
//! rationale.
//!
//! The layout is specified in `docs/SPEC.md` §4. Helper primitives:
//!
//! * `StringMap`: `u32 count, { u32 key_len, key, u32 value_len, value }*`
//! * variable-length arrays: `u32 count, item*`
//! * strings: UTF-8, length-prefixed with `u32`
//! * integers: little-endian

use crate::checksum::CHECKSUM_LEN;
use crate::error::{Result, RsmfError};
use crate::tensor::descriptor::TensorDescriptor;
use crate::tensor::dtype::{LogicalDtype, StorageDtype};
use crate::tensor::variant::{EncodingKind, LayoutTag, TargetTag, VariantDescriptor, VariantMeta};

/// Current manifest version on disk.
pub const MANIFEST_VERSION: u32 = 1;

/// Opaque graph kind carried on [`GraphDescriptor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GraphKind {
    /// ONNX protobuf bytes.
    Onnx = 1,
    /// ORT (`.ort`) bytes.
    Ort = 2,
    /// Some other opaque bytes. Interpretation left to the caller.
    Other = 3,
}

impl GraphKind {
    /// Decode from the on-disk discriminant.
    pub fn from_raw(raw: u8) -> Result<Self> {
        Ok(match raw {
            1 => Self::Onnx,
            2 => Self::Ort,
            3 => Self::Other,
            other => {
                return Err(RsmfError::structural(format!("unknown graph kind {other}")));
            }
        })
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Ort => "ort",
            Self::Other => "other",
        }
    }
}

/// Descriptor for the opaque graph section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphDescriptor {
    /// Graph kind.
    pub kind: GraphKind,
    /// Alignment of the graph payload.
    pub alignment: u32,
    /// Offset inside the graph section payload.
    pub offset: u64,
    /// Length of the graph payload.
    pub length: u64,
    /// BLAKE3-128 of the graph payload.
    pub checksum: [u8; CHECKSUM_LEN],
    /// Freeform metadata.
    pub metadata: Vec<(String, String)>,
}

/// Descriptor for a named asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetDescriptor {
    /// Asset name (unique within the file).
    pub name: String,
    /// Alignment of the asset payload.
    pub alignment: u32,
    /// Offset inside the Assets section payload.
    pub offset: u64,
    /// Length of the asset payload.
    pub length: u64,
    /// BLAKE3-128 of the asset payload.
    pub checksum: [u8; CHECKSUM_LEN],
    /// Freeform metadata.
    pub metadata: Vec<(String, String)>,
}

/// Fully parsed manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Manifest version.
    pub version: u32,
    /// Global metadata map.
    pub metadata: Vec<(String, String)>,
    /// Tensor descriptors.
    pub tensors: Vec<TensorDescriptor>,
    /// Variant descriptors referenced by tensors.
    pub variants: Vec<VariantDescriptor>,
    /// Optional graph descriptors.
    pub graphs: Vec<GraphDescriptor>,
    /// Asset descriptors.
    pub assets: Vec<AssetDescriptor>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            metadata: Vec::new(),
            tensors: Vec::new(),
            variants: Vec::new(),
            graphs: Vec::new(),
            assets: Vec::new(),
        }
    }
}

impl Manifest {
    /// Encode the manifest to a byte vector.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut w = Writer::default();
        // Envelope
        w.u32(self.version);
        w.u32(0); // reserved

        // Global metadata
        write_string_map(&mut w, &self.metadata)?;

        // Tensors
        w.u32(cast_u32(self.tensors.len())?);
        for t in &self.tensors {
            write_tensor_descriptor(&mut w, t)?;
        }

        // Variants
        w.u32(cast_u32(self.variants.len())?);
        for v in &self.variants {
            write_variant_descriptor(&mut w, v)?;
        }

        // Graphs
        w.u32(cast_u32(self.graphs.len())?);
        for g in &self.graphs {
            write_graph_descriptor(&mut w, g)?;
        }

        // Assets
        w.u32(cast_u32(self.assets.len())?);
        for a in &self.assets {
            write_asset_descriptor(&mut w, a)?;
        }

        Ok(w.into_vec())
    }

    /// Decode a manifest from `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let version = r.u32()?;
        if version != MANIFEST_VERSION {
            return Err(RsmfError::structural(format!(
                "unsupported manifest version {version}"
            )));
        }
        let reserved = r.u32()?;
        if reserved != 0 {
            return Err(RsmfError::structural(
                "manifest reserved field must be zero".to_string(),
            ));
        }

        let metadata = read_string_map(&mut r)?;

        let tensor_count = r.u32()? as usize;
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            tensors.push(read_tensor_descriptor(&mut r)?);
        }

        let variant_count = r.u32()? as usize;
        let mut variants = Vec::with_capacity(variant_count);
        for _ in 0..variant_count {
            variants.push(read_variant_descriptor(&mut r)?);
        }

        let graph_count = r.u32()? as usize;
        let mut graphs = Vec::with_capacity(graph_count);
        for _ in 0..graph_count {
            graphs.push(read_graph_descriptor(&mut r)?);
        }

        let asset_count = r.u32()? as usize;
        let mut assets = Vec::with_capacity(asset_count);
        for _ in 0..asset_count {
            assets.push(read_asset_descriptor(&mut r)?);
        }

        if r.remaining() != 0 {
            return Err(RsmfError::structural(format!(
                "manifest has {} trailing bytes",
                r.remaining()
            )));
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            variants,
            graphs,
            assets,
        })
    }
}

// -- low-level cursor helpers ---------------------------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(RsmfError::structural(format!(
                "manifest truncated: need {n} more bytes at offset {}",
                self.pos
            )));
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.bytes[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.bytes[self.pos], self.bytes[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let v = u32::from_le_bytes([
            self.bytes[self.pos],
            self.bytes[self.pos + 1],
            self.bytes[self.pos + 2],
            self.bytes[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn u64(&mut self) -> Result<u64> {
        self.need(8)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.bytes[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(b))
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn checksum(&mut self) -> Result<[u8; CHECKSUM_LEN]> {
        let bs = self.bytes(CHECKSUM_LEN)?;
        let mut out = [0u8; CHECKSUM_LEN];
        out.copy_from_slice(bs);
        Ok(out)
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        let bs = self.bytes(len)?;
        std::str::from_utf8(bs)
            .map(String::from)
            .map_err(|e| RsmfError::structural(format!("invalid utf-8: {e}")))
    }
}

#[derive(Default)]
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    fn string(&mut self, s: &str) -> Result<()> {
        self.u32(cast_u32(s.len())?);
        self.buf.extend_from_slice(s.as_bytes());
        Ok(())
    }

    fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

fn cast_u32(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| RsmfError::structural("manifest count exceeds u32::MAX"))
}

fn cast_u16(n: usize) -> Result<u16> {
    u16::try_from(n).map_err(|_| RsmfError::structural("count exceeds u16::MAX"))
}

// -- StringMap ------------------------------------------------------------------

fn write_string_map(w: &mut Writer, map: &[(String, String)]) -> Result<()> {
    w.u32(cast_u32(map.len())?);
    for (k, v) in map {
        w.string(k)?;
        w.string(v)?;
    }
    Ok(())
}

fn read_string_map(r: &mut Reader<'_>) -> Result<Vec<(String, String)>> {
    let n = r.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let k = r.string()?;
        let v = r.string()?;
        out.push((k, v));
    }
    Ok(out)
}

// -- Tensor descriptor ----------------------------------------------------------

fn write_tensor_descriptor(w: &mut Writer, t: &TensorDescriptor) -> Result<()> {
    w.string(&t.name)?;
    w.u16(t.dtype as u16);
    w.u16(0); // reserved
    w.u32(cast_u32(t.shape.len())?);
    for &d in &t.shape {
        w.u64(d);
    }
    w.u32(t.canonical_variant);
    w.u32(cast_u32(t.packed_variants.len())?);
    for &idx in &t.packed_variants {
        w.u32(idx);
    }
    w.u64(t.shard_id);
    write_string_map(w, &t.metadata)?;
    Ok(())
}

fn read_tensor_descriptor(r: &mut Reader<'_>) -> Result<TensorDescriptor> {
    let name = r.string()?;
    let dtype_raw = r.u16()?;
    let dtype = LogicalDtype::from_raw(dtype_raw)?;
    let reserved = r.u16()?;
    if reserved != 0 {
        return Err(RsmfError::structural(
            "tensor reserved field must be zero".to_string(),
        ));
    }
    let rank = r.u32()? as usize;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(r.u64()?);
    }
    let canonical_variant = r.u32()?;
    let packed_count = r.u32()? as usize;
    let mut packed_variants = Vec::with_capacity(packed_count);
    for _ in 0..packed_count {
        packed_variants.push(r.u32()?);
    }
    let shard_id = r.u64()?;
    let metadata = read_string_map(r)?;
    Ok(TensorDescriptor {
        name,
        dtype,
        shape,
        canonical_variant,
        packed_variants,
        shard_id,
        metadata,
    })
}

// -- Variant descriptor ---------------------------------------------------------

fn write_variant_descriptor(w: &mut Writer, v: &VariantDescriptor) -> Result<()> {
    w.u16(v.target as u16);
    w.u16(v.encoding as u16);
    w.u16(v.storage_dtype.to_raw());
    w.u16(v.layout as u16);
    w.u32(v.alignment);
    w.u32(0); // reserved
    w.u64(v.section_relative_offset);
    w.u64(v.length);
    w.bytes(&v.checksum);
    w.u8(v.section_kind);
    w.u8(v.section_index);
    w.u16(0); // reserved2
    write_variant_meta(w, &v.meta)?;
    Ok(())
}

fn read_variant_descriptor(r: &mut Reader<'_>) -> Result<VariantDescriptor> {
    let target = TargetTag::from_raw(r.u16()?)?;
    let encoding = EncodingKind::from_raw(r.u16()?)?;
    let storage_dtype = StorageDtype::from_raw(r.u16()?)?;
    let layout = LayoutTag::from_raw(r.u16()?)?;
    let alignment = r.u32()?;
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return Err(RsmfError::structural(format!(
            "variant alignment not power of two: {alignment}"
        )));
    }
    let reserved = r.u32()?;
    if reserved != 0 {
        return Err(RsmfError::structural(
            "variant reserved field must be zero".to_string(),
        ));
    }
    let offset = r.u64()?;
    let length = r.u64()?;
    let checksum = r.checksum()?;
    let section_kind = r.u8()?;
    let section_index = r.u8()?;
    let reserved2 = r.u16()?;
    if reserved2 != 0 {
        return Err(RsmfError::structural(
            "variant reserved2 must be zero".to_string(),
        ));
    }
    let meta = read_variant_meta(r)?;

    Ok(VariantDescriptor {
        target,
        encoding,
        storage_dtype,
        layout,
        alignment,
        section_relative_offset: offset,
        length,
        checksum,
        section_kind,
        section_index,
        meta,
    })
}

// -- Variant meta ---------------------------------------------------------------

fn write_variant_meta(w: &mut Writer, m: &VariantMeta) -> Result<()> {
    w.u16(cast_u16(m.block_shape.len())?);
    for &d in &m.block_shape {
        w.u64(d);
    }
    w.u32(m.group_size);
    w.u16(
        m.scale_dtype
            .map_or(VariantMeta::NONE_DTYPE_SENTINEL, |d| d.to_raw()),
    );
    w.u16(
        m.zero_point_dtype
            .map_or(VariantMeta::NONE_DTYPE_SENTINEL, |d| d.to_raw()),
    );
    w.u32(0); // reserved
    write_string_map(w, &m.extra)?;
    Ok(())
}

fn read_variant_meta(r: &mut Reader<'_>) -> Result<VariantMeta> {
    let block_rank = r.u16()? as usize;
    let mut block_shape = Vec::with_capacity(block_rank);
    for _ in 0..block_rank {
        block_shape.push(r.u64()?);
    }
    let group_size = r.u32()?;
    let scale_raw = r.u16()?;
    let scale_dtype = if scale_raw == VariantMeta::NONE_DTYPE_SENTINEL {
        None
    } else {
        Some(StorageDtype::from_raw(scale_raw)?)
    };
    let zp_raw = r.u16()?;
    let zero_point_dtype = if zp_raw == VariantMeta::NONE_DTYPE_SENTINEL {
        None
    } else {
        Some(StorageDtype::from_raw(zp_raw)?)
    };
    let reserved = r.u32()?;
    if reserved != 0 {
        return Err(RsmfError::structural(
            "variant meta reserved must be zero".to_string(),
        ));
    }
    let extra = read_string_map(r)?;
    Ok(VariantMeta {
        block_shape,
        group_size,
        scale_dtype,
        zero_point_dtype,
        extra,
    })
}

// -- Graph descriptor -----------------------------------------------------------

fn write_graph_descriptor(w: &mut Writer, g: &GraphDescriptor) -> Result<()> {
    w.u8(g.kind as u8);
    w.u8(0); // reserved
    w.u16(0); // reserved
    w.u32(g.alignment);
    w.u64(g.offset);
    w.u64(g.length);
    w.bytes(&g.checksum);
    write_string_map(w, &g.metadata)?;
    Ok(())
}

fn read_graph_descriptor(r: &mut Reader<'_>) -> Result<GraphDescriptor> {
    let kind = GraphKind::from_raw(r.u8()?)?;
    let reserved0 = r.u8()?;
    let reserved1 = r.u16()?;
    if reserved0 != 0 || reserved1 != 0 {
        return Err(RsmfError::structural(
            "graph descriptor reserved must be zero".to_string(),
        ));
    }
    let alignment = r.u32()?;
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return Err(RsmfError::structural(format!(
            "graph alignment not power of two: {alignment}"
        )));
    }
    let offset = r.u64()?;
    let length = r.u64()?;
    let checksum = r.checksum()?;
    let metadata = read_string_map(r)?;
    Ok(GraphDescriptor {
        kind,
        alignment,
        offset,
        length,
        checksum,
        metadata,
    })
}

// -- Asset descriptor -----------------------------------------------------------

fn write_asset_descriptor(w: &mut Writer, a: &AssetDescriptor) -> Result<()> {
    w.string(&a.name)?;
    w.u32(0); // reserved
    w.u32(a.alignment);
    w.u64(a.offset);
    w.u64(a.length);
    w.bytes(&a.checksum);
    write_string_map(w, &a.metadata)?;
    Ok(())
}

fn read_asset_descriptor(r: &mut Reader<'_>) -> Result<AssetDescriptor> {
    let name = r.string()?;
    let reserved = r.u32()?;
    if reserved != 0 {
        return Err(RsmfError::structural(
            "asset reserved must be zero".to_string(),
        ));
    }
    let alignment = r.u32()?;
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return Err(RsmfError::structural(format!(
            "asset alignment not power of two: {alignment}"
        )));
    }
    let offset = r.u64()?;
    let length = r.u64()?;
    let checksum = r.checksum()?;
    let metadata = read_string_map(r)?;
    Ok(AssetDescriptor {
        name,
        alignment,
        offset,
        length,
        checksum,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::section::SectionKind;

    #[test]
    fn empty_manifest_roundtrip() {
        let m = Manifest::default();
        let bytes = m.encode().unwrap();
        let back = Manifest::decode(&bytes).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn populated_manifest_roundtrip() {
        let v = VariantDescriptor {
            target: TargetTag::Canonical,
            encoding: EncodingKind::Raw,
            storage_dtype: StorageDtype::Logical(LogicalDtype::F32),
            layout: LayoutTag::RowMajor,
            alignment: 64,
            section_relative_offset: 0,
            length: 16,
            checksum: [9u8; CHECKSUM_LEN],
            section_kind: SectionKind::CanonicalArena.to_raw() as u8,
            section_index: 0,
            meta: VariantMeta::default(),
        };
        let t = TensorDescriptor {
            name: "weight".into(),
            dtype: LogicalDtype::F32,
            shape: vec![2, 2],
            canonical_variant: 0,
            packed_variants: vec![],
            shard_id: 0,
            metadata: vec![("framework".into(), "rsmf".into())],
        };
        let g = GraphDescriptor {
            kind: GraphKind::Onnx,
            alignment: 8,
            offset: 0,
            length: 64,
            checksum: [1u8; CHECKSUM_LEN],
            metadata: vec![],
        };
        let a = AssetDescriptor {
            name: "tokenizer.json".into(),
            alignment: 8,
            offset: 0,
            length: 16,
            checksum: [2u8; CHECKSUM_LEN],
            metadata: vec![],
        };
        let m = Manifest {
            version: MANIFEST_VERSION,
            metadata: vec![("arch".into(), "mini".into())],
            tensors: vec![t],
            variants: vec![v],
            graphs: vec![g],
            assets: vec![a],
        };
        let bytes = m.encode().unwrap();
        let back = Manifest::decode(&bytes).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn reject_truncated_manifest() {
        let bytes = Manifest::default().encode().unwrap();
        let err = Manifest::decode(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(matches!(err, RsmfError::Structural(_)));
    }

    #[test]
    fn reject_wrong_version() {
        let mut bytes = Manifest::default().encode().unwrap();
        bytes[0..4].copy_from_slice(&999u32.to_le_bytes());
        let err = Manifest::decode(&bytes).unwrap_err();
        assert!(matches!(err, RsmfError::Structural(_)));
    }
}
