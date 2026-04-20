//! Lazy, range-based RSMF reader.
//!
//! [`LazyRsmfFile`] opens an RSMF file without materialising the whole byte
//! stream. It issues three small range reads — preamble, section table,
//! manifest — and then fetches each tensor variant, asset, or graph on
//! demand when the caller asks. The use case is loading a multi-GB model
//! over HTTP/S3 when the caller only needs the manifest (to `inspect` or to
//! decide which tensors are worth downloading).
//!
//! The byte source is abstracted as [`RangeReader`]. [`SliceRangeReader`]
//! wraps an in-memory `Vec<u8>` for tests and for callers who already have
//! the bytes. Third-party crates (e.g. an HTTP client) can supply their own
//! [`RangeReader`] impl without this crate pulling in networking runtimes.
//!
//! **Compression.** A compressed section is fetched in its entirety and
//! decompressed before the caller's variant is sliced out. This module does
//! not cache decompressed sections — two successive `fetch_*` calls against
//! the same compressed section re-fetch and re-decompress. Callers that
//! repeatedly access a compressed section should wrap their `RangeReader`
//! in a cache, or use [`crate::RsmfFile`] which caches per-section.

use std::collections::HashMap;

#[cfg(feature = "tracing")]
use tracing::info_span;

use crate::adapter::{AdapterIndex, adapter_index_from_manifest};
use crate::error::{Result, RsmfError};
use crate::manifest::Manifest;
use crate::preamble::{PREAMBLE_LEN, Preamble};
use crate::reader::{decompress_zstd, validate_manifest, validate_section_table};
use crate::section::{SECTION_DESC_LEN, SectionDescriptor, SectionKind};
use crate::tensor::variant::VariantDescriptor;

/// Pluggable byte source used by [`LazyRsmfFile`].
///
/// Implementers must return the requested range exactly — no short reads,
/// no extra bytes. `read_range` returning fewer than `length` bytes is a
/// [`RsmfError::Structural`] in the caller's view.
pub trait RangeReader: Send + Sync {
    /// Total length of the underlying object, in bytes.
    fn len(&self) -> u64;

    /// True when [`Self::len`] is zero. Default impl suffices for most cases.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read exactly `length` bytes starting at `offset`.
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>>;
}

impl<R: RangeReader + ?Sized> RangeReader for &R {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        (**self).read_range(offset, length)
    }
}

impl<R: RangeReader + ?Sized> RangeReader for Box<R> {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        (**self).read_range(offset, length)
    }
}

impl<R: RangeReader + ?Sized> RangeReader for std::sync::Arc<R> {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        (**self).read_range(offset, length)
    }
}

/// [`RangeReader`] backed by an owned byte buffer.
///
/// Useful in tests, for files already fully in memory (e.g. returned from
/// an earlier `write_to_bytes()`), or as a substitute for a real remote
/// fetcher when one isn't available.
#[derive(Debug, Clone)]
pub struct SliceRangeReader {
    bytes: Vec<u8>,
}

impl SliceRangeReader {
    /// Wrap an owned byte buffer.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Borrow the backing bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl RangeReader for SliceRangeReader {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| RsmfError::structural("read_range length overflow".to_string()))?;
        if end > self.bytes.len() as u64 {
            return Err(RsmfError::structural(format!(
                "read_range {offset}..{end} exceeds source length {}",
                self.bytes.len()
            )));
        }
        Ok(self.bytes[offset as usize..end as usize].to_vec())
    }
}

/// A parsed-but-lazy RSMF file. Metadata (preamble, section table, manifest)
/// is in memory; tensor/asset/graph payloads are fetched on demand.
#[derive(Debug)]
pub struct LazyRsmfFile<R: RangeReader> {
    reader: R,
    preamble: Preamble,
    sections: Vec<SectionDescriptor>,
    manifest: Manifest,
    canonical_section_idx: usize,
    packed_section_indices: Vec<usize>,
    graph_section_idx: Option<usize>,
    assets_section_idx: Option<usize>,
    tensor_by_name: HashMap<String, usize>,
    asset_by_name: HashMap<String, usize>,
}

impl<R: RangeReader> LazyRsmfFile<R> {
    /// Open a lazy reader, reading only the preamble, section table, and
    /// manifest from `reader`.
    pub fn open(reader: R) -> Result<Self> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("LazyRsmfFile::open", len = reader.len()).entered();

        let file_len = reader.len();
        if file_len < PREAMBLE_LEN {
            return Err(RsmfError::structural(format!(
                "source too short for preamble: {file_len} bytes",
            )));
        }

        // 1. Preamble.
        let preamble_bytes = reader.read_range(0, PREAMBLE_LEN)?;
        let preamble = Preamble::decode(&preamble_bytes)?;
        if preamble.section_tbl_off != PREAMBLE_LEN {
            return Err(RsmfError::structural(format!(
                "preamble.section_tbl_off {} must equal preamble length {PREAMBLE_LEN}",
                preamble.section_tbl_off
            )));
        }

        // 2. Section table.
        let table_len = preamble
            .section_tbl_count
            .checked_mul(SECTION_DESC_LEN)
            .ok_or_else(|| RsmfError::structural("section table length overflow"))?;
        let table_end = preamble
            .section_tbl_off
            .checked_add(table_len)
            .ok_or_else(|| RsmfError::structural("section table end overflow"))?;
        if table_end > file_len {
            return Err(RsmfError::structural(
                "section table extends past source end".to_string(),
            ));
        }
        let table_bytes = reader.read_range(preamble.section_tbl_off, table_len)?;
        if (table_bytes.len() as u64) != table_len {
            return Err(RsmfError::structural(
                "RangeReader returned short section table".to_string(),
            ));
        }
        let mut sections = Vec::with_capacity(preamble.section_tbl_count as usize);
        for i in 0..preamble.section_tbl_count {
            let off = (i * SECTION_DESC_LEN) as usize;
            sections.push(SectionDescriptor::decode(
                &table_bytes[off..off + SECTION_DESC_LEN as usize],
            )?);
        }

        validate_section_table(&sections, file_len)?;

        // 3. Manifest.
        let manifest_section_idx = sections
            .iter()
            .position(|s| s.kind == SectionKind::Manifest)
            .ok_or_else(|| RsmfError::structural("no manifest section".to_string()))?;
        let manifest_section = &sections[manifest_section_idx];
        if manifest_section.offset != preamble.manifest_off
            || manifest_section.length != preamble.manifest_len
        {
            return Err(RsmfError::structural(
                "preamble manifest pointers disagree with section table".to_string(),
            ));
        }
        let manifest_bytes = reader.read_range(manifest_section.offset, manifest_section.length)?;
        if (manifest_bytes.len() as u64) != manifest_section.length {
            return Err(RsmfError::structural(
                "RangeReader returned short manifest".to_string(),
            ));
        }
        let manifest = Manifest::decode(&manifest_bytes)?;

        let canonical_section_idx = sections
            .iter()
            .position(|s| s.kind == SectionKind::CanonicalArena)
            .ok_or_else(|| RsmfError::structural("no canonical arena section".to_string()))?;
        let mut packed_section_indices: Vec<usize> = sections
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (s.kind == SectionKind::PackedArena).then_some(i))
            .collect();
        packed_section_indices.sort_unstable();
        let graph_section_idx = sections.iter().position(|s| s.kind == SectionKind::Graph);
        let assets_section_idx = sections.iter().position(|s| s.kind == SectionKind::Assets);

        validate_manifest(&manifest, &sections, &packed_section_indices)?;

        let tensor_by_name = manifest
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let asset_by_name = manifest
            .assets
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name.clone(), i))
            .collect();

        Ok(Self {
            reader,
            preamble,
            sections,
            manifest,
            canonical_section_idx,
            packed_section_indices,
            graph_section_idx,
            assets_section_idx,
            tensor_by_name,
            asset_by_name,
        })
    }

    /// Borrow the underlying range reader.
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Parsed preamble.
    #[must_use]
    pub fn preamble(&self) -> &Preamble {
        &self.preamble
    }

    /// Parsed section table.
    #[must_use]
    pub fn sections(&self) -> &[SectionDescriptor] {
        &self.sections
    }

    /// Parsed manifest. Fully in memory — no further fetches required.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Index adapter tensors (metadata-only, no payload fetch).
    pub fn adapters(&self) -> Result<AdapterIndex> {
        adapter_index_from_manifest(&self.manifest)
    }

    /// Fetch the canonical variant's bytes for a named tensor.
    pub fn fetch_tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let ti = *self
            .tensor_by_name
            .get(name)
            .ok_or_else(|| RsmfError::not_found(format!("tensor {name}")))?;
        let t = &self.manifest.tensors[ti];
        let v = self
            .manifest
            .variants
            .get(t.canonical_variant as usize)
            .ok_or_else(|| {
                RsmfError::structural(format!(
                    "tensor {name} references missing variant {}",
                    t.canonical_variant
                ))
            })?;
        self.fetch_variant_bytes(v)
    }

    /// Fetch a specific variant of a named tensor by global variant index.
    pub fn fetch_tensor_variant_bytes(&self, name: &str, variant_idx: u32) -> Result<Vec<u8>> {
        let ti = *self
            .tensor_by_name
            .get(name)
            .ok_or_else(|| RsmfError::not_found(format!("tensor {name}")))?;
        let t = &self.manifest.tensors[ti];
        if t.canonical_variant != variant_idx && !t.packed_variants.contains(&variant_idx) {
            return Err(RsmfError::not_found(format!(
                "tensor {name} does not own variant {variant_idx}"
            )));
        }
        let v = self
            .manifest
            .variants
            .get(variant_idx as usize)
            .ok_or_else(|| RsmfError::not_found(format!("variant index {variant_idx}")))?;
        self.fetch_variant_bytes(v)
    }

    /// Fetch a named asset's bytes.
    pub fn fetch_asset_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let idx = *self
            .asset_by_name
            .get(name)
            .ok_or_else(|| RsmfError::not_found(format!("asset {name}")))?;
        let a = &self.manifest.assets[idx];
        let section_idx = self
            .assets_section_idx
            .ok_or_else(|| RsmfError::structural("no assets section".to_string()))?;
        let section = &self.sections[section_idx];
        self.fetch_section_slice(section, a.offset, a.length)
    }

    /// Fetch a graph payload by its index in [`Manifest::graphs`].
    pub fn fetch_graph_bytes(&self, idx: usize) -> Result<Vec<u8>> {
        let g = self
            .manifest
            .graphs
            .get(idx)
            .ok_or_else(|| RsmfError::not_found(format!("graph index {idx}")))?;
        let section_idx = self
            .graph_section_idx
            .ok_or_else(|| RsmfError::structural("no graph section".to_string()))?;
        let section = &self.sections[section_idx];
        self.fetch_section_slice(section, g.offset, g.length)
    }

    fn fetch_variant_bytes(&self, v: &VariantDescriptor) -> Result<Vec<u8>> {
        let section = self.variant_section(v)?;
        self.fetch_section_slice(section, v.section_relative_offset, v.length)
    }

    fn fetch_section_slice(
        &self,
        section: &SectionDescriptor,
        relative_offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if section.is_compressed() {
            let compressed = self.reader.read_range(section.offset, section.length)?;
            let decompressed = decompress_zstd(&compressed, section.is_bit_shuffled())?;
            let end = relative_offset
                .checked_add(length)
                .ok_or_else(|| RsmfError::structural("variant range overflow".to_string()))?;
            if end > decompressed.len() as u64 {
                return Err(RsmfError::structural(
                    "variant extends past decompressed section".to_string(),
                ));
            }
            Ok(decompressed[relative_offset as usize..end as usize].to_vec())
        } else {
            let absolute = section
                .offset
                .checked_add(relative_offset)
                .ok_or_else(|| RsmfError::structural("variant offset overflow".to_string()))?;
            let bytes = self.reader.read_range(absolute, length)?;
            if (bytes.len() as u64) != length {
                return Err(RsmfError::structural(
                    "RangeReader returned short variant".to_string(),
                ));
            }
            Ok(bytes)
        }
    }

    fn variant_section(&self, v: &VariantDescriptor) -> Result<&SectionDescriptor> {
        let section_idx = match v.section_kind {
            k if k == SectionKind::CanonicalArena.to_raw() as u8 => self.canonical_section_idx,
            k if k == SectionKind::PackedArena.to_raw() as u8 => {
                let i = v.section_index as usize;
                *self.packed_section_indices.get(i).ok_or_else(|| {
                    RsmfError::structural(format!(
                        "variant references missing packed arena index {i}"
                    ))
                })?
            }
            other => {
                return Err(RsmfError::structural(format!(
                    "variant section_kind {other} not a tensor arena"
                )));
            }
        };
        self.sections.get(section_idx).ok_or_else(|| {
            RsmfError::structural(format!(
                "variant references missing section index {section_idx}"
            ))
        })
    }
}
