//! Streaming writer that materialises an RSMF file without buffering
//! tensor, graph, or asset bytes in memory.
//!
//! Payloads flow through the writer via `Read` handles and land on disk
//! immediately; only descriptors (small) accumulate. This makes packing
//! multi-hundred-GB weight files tractable on machines with modest RAM —
//! the batch [`crate::RsmfWriter`] still requires every payload to fit
//! in memory simultaneously.
//!
//! # On-disk layout
//!
//! The streaming writer emits sections in a different on-disk order to
//! the batch writer (manifest comes **last** so it can reference
//! previously-written offsets):
//!
//! ```text
//! [ Preamble (64 bytes)                   ]  patched at finish()
//! [ Section Table (up to 4 × 64 bytes)    ]  patched at finish()
//! [ Canonical Arena (tensor bodies)       ]  streamed
//! [ Graph   section (optional)            ]  streamed
//! [ Assets  section (optional)            ]  streamed
//! [ Manifest                              ]  serialised last
//! ```
//!
//! The section table lists sections in ascending file-offset order. The
//! resulting file is fully valid per [`crate::RsmfFile::open`] and
//! byte-level-equivalent in semantics, just not byte-identical to the
//! batch writer's output (which puts `Manifest` first).
//!
//! # State machine
//!
//! Callers move through these section states in order:
//!
//! ```text
//! Canonical ──(stream_canonical_tensor)──> Canonical
//!            ──(stream_graph)──────────── > Graph
//!            ──(stream_asset)──────────── > Asset
//! Graph     ──(stream_graph)──────────── > Graph
//!            ──(stream_asset)──────────── > Asset
//! Asset     ──(stream_asset)──────────── > Asset
//! Any other transition is rejected with a Structural error.
//! ```
//!
//! # MVP scope (this crate version)
//!
//! Currently supported:
//! - Multiple canonical tensors with raw row-major bodies.
//! - Multiple graphs (ONNX / ORT / Other).
//! - Multiple named assets.
//! - Global manifest metadata.
//!
//! Not yet supported:
//! - Packed / quantized variants.
//! - Compression on any section (`zstd`).
//! - Tensor-level, variant-level, graph-level, or asset-level metadata.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use blake3::Hasher;

use crate::checksum::{CHECKSUM_LEN, PREAMBLE_CHECKSUM_LEN};
use crate::error::{Result, RsmfError};
use crate::manifest::{AssetDescriptor, GraphDescriptor, GraphKind, MANIFEST_VERSION, Manifest};
use crate::preamble::{FORMAT_MAJOR, FORMAT_MINOR, MAGIC, PREAMBLE_LEN, Preamble};
use crate::section::{SECTION_DESC_LEN, SectionDescriptor, SectionKind};
use crate::tensor::descriptor::TensorDescriptor;
use crate::tensor::dtype::{LogicalDtype, StorageDtype};
use crate::tensor::variant::{EncodingKind, LayoutTag, TargetTag, VariantDescriptor, VariantMeta};

const CANONICAL_ARENA_ALIGN: u64 = 64;
const GRAPH_SECTION_ALIGN: u64 = 8;
const ASSETS_SECTION_ALIGN: u64 = 8;
const MANIFEST_ALIGN: u64 = 8;
const VARIANT_ALIGN: u32 = 64;
const GRAPH_BODY_ALIGN: u64 = 8;
const ASSET_BODY_ALIGN: u64 = 8;

/// Reserved slots in the section table: CanonicalArena, Graph, Assets,
/// Manifest. Anything we don't actually emit stays zero-filled; the
/// preamble's `section_tbl_count` tracks the true count.
const MAX_SECTIONS: u16 = 4;

/// The streaming writer cursor. Section writes are strictly ordered:
/// canonical first, then graph, then assets, then manifest at finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cursor {
    Canonical,
    Graph,
    Asset,
    Finished,
}

/// Partial section state kept while the cursor is inside one.
struct SectionInProgress {
    file_off: u64,
    bytes_written: u64,
    hasher: Hasher,
}

impl SectionInProgress {
    fn new(file_off: u64) -> Self {
        Self {
            file_off,
            bytes_written: 0,
            hasher: Hasher::new(),
        }
    }

    fn finalize(self) -> (u64, u64, [u8; CHECKSUM_LEN]) {
        let mut cs = [0u8; CHECKSUM_LEN];
        cs.copy_from_slice(&self.hasher.finalize().as_bytes()[..CHECKSUM_LEN]);
        (self.file_off, self.bytes_written, cs)
    }
}

/// Closed section summary stashed for the section-table patch at finish.
struct ClosedSection {
    kind: SectionKind,
    align: u16,
    file_off: u64,
    length: u64,
    checksum: [u8; CHECKSUM_LEN],
}

/// RSMF writer that streams payloads straight to disk.
pub struct StreamingRsmfWriter {
    file: File,
    path: PathBuf,
    cursor: Cursor,
    metadata: Vec<(String, String)>,
    tensor_descriptors: Vec<TensorDescriptor>,
    variant_descriptors: Vec<VariantDescriptor>,
    graph_descriptors: Vec<GraphDescriptor>,
    asset_descriptors: Vec<AssetDescriptor>,
    tensor_names: Vec<String>,
    asset_names: Vec<String>,
    active: Option<SectionInProgress>,
    closed: Vec<ClosedSection>,
}

impl StreamingRsmfWriter {
    /// Create a new streaming writer at `path`. The file is truncated
    /// if it exists.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path).map_err(|e| RsmfError::IoWithPath {
            path: path.clone(),
            source: e,
        })?;

        let table_end = PREAMBLE_LEN + u64::from(MAX_SECTIONS) * SECTION_DESC_LEN;
        let canonical_arena_file_off = align_up(table_end, CANONICAL_ARENA_ALIGN);

        let mut writer = Self {
            file,
            path,
            cursor: Cursor::Canonical,
            metadata: Vec::new(),
            tensor_descriptors: Vec::new(),
            variant_descriptors: Vec::new(),
            graph_descriptors: Vec::new(),
            asset_descriptors: Vec::new(),
            tensor_names: Vec::new(),
            asset_names: Vec::new(),
            active: Some(SectionInProgress::new(canonical_arena_file_off)),
            closed: Vec::new(),
        };

        // Preamble + section-table slots stay as a zero-filled hole
        // until finish() patches them. Bytes between the table end and
        // the arena start are explicit zeros so readers see a fully
        // populated file on return.
        let pad_len = canonical_arena_file_off - table_end;
        writer.seek_to(table_end)?;
        if pad_len > 0 {
            writer.write_all(&vec![0u8; pad_len as usize])?;
        }

        Ok(writer)
    }

    /// Append a global metadata key/value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// Stream one canonical tensor's bytes into the output file.
    ///
    /// Must be called before any `stream_graph` or `stream_asset`. The
    /// tensor is aligned to [`VARIANT_ALIGN`] inside the arena, so
    /// several zero-filled bytes may be written before the first byte
    /// from `reader`.
    pub fn stream_canonical_tensor<R: Read>(
        &mut self,
        name: impl Into<String>,
        dtype: LogicalDtype,
        shape: Vec<u64>,
        reader: R,
    ) -> Result<()> {
        if self.cursor != Cursor::Canonical {
            return Err(RsmfError::structural(
                "streaming writer: tensors must be streamed before any graph or asset".to_string(),
            ));
        }
        let name: String = name.into();
        if self.tensor_names.iter().any(|n| n == &name) {
            return Err(RsmfError::structural(format!(
                "duplicate tensor name {name}"
            )));
        }

        // Current section is the canonical arena. Align and stream.
        let (rel_off, variant_len, variant_checksum) =
            self.append_body_to_active(u64::from(VARIANT_ALIGN), reader)?;

        let variant_idx = self.variant_descriptors.len() as u32;
        self.variant_descriptors.push(VariantDescriptor {
            target: TargetTag::Canonical,
            encoding: EncodingKind::Raw,
            storage_dtype: StorageDtype::Logical(dtype),
            layout: LayoutTag::RowMajor,
            alignment: VARIANT_ALIGN,
            section_relative_offset: rel_off,
            length: variant_len,
            checksum: variant_checksum,
            section_kind: SectionKind::CanonicalArena.to_raw() as u8,
            section_index: 0,
            meta: VariantMeta::default(),
        });
        self.tensor_descriptors.push(TensorDescriptor {
            name: name.clone(),
            dtype,
            shape,
            canonical_variant: variant_idx,
            packed_variants: Vec::new(),
            shard_id: 0,
            metadata: Vec::new(),
        });
        self.tensor_names.push(name);
        Ok(())
    }

    /// Stream one graph payload (ONNX, ORT, or Other) into the graph
    /// section. Transitions the cursor into Graph state on first call;
    /// rejected after any `stream_asset` has been made.
    pub fn stream_graph<R: Read>(&mut self, kind: GraphKind, reader: R) -> Result<()> {
        match self.cursor {
            Cursor::Canonical => self.transition_to(Cursor::Graph)?,
            Cursor::Graph => {}
            Cursor::Asset | Cursor::Finished => {
                return Err(RsmfError::structural(
                    "streaming writer: graphs must be streamed before any asset".to_string(),
                ));
            }
        }

        let (rel_off, length, checksum) = self.append_body_to_active(GRAPH_BODY_ALIGN, reader)?;
        self.graph_descriptors.push(GraphDescriptor {
            kind,
            alignment: GRAPH_BODY_ALIGN as u32,
            offset: rel_off,
            length,
            checksum,
            metadata: Vec::new(),
        });
        Ok(())
    }

    /// Stream one named asset into the assets section. Transitions the
    /// cursor into Asset state on first call.
    pub fn stream_asset<R: Read>(&mut self, name: impl Into<String>, reader: R) -> Result<()> {
        match self.cursor {
            Cursor::Canonical | Cursor::Graph => self.transition_to(Cursor::Asset)?,
            Cursor::Asset => {}
            Cursor::Finished => {
                return Err(RsmfError::structural(
                    "streaming writer: already finished".to_string(),
                ));
            }
        }
        let name: String = name.into();
        if self.asset_names.iter().any(|n| n == &name) {
            return Err(RsmfError::structural(format!(
                "duplicate asset name {name}"
            )));
        }

        let (rel_off, length, checksum) = self.append_body_to_active(ASSET_BODY_ALIGN, reader)?;
        self.asset_descriptors.push(AssetDescriptor {
            name: name.clone(),
            alignment: ASSET_BODY_ALIGN as u32,
            offset: rel_off,
            length,
            checksum,
            metadata: Vec::new(),
        });
        self.asset_names.push(name);
        Ok(())
    }

    /// Serialise the manifest and patch the preamble + section table.
    /// Consumes the writer; the file is fully valid on return.
    pub fn finish(mut self) -> Result<()> {
        if self.tensor_descriptors.is_empty() {
            return Err(RsmfError::structural(
                "streaming writer: at least one tensor must be streamed".to_string(),
            ));
        }

        // Close whichever section is currently open.
        self.close_active_section()?;
        self.cursor = Cursor::Finished;

        // Align before the manifest body.
        let end_of_last_section = self.current_file_offset()?;
        let manifest_file_off = align_up(end_of_last_section, MANIFEST_ALIGN);
        if manifest_file_off > end_of_last_section {
            let pad_len = (manifest_file_off - end_of_last_section) as usize;
            self.write_all(&vec![0u8; pad_len])?;
        }

        // Encode manifest.
        let manifest = Manifest {
            version: MANIFEST_VERSION,
            metadata: self.metadata.clone(),
            tensors: self.tensor_descriptors.clone(),
            variants: self.variant_descriptors.clone(),
            graphs: self.graph_descriptors.clone(),
            assets: self.asset_descriptors.clone(),
        };
        let manifest_bytes = manifest.encode()?;
        let manifest_len = manifest_bytes.len() as u64;
        self.write_all(&manifest_bytes)?;

        let mut manifest_checksum = [0u8; CHECKSUM_LEN];
        manifest_checksum
            .copy_from_slice(&blake3::hash(&manifest_bytes).as_bytes()[..CHECKSUM_LEN]);

        // Build the full section list. Order must be ascending by offset.
        let mut sections: Vec<SectionDescriptor> = Vec::with_capacity(self.closed.len() + 1);
        for c in &self.closed {
            sections.push(SectionDescriptor {
                kind: c.kind,
                align: c.align,
                flags: 0,
                offset: c.file_off,
                length: c.length,
                checksum: c.checksum,
            });
        }
        sections.push(SectionDescriptor {
            kind: SectionKind::Manifest,
            align: MANIFEST_ALIGN as u16,
            flags: 0,
            offset: manifest_file_off,
            length: manifest_len,
            checksum: manifest_checksum,
        });
        sections.sort_by_key(|s| s.offset);

        let used_sections = sections.len() as u16;
        assert!(used_sections <= MAX_SECTIONS);

        // Patch the section table, then the preamble. Unused reserved
        // slots in the table stay zero.
        self.seek_to(PREAMBLE_LEN)?;
        for s in &sections {
            self.write_all(&s.encode())?;
        }
        for _ in used_sections..MAX_SECTIONS {
            self.write_all(&[0u8; SECTION_DESC_LEN as usize])?;
        }

        let preamble = Preamble {
            magic: MAGIC,
            major: FORMAT_MAJOR,
            minor: FORMAT_MINOR,
            flags: 0,
            header_len: PREAMBLE_LEN,
            section_tbl_off: PREAMBLE_LEN,
            section_tbl_count: u64::from(used_sections),
            manifest_off: manifest_file_off,
            manifest_len,
            preamble_checksum: [0u8; PREAMBLE_CHECKSUM_LEN],
        };
        self.seek_to(0)?;
        self.write_all(&preamble.encode())?;

        self.file.flush().map_err(|e| RsmfError::IoWithPath {
            path: self.path.clone(),
            source: e,
        })?;
        self.file.sync_data().map_err(|e| RsmfError::IoWithPath {
            path: self.path.clone(),
            source: e,
        })?;
        Ok(())
    }

    // ---- internal helpers ----

    /// Align inside the active section, stream the reader to disk, and
    /// return `(relative_offset, length, body_checksum)` where:
    /// - `relative_offset` is the start offset inside the section,
    /// - `length` is the body length (excluding padding),
    /// - `body_checksum` is BLAKE3 over the body bytes only (used as
    ///   the per-variant / per-graph / per-asset checksum).
    fn append_body_to_active<R: Read>(
        &mut self,
        body_align: u64,
        mut reader: R,
    ) -> Result<(u64, u64, [u8; CHECKSUM_LEN])> {
        let active = self.active.as_mut().ok_or_else(|| {
            RsmfError::structural("streaming writer: no active section".to_string())
        })?;

        // Align inside the section.
        let aligned_off = align_up(active.bytes_written, body_align);
        if aligned_off > active.bytes_written {
            let pad_len = (aligned_off - active.bytes_written) as usize;
            let zeros = vec![0u8; pad_len];
            self.file
                .write_all(&zeros)
                .map_err(|e| RsmfError::IoWithPath {
                    path: self.path.clone(),
                    source: e,
                })?;
            active.hasher.update(&zeros);
            active.bytes_written = aligned_off;
        }

        // Stream.
        let mut body_hasher = Hasher::new();
        let mut body_len: u64 = 0;
        let mut buf = [0u8; 65536];
        loop {
            let n = reader.read(&mut buf).map_err(|e| RsmfError::IoWithPath {
                path: self.path.clone(),
                source: e,
            })?;
            if n == 0 {
                break;
            }
            self.file
                .write_all(&buf[..n])
                .map_err(|e| RsmfError::IoWithPath {
                    path: self.path.clone(),
                    source: e,
                })?;
            active.hasher.update(&buf[..n]);
            body_hasher.update(&buf[..n]);
            body_len += n as u64;
        }
        active.bytes_written += body_len;

        let mut body_checksum = [0u8; CHECKSUM_LEN];
        body_checksum.copy_from_slice(&body_hasher.finalize().as_bytes()[..CHECKSUM_LEN]);
        Ok((aligned_off, body_len, body_checksum))
    }

    /// Close the currently active section, recording it in `closed`.
    fn close_active_section(&mut self) -> Result<()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let kind = match self.cursor {
            Cursor::Canonical => SectionKind::CanonicalArena,
            Cursor::Graph => SectionKind::Graph,
            Cursor::Asset => SectionKind::Assets,
            Cursor::Finished => {
                return Err(RsmfError::structural(
                    "streaming writer: close called after finish".to_string(),
                ));
            }
        };
        let align: u16 = match kind {
            SectionKind::CanonicalArena => CANONICAL_ARENA_ALIGN as u16,
            SectionKind::Graph => GRAPH_SECTION_ALIGN as u16,
            SectionKind::Assets => ASSETS_SECTION_ALIGN as u16,
            _ => unreachable!("close called for non-arena/graph/asset section"),
        };
        // Graph / Asset sections disallow zero length per section-table
        // validator; skip recording them if they're empty (the cursor
        // transitioned into them but nothing was actually streamed).
        if active.bytes_written == 0 && !matches!(kind, SectionKind::CanonicalArena) {
            return Ok(());
        }
        let (file_off, length, checksum) = active.finalize();
        self.closed.push(ClosedSection {
            kind,
            align,
            file_off,
            length,
            checksum,
        });
        Ok(())
    }

    /// Move the cursor to `next`, closing the current section and
    /// opening a fresh one at the appropriate alignment.
    fn transition_to(&mut self, next: Cursor) -> Result<()> {
        debug_assert!(next != self.cursor);
        self.close_active_section()?;
        let next_section_align = match next {
            Cursor::Canonical => CANONICAL_ARENA_ALIGN,
            Cursor::Graph => GRAPH_SECTION_ALIGN,
            Cursor::Asset => ASSETS_SECTION_ALIGN,
            Cursor::Finished => {
                return Err(RsmfError::structural(
                    "streaming writer: cannot transition to Finished directly".to_string(),
                ));
            }
        };
        let here = self.current_file_offset()?;
        let aligned = align_up(here, next_section_align);
        if aligned > here {
            let pad_len = (aligned - here) as usize;
            self.write_all(&vec![0u8; pad_len])?;
        }
        self.active = Some(SectionInProgress::new(aligned));
        self.cursor = next;
        Ok(())
    }

    fn current_file_offset(&mut self) -> Result<u64> {
        self.file
            .stream_position()
            .map_err(|e| RsmfError::IoWithPath {
                path: self.path.clone(),
                source: e,
            })
    }

    fn seek_to(&mut self, off: u64) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| RsmfError::IoWithPath {
                path: self.path.clone(),
                source: e,
            })?;
        Ok(())
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.file
            .write_all(bytes)
            .map_err(|e| RsmfError::IoWithPath {
                path: self.path.clone(),
                source: e,
            })
    }
}

fn align_up(off: u64, align: u64) -> u64 {
    if align <= 1 {
        return off;
    }
    (off + align - 1) & !(align - 1)
}
