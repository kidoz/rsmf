//! Streaming writer that materialises an RSMF file without buffering
//! tensor bytes in memory.
//!
//! Tensor payloads flow through the writer via a `Read` handle and land
//! on disk immediately; only descriptors (small) accumulate. This makes
//! packing multi-hundred-GB weight files tractable on machines with
//! modest RAM — the batch [`crate::RsmfWriter`] still requires every
//! tensor's canonical bytes to fit in memory simultaneously.
//!
//! # On-disk layout
//!
//! The streaming writer emits sections in a different on-disk order to
//! the batch writer (manifest comes **last** so it can reference
//! previously-written offsets):
//!
//! ```text
//! [ Preamble (64 bytes)            ]  patched at finish()
//! [ Section Table (N × 64 bytes)   ]  patched at finish()
//! [ Canonical Arena (tensor bytes) ]  streamed
//! [ Manifest                       ]  serialised last
//! ```
//!
//! The section table lists sections in ascending file-offset order, so
//! for this layout it contains `[CanonicalArena, Manifest]`. The
//! resulting file is fully valid per [`crate::RsmfFile::open`] and
//! byte-level-equivalent in semantics, just not byte-identical to the
//! batch writer's output (which puts `Manifest` first).
//!
//! # MVP scope (this crate version)
//!
//! Currently supported:
//!
//! - Multiple canonical tensors with raw row-major bodies.
//! - Global manifest metadata via [`StreamingRsmfWriter::with_metadata`].
//!
//! Not yet supported (adding any of these returns `Unsupported`):
//!
//! - Packed / quantized variants.
//! - Compression (`compression` feature gate stays off in the writer).
//! - Graph section.
//! - Assets section.
//! - Tensor-level metadata and variant-level metadata.
//!
//! These are all orthogonal features that can be added later without
//! changing the file format or the read path.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use blake3::Hasher;

use crate::checksum::{CHECKSUM_LEN, PREAMBLE_CHECKSUM_LEN};
use crate::error::{Result, RsmfError};
use crate::manifest::{MANIFEST_VERSION, Manifest};
use crate::preamble::{FORMAT_MAJOR, FORMAT_MINOR, MAGIC, PREAMBLE_LEN, Preamble};
use crate::section::{SECTION_DESC_LEN, SectionDescriptor, SectionKind};
use crate::tensor::descriptor::TensorDescriptor;
use crate::tensor::dtype::{LogicalDtype, StorageDtype};
use crate::tensor::variant::{EncodingKind, LayoutTag, TargetTag, VariantDescriptor, VariantMeta};

const CANONICAL_ARENA_ALIGN: u64 = 64;
const MANIFEST_ALIGN: u64 = 8;
const VARIANT_ALIGN: u32 = 64;

// Reserved slots in the section table. For this MVP we only ever emit
// two (CanonicalArena + Manifest), but reserving a small fixed budget
// lets future additions (Graph / Assets) land without re-jiggling the
// on-disk offsets.
const MAX_SECTIONS: u16 = 4;

/// RSMF writer that streams tensor bytes straight to disk.
///
/// Construction reserves space for the preamble and section table,
/// then positions the cursor at the start of the canonical arena.
/// Each [`StreamingRsmfWriter::stream_canonical_tensor`] call appends
/// one tensor's bytes. [`StreamingRsmfWriter::finish`] serialises the
/// manifest, patches the section table, patches the preamble, and
/// returns ownership of the output path.
pub struct StreamingRsmfWriter {
    file: File,
    path: PathBuf,
    metadata: Vec<(String, String)>,
    tensor_descriptors: Vec<TensorDescriptor>,
    variant_descriptors: Vec<VariantDescriptor>,
    tensor_names: Vec<String>,
    canonical_arena_file_off: u64,
    canonical_arena_write_off: u64,
    canonical_hasher: Hasher,
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
            metadata: Vec::new(),
            tensor_descriptors: Vec::new(),
            variant_descriptors: Vec::new(),
            tensor_names: Vec::new(),
            canonical_arena_file_off,
            canonical_arena_write_off: 0,
            canonical_hasher: Hasher::new(),
        };

        // Seek to the start of the canonical arena. The preamble and
        // section-table slots stay as a zero-filled hole until finish()
        // patches them. Any bytes between the table end and the arena
        // start are explicit zeros so readers see a fully populated
        // file on return.
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
    /// The reader is drained; callers that supply a fixed-length source
    /// should make sure its `Read` impl returns `Ok(0)` at EOF. The
    /// tensor is aligned to [`VARIANT_ALIGN`] inside the arena, so
    /// several zero-filled bytes may be written before the first byte
    /// from `reader`.
    ///
    /// Returns `RsmfError::Structural` if a tensor with the same name
    /// was already streamed, or if the reader yields an error.
    pub fn stream_canonical_tensor<R: Read>(
        &mut self,
        name: impl Into<String>,
        dtype: LogicalDtype,
        shape: Vec<u64>,
        mut reader: R,
    ) -> Result<()> {
        let name: String = name.into();
        if self.tensor_names.iter().any(|n| n == &name) {
            return Err(RsmfError::structural(format!(
                "duplicate tensor name {name}"
            )));
        }

        // Align the tensor start inside the arena.
        let aligned_off = align_up(self.canonical_arena_write_off, u64::from(VARIANT_ALIGN));
        if aligned_off > self.canonical_arena_write_off {
            let pad_len = (aligned_off - self.canonical_arena_write_off) as usize;
            let zeros = vec![0u8; pad_len];
            self.write_all(&zeros)?;
            self.canonical_hasher.update(&zeros);
            self.canonical_arena_write_off = aligned_off;
        }

        // Stream the reader into the file, updating both the section
        // hash (covers the whole arena) and a per-variant hash.
        let mut variant_hasher = Hasher::new();
        let mut variant_len: u64 = 0;
        let mut buf = [0u8; 65536];
        loop {
            let n = reader.read(&mut buf).map_err(|e| RsmfError::IoWithPath {
                path: self.path.clone(),
                source: e,
            })?;
            if n == 0 {
                break;
            }
            self.write_all(&buf[..n])?;
            self.canonical_hasher.update(&buf[..n]);
            variant_hasher.update(&buf[..n]);
            variant_len += n as u64;
        }
        self.canonical_arena_write_off += variant_len;

        // Record descriptors.
        let variant_idx = self.variant_descriptors.len() as u32;
        let mut variant_checksum = [0u8; CHECKSUM_LEN];
        variant_checksum.copy_from_slice(&variant_hasher.finalize().as_bytes()[..CHECKSUM_LEN]);
        self.variant_descriptors.push(VariantDescriptor {
            target: TargetTag::Canonical,
            encoding: EncodingKind::Raw,
            storage_dtype: StorageDtype::Logical(dtype),
            layout: LayoutTag::RowMajor,
            alignment: VARIANT_ALIGN,
            section_relative_offset: aligned_off,
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

    /// Serialise the manifest and patch the preamble + section table.
    /// Consumes the writer; the file is fully valid on return.
    pub fn finish(mut self) -> Result<()> {
        if self.tensor_descriptors.is_empty() {
            return Err(RsmfError::structural(
                "streaming writer: at least one tensor must be streamed".to_string(),
            ));
        }

        let canonical_arena_len = self.canonical_arena_write_off;
        let mut canonical_checksum = [0u8; CHECKSUM_LEN];
        canonical_checksum
            .copy_from_slice(&self.canonical_hasher.finalize().as_bytes()[..CHECKSUM_LEN]);

        // Align before the manifest body.
        let canonical_end = self
            .canonical_arena_file_off
            .checked_add(canonical_arena_len)
            .ok_or_else(|| RsmfError::structural("canonical arena end overflow".to_string()))?;
        let manifest_file_off = align_up(canonical_end, MANIFEST_ALIGN);
        if manifest_file_off > canonical_end {
            let pad_len = (manifest_file_off - canonical_end) as usize;
            self.write_all(&vec![0u8; pad_len])?;
        }

        // Encode the manifest from accumulated descriptors.
        let manifest = Manifest {
            version: MANIFEST_VERSION,
            metadata: self.metadata.clone(),
            tensors: self.tensor_descriptors.clone(),
            variants: self.variant_descriptors.clone(),
            graphs: Vec::new(),
            assets: Vec::new(),
        };
        let manifest_bytes = manifest.encode()?;
        let manifest_len = manifest_bytes.len() as u64;
        self.write_all(&manifest_bytes)?;

        let mut manifest_checksum = [0u8; CHECKSUM_LEN];
        manifest_checksum
            .copy_from_slice(&blake3::hash(&manifest_bytes).as_bytes()[..CHECKSUM_LEN]);

        // Ordered by ascending file offset — the section-table
        // validator requires it.
        let sections = [
            SectionDescriptor {
                kind: SectionKind::CanonicalArena,
                align: CANONICAL_ARENA_ALIGN as u16,
                flags: 0,
                offset: self.canonical_arena_file_off,
                length: canonical_arena_len,
                checksum: canonical_checksum,
            },
            SectionDescriptor {
                kind: SectionKind::Manifest,
                align: MANIFEST_ALIGN as u16,
                flags: 0,
                offset: manifest_file_off,
                length: manifest_len,
                checksum: manifest_checksum,
            },
        ];
        let used_sections = sections.len() as u16;
        assert!(used_sections <= MAX_SECTIONS);

        // Patch section table: always at offset PREAMBLE_LEN.
        self.seek_to(PREAMBLE_LEN)?;
        for s in &sections {
            self.write_all(&s.encode())?;
        }
        // Remaining reserved slots stay zero (the file was already
        // zero-padded during `new`, but re-write to be explicit in
        // case the OS lost that state).
        for _ in used_sections..MAX_SECTIONS {
            self.write_all(&[0u8; SECTION_DESC_LEN as usize])?;
        }

        // Patch preamble at offset 0.
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
