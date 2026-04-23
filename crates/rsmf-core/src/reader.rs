//! `RsmfFile` — mmap-backed reader.
//!
//! Opening a file runs a full structural validation pass (preamble, section
//! table, manifest, cross-references). Optional full checksum verification is
//! exposed through [`RsmfFile::full_verify`] and [`crate::validator::Validator`].

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;

use memmap2::Mmap;

#[cfg(feature = "tracing")]
use tracing::{info, info_span};

use crate::checksum::digest_128;
use crate::error::{Result, RsmfError};
use crate::manifest::{GraphKind, Manifest};
use crate::preamble::{PREAMBLE_LEN, Preamble};
use crate::section::{SECTION_DESC_LEN, SectionDescriptor, SectionKind};
use crate::selection::{Capabilities, ExecutionMode, TensorPlan, select_variants};
use crate::tensor::variant::VariantDescriptor;
use crate::tensor::view::TensorView;

/// Human-readable summary of a file. Useful for `rsmf inspect` output.
#[derive(Debug, Clone)]
pub struct ManifestSummary {
    /// Path the file was opened from, if any.
    pub path: Option<PathBuf>,
    /// Size of the file in bytes.
    pub file_size: u64,
    /// Format major version.
    pub format_major: u16,
    /// Format minor version.
    pub format_minor: u16,
    /// Number of sections.
    pub section_count: u64,
    /// Number of tensors.
    pub tensor_count: usize,
    /// Number of variant descriptors.
    pub variant_count: usize,
    /// Graph kinds found in the file.
    pub graph_kinds: Vec<GraphKind>,
    /// Number of assets.
    pub asset_count: usize,
    /// Global metadata map (cloned).
    pub metadata: Vec<(String, String)>,
}

/// Opaque graph payload handle.
#[derive(Debug, Clone)]
pub struct GraphPayload<'a> {
    /// Graph kind.
    pub kind: GraphKind,
    /// Graph bytes (mmap-backed or decompressed cache).
    pub bytes: &'a [u8],
    /// Graph metadata from the descriptor.
    pub metadata: &'a [(String, String)],
}

/// Asset reference returned by [`RsmfFile::asset`].
#[derive(Debug, Clone)]
pub struct AssetRef<'a> {
    /// Asset name.
    pub name: &'a str,
    /// Asset bytes (mmap-backed or decompressed cache).
    pub bytes: &'a [u8],
    /// Asset metadata from the descriptor.
    pub metadata: &'a [(String, String)],
}

/// A validated, mmap-backed RSMF file.
#[derive(Debug)]
pub struct RsmfFile {
    master_mmap: Arc<Mmap>,
    shard_mmaps: HashMap<u64, Arc<Mmap>>,
    path: Option<PathBuf>,
    preamble: Preamble,
    sections: Vec<SectionDescriptor>,
    manifest: Manifest,
    canonical_section_idx: usize,
    packed_section_indices: Vec<usize>,
    graph_section_idx: Option<usize>,
    assets_section_idx: Option<usize>,
    tensor_by_name: HashMap<String, usize>,
    asset_by_name: HashMap<String, usize>,
    variant_to_tensor: Vec<usize>,
    /// Cached decompressed canonical arena.
    decompressed_canonical_arena: OnceLock<Vec<u8>>,
    /// Cached decompressed packed arenas (by packed arena index).
    decompressed_packed_arenas: Vec<OnceLock<Vec<u8>>>,
    /// Cached decompressed graph payload.
    decompressed_graph: OnceLock<Vec<u8>>,
    /// Cached decompressed assets payload.
    decompressed_assets: OnceLock<Vec<u8>>,
}

impl RsmfFile {
    /// Open and validate an RSMF file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        #[cfg(feature = "tracing")]
        let _span = info_span!("RsmfFile::open", path = ?path).entered();

        let file = File::open(path).map_err(|e| RsmfError::IoWithPath {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| RsmfError::IoWithPath {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::from_mmap(mmap, Some(path.to_path_buf()))
    }

    /// Build from an already-materialised `Mmap`. Used mostly by tests.
    pub fn from_mmap(mmap: Mmap, path: Option<PathBuf>) -> Result<Self> {
        let mmap = Arc::new(mmap);
        let bytes = &mmap[..];
        if (bytes.len() as u64) < PREAMBLE_LEN {
            return Err(RsmfError::structural(format!(
                "file too short for preamble: {} bytes",
                bytes.len()
            )));
        }
        let preamble = Preamble::decode(&bytes[..PREAMBLE_LEN as usize])?;
        if preamble.section_tbl_off != PREAMBLE_LEN {
            return Err(RsmfError::structural(format!(
                "preamble.section_tbl_off {} must equal preamble length {PREAMBLE_LEN}",
                preamble.section_tbl_off
            )));
        }
        let table_len = preamble
            .section_tbl_count
            .checked_mul(SECTION_DESC_LEN)
            .ok_or_else(|| RsmfError::structural("section table length overflow"))?;
        let table_end = preamble
            .section_tbl_off
            .checked_add(table_len)
            .ok_or_else(|| RsmfError::structural("section table end overflow"))?;
        if table_end > bytes.len() as u64 {
            return Err(RsmfError::structural(
                "section table extends past file end".to_string(),
            ));
        }

        let mut sections = Vec::with_capacity(preamble.section_tbl_count as usize);
        for i in 0..preamble.section_tbl_count {
            let off = (preamble.section_tbl_off + i * SECTION_DESC_LEN) as usize;
            sections.push(SectionDescriptor::decode(
                &bytes[off..off + SECTION_DESC_LEN as usize],
            )?);
        }

        validate_section_table(&sections, bytes.len() as u64)?;

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
        let manifest_bytes = &bytes[manifest_section.offset as usize
            ..(manifest_section.offset + manifest_section.length) as usize];
        let manifest = Manifest::decode(manifest_bytes)?;

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

        let mut variant_to_tensor = vec![0usize; manifest.variants.len()];
        for (ti, t) in manifest.tensors.iter().enumerate() {
            if let Some(slot) = variant_to_tensor.get_mut(t.canonical_variant as usize) {
                *slot = ti;
            } else {
                return Err(RsmfError::structural(format!(
                    "tensor {} has invalid canonical variant index {}",
                    t.name, t.canonical_variant
                )));
            }
            for &pi in &t.packed_variants {
                if let Some(slot) = variant_to_tensor.get_mut(pi as usize) {
                    *slot = ti;
                } else {
                    return Err(RsmfError::structural(format!(
                        "tensor {} has invalid packed variant index {}",
                        t.name, pi
                    )));
                }
            }
        }

        let mut decompressed_packed_arenas = Vec::with_capacity(packed_section_indices.len());
        for _ in 0..packed_section_indices.len() {
            decompressed_packed_arenas.push(OnceLock::new());
        }

        #[cfg(feature = "tracing")]
        info!(
            tensors = manifest.tensors.len(),
            variants = manifest.variants.len(),
            graphs = manifest.graphs.len(),
            assets = manifest.assets.len(),
            "RSMF manifest decoded successfully"
        );

        Ok(Self {
            master_mmap: mmap,
            shard_mmaps: HashMap::new(),
            path,
            preamble,
            sections,
            manifest,
            canonical_section_idx,
            packed_section_indices,
            graph_section_idx,
            assets_section_idx,
            tensor_by_name,
            asset_by_name,
            variant_to_tensor,
            decompressed_canonical_arena: OnceLock::new(),
            decompressed_packed_arenas,
            decompressed_graph: OnceLock::new(),
            decompressed_assets: OnceLock::new(),
        })
    }

    /// Attach a physical shard file to this reader.
    pub fn with_shard(mut self, shard_id: u64, mmap: Mmap) -> Self {
        #[cfg(feature = "tracing")]
        info!(shard_id, size = mmap.len(), "attaching external shard");
        self.shard_mmaps.insert(shard_id, Arc::new(mmap));
        self
    }

    /// Return a reference to the raw bytes of the master mapping.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.master_mmap[..]
    }

    /// Return the parsed preamble.
    #[must_use]
    pub fn preamble(&self) -> &Preamble {
        &self.preamble
    }

    /// Return the parsed section table.
    #[must_use]
    pub fn sections(&self) -> &[SectionDescriptor] {
        &self.sections
    }

    /// Return the parsed manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Index the file's adapter tensors (LoRA / DoRA / IA³).
    ///
    /// Walks tensors carrying `adapter.*` metadata and groups them by
    /// `adapter.name`. Returns an empty index for files without adapter
    /// annotations. See `docs/CONVENTIONS.md` for the metadata convention.
    pub fn adapters(&self) -> Result<crate::adapter::AdapterIndex> {
        crate::adapter::adapter_index_from_manifest(&self.manifest)
    }

    /// Build a high-level summary.
    #[must_use]
    pub fn inspect(&self) -> ManifestSummary {
        ManifestSummary {
            path: self.path.clone(),
            file_size: self.master_mmap.len() as u64,
            format_major: self.preamble.major,
            format_minor: self.preamble.minor,
            section_count: self.preamble.section_tbl_count,
            tensor_count: self.manifest.tensors.len(),
            variant_count: self.manifest.variants.len(),
            graph_kinds: self.manifest.graphs.iter().map(|g| g.kind).collect(),
            asset_count: self.manifest.assets.len(),
            metadata: self.manifest.metadata.clone(),
        }
    }

    /// Compute a variant-selection plan for this file.
    pub fn select_variants(&self, mode: ExecutionMode, caps: &Capabilities) -> Result<TensorPlan> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("select_variants", mode = ?mode).entered();
        select_variants(&self.manifest, mode, caps)
    }

    /// Return a mmap-backed view over the canonical variant of the named tensor.
    pub fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
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
        let bytes = self.variant_bytes(v)?;
        Ok(TensorView {
            descriptor: t,
            bytes,
            encoding: v.encoding,
            layout: v.layout,
            meta: &v.meta,
            storage_dtype: v.storage_dtype,
        })
    }

    /// Return a mmap-backed view over a specific variant of the named tensor.
    ///
    /// `variant_idx` is a **global** index into `manifest.variants` and MUST
    /// be either the tensor's `canonical_variant` or one of its
    /// `packed_variants`. Passing a variant that belongs to a different
    /// tensor returns `RsmfError::NotFound` — the previous implementation
    /// silently returned that other tensor's bytes under this tensor's
    /// descriptor, producing reshape-mismatched NumPy arrays for the
    /// Python bindings.
    pub fn tensor_view_variant(&self, name: &str, variant_idx: u32) -> Result<TensorView<'_>> {
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
        let bytes = self.variant_bytes(v)?;
        Ok(TensorView {
            descriptor: t,
            bytes,
            encoding: v.encoding,
            layout: v.layout,
            meta: &v.meta,
            storage_dtype: v.storage_dtype,
        })
    }

    /// Return the raw bytes of any variant.
    ///
    /// When the owning tensor's `shard_id` is non-zero, the bytes are
    /// resolved out of the attached shard file (see [`Self::with_shard`]).
    /// Shards are raw **arena byte buffers**: the variant lives at
    /// `v.section_relative_offset..v.section_relative_offset + v.length`,
    /// not at the absolute section offset recorded in the master. The
    /// master's arena section still carries a descriptor (so the
    /// manifest round-trips and structural validation passes), but the
    /// bytes at that position are placeholder and will not be consulted
    /// once a shard is attached. Compressed shard arenas are not
    /// supported in v1 — the shard file must hold raw bytes.
    pub fn variant_bytes(&self, v: &VariantDescriptor) -> Result<&[u8]> {
        let section = self.variant_section(v)?;

        let v_idx = self
            .manifest
            .variants
            .iter()
            .position(|x| x == v)
            .unwrap_or(0);
        let t_idx = self.variant_to_tensor[v_idx];
        let shard_id = self.manifest.tensors[t_idx].shard_id;

        if shard_id != 0 {
            // Shard path. The shard file is a raw arena buffer — we index
            // directly by `v.section_relative_offset` without adding the
            // master's section offset, which is meaningless here.
            let shard = self
                .shard_mmaps
                .get(&shard_id)
                .ok_or_else(|| RsmfError::unsupported(format!("shard {shard_id} not loaded")))?;
            if section.is_compressed() {
                return Err(RsmfError::unsupported(format!(
                    "shard {shard_id} references a compressed arena section; \
                     compressed shard arenas are reserved for a future format version"
                )));
            }
            let start = v.section_relative_offset;
            let end = start.checked_add(v.length).ok_or_else(|| {
                RsmfError::structural("variant length overflow in shard".to_string())
            })?;
            if end > shard.len() as u64 {
                return Err(RsmfError::structural(format!(
                    "variant extends past shard {shard_id}: needs {end} bytes, shard has {}",
                    shard.len()
                )));
            }
            return Ok(&shard[start as usize..end as usize]);
        }

        // Master path.
        let mmap = &self.master_mmap;
        if section.is_compressed() {
            let cache = if section.kind == SectionKind::CanonicalArena {
                &self.decompressed_canonical_arena
            } else {
                let i = v.section_index as usize;
                &self.decompressed_packed_arenas[i]
            };
            let bytes = self.section_bytes_decompressed(section, cache, mmap)?;
            let start = v.section_relative_offset;
            let end = start + v.length;
            if end > bytes.len() as u64 {
                return Err(RsmfError::structural(
                    "variant extends past arena".to_string(),
                ));
            }
            Ok(&bytes[start as usize..end as usize])
        } else {
            let start = section.offset + v.section_relative_offset;
            let end = start + v.length;
            if end > mmap.len() as u64 {
                return Err(RsmfError::structural(
                    "variant extends past file".to_string(),
                ));
            }
            Ok(&mmap[start as usize..end as usize])
        }
    }

    /// Return all opaque graph payloads.
    pub fn graph_payloads(&self) -> Vec<GraphPayload<'_>> {
        let Some(idx) = self.graph_section_idx else {
            return Vec::new();
        };
        let section = &self.sections[idx];
        let Ok(bytes) =
            self.section_bytes_decompressed(section, &self.decompressed_graph, &self.master_mmap)
        else {
            return Vec::new();
        };

        self.manifest
            .graphs
            .iter()
            .filter_map(|g| {
                let start = g.offset as usize;
                let end = start + g.length as usize;
                if end > bytes.len() {
                    return None;
                }
                Some(GraphPayload {
                    kind: g.kind,
                    bytes: &bytes[start..end],
                    metadata: &g.metadata,
                })
            })
            .collect()
    }

    /// Return the named asset, if any.
    pub fn asset(&self, name: &str) -> Option<AssetRef<'_>> {
        let idx = *self.asset_by_name.get(name)?;
        let a = &self.manifest.assets[idx];
        let section_idx = self.assets_section_idx?;
        let section = &self.sections[section_idx];
        let bytes = self
            .section_bytes_decompressed(section, &self.decompressed_assets, &self.master_mmap)
            .ok()?;
        let start = a.offset as usize;
        let end = start + a.length as usize;
        if end > bytes.len() {
            return None;
        }
        Some(AssetRef {
            name: a.name.as_str(),
            bytes: &bytes[start..end],
            metadata: &a.metadata,
        })
    }

    /// Run the full checksum-verification pass using a specific thread pool.
    ///
    /// This allows controlling the parallelism used for hashing to prevent
    /// monopolizing the CPU when embedded in a larger application.
    #[cfg(feature = "parallel")]
    pub fn full_verify_with_pool(&self, pool: &rayon::ThreadPool) -> Result<()> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("full_verify_with_pool").entered();

        pool.install(|| {
            use rayon::prelude::*;

            let section_err = self.sections.par_iter().enumerate().find_map_any(|(i, s)| {
                let bytes = &self.master_mmap[s.offset as usize..(s.offset + s.length) as usize];
                if digest_128(bytes) != s.checksum {
                    Some(RsmfError::ChecksumMismatch {
                        kind: format!("section[{i}] {}", s.kind.name()),
                    })
                } else {
                    None
                }
            });
            if let Some(e) = section_err {
                return Err(e);
            }

            self.manifest.variants.par_iter().try_for_each(|v| {
                let bytes = self.variant_bytes(v)?;
                if digest_128(bytes) != v.checksum {
                    return Err(RsmfError::ChecksumMismatch {
                        kind: format!("variant target={}", v.target.name()),
                    });
                }
                Ok(())
            })
        })?;

        self.verify_graphs_and_assets()
    }

    /// Run the full checksum-verification pass.
    ///
    /// With the `parallel` cargo feature enabled the section and variant
    /// hash loops fan out over rayon's global thread pool — an N-core
    /// machine verifies a multi-GB file ~N× faster. The parallel path
    /// is numerically and error-wise identical to the sequential one;
    /// the first mismatch (by any thread) wins.
    pub fn full_verify(&self) -> Result<()> {
        #[cfg(feature = "tracing")]
        let _span = info_span!("full_verify").entered();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Section hashes are embarrassingly parallel: each section's
            // byte range is a disjoint subslice of the mmap.
            let section_err = self.sections.par_iter().enumerate().find_map_any(|(i, s)| {
                let bytes = &self.master_mmap[s.offset as usize..(s.offset + s.length) as usize];
                if digest_128(bytes) != s.checksum {
                    Some(RsmfError::ChecksumMismatch {
                        kind: format!("section[{i}] {}", s.kind.name()),
                    })
                } else {
                    None
                }
            });
            if let Some(e) = section_err {
                return Err(e);
            }

            // Variant hashes fan out too, but `variant_bytes` may return
            // `Result`; bubble the first error out.
            let variant_result: Result<()> = self.manifest.variants.par_iter().try_for_each(|v| {
                let bytes = self.variant_bytes(v)?;
                if digest_128(bytes) != v.checksum {
                    return Err(RsmfError::ChecksumMismatch {
                        kind: format!("variant target={}", v.target.name()),
                    });
                }
                Ok(())
            });
            variant_result?;
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (i, s) in self.sections.iter().enumerate() {
                let bytes = &self.master_mmap[s.offset as usize..(s.offset + s.length) as usize];
                let got = digest_128(bytes);
                if got != s.checksum {
                    return Err(RsmfError::ChecksumMismatch {
                        kind: format!("section[{i}] {}", s.kind.name()),
                    });
                }
            }
            for v in &self.manifest.variants {
                let bytes = self.variant_bytes(v)?;
                let got = digest_128(bytes);
                if got != v.checksum {
                    return Err(RsmfError::ChecksumMismatch {
                        kind: format!("variant target={}", v.target.name()),
                    });
                }
            }
        }

        // Graphs and assets stay sequential: typically <5 items per
        // file, so the rayon dispatch overhead would dominate.
        self.verify_graphs_and_assets()
    }

    fn verify_graphs_and_assets(&self) -> Result<()> {
        let payloads = self.graph_payloads();
        for (i, g) in self.manifest.graphs.iter().enumerate() {
            if let Some(payload) = payloads.get(i) {
                let got = digest_128(payload.bytes);
                if got != g.checksum {
                    return Err(RsmfError::ChecksumMismatch {
                        kind: format!("graph[{i}]"),
                    });
                }
            }
        }
        for a in &self.manifest.assets {
            let asset = self.asset(&a.name).ok_or_else(|| {
                RsmfError::structural(format!("asset descriptor {} missing its section", a.name))
            })?;
            let got = digest_128(asset.bytes);
            if got != a.checksum {
                return Err(RsmfError::ChecksumMismatch {
                    kind: format!("asset {}", a.name),
                });
            }
        }
        Ok(())
    }

    fn section_bytes_decompressed<'c>(
        &'c self,
        section: &SectionDescriptor,
        cache: &'c OnceLock<Vec<u8>>,
        mmap: &'c Mmap,
    ) -> Result<&'c [u8]> {
        let raw = &mmap[section.offset as usize..(section.offset + section.length) as usize];
        if !section.is_compressed() {
            return Ok(raw);
        }
        if let Some(cached) = cache.get() {
            return Ok(cached);
        }

        #[cfg(feature = "tracing")]
        let _span =
            info_span!("decompress_section", kind = ?section.kind, size = section.length).entered();

        let decompressed = decompress_zstd(raw, section.is_bit_shuffled())?;
        let _ = cache.set(decompressed);
        Ok(cache.get().expect("just set"))
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

pub(crate) fn validate_section_table(sections: &[SectionDescriptor], file_len: u64) -> Result<()> {
    let mut last_end: u64 = 0;
    for (i, s) in sections.iter().enumerate() {
        if s.length == 0 {
            return Err(RsmfError::structural(format!(
                "section[{i}] {} has length 0",
                s.kind.name()
            )));
        }
        let end = s
            .offset
            .checked_add(s.length)
            .ok_or_else(|| RsmfError::structural(format!("section[{i}] length overflow")))?;
        if end > file_len {
            return Err(RsmfError::structural(format!(
                "section[{i}] extends past file end"
            )));
        }
        let align = u64::from(s.align);
        if align != 0 && s.offset % align != 0 {
            return Err(RsmfError::structural(format!(
                "section[{i}] offset {} not aligned to {}",
                s.offset, align
            )));
        }
        if s.offset < last_end {
            return Err(RsmfError::structural(format!(
                "section[{i}] overlaps previous section"
            )));
        }
        last_end = end;
    }
    Ok(())
}

pub(crate) fn decompress_zstd(compressed: &[u8], bit_shuffled: bool) -> Result<Vec<u8>> {
    #[cfg(feature = "compression")]
    {
        let decompressed = zstd::decode_all(std::io::Cursor::new(compressed))
            .map_err(|e| RsmfError::structural(format!("zstd decompression failed: {e}")))?;
        // Bit-shuffling was originally implicit on every compressed section
        // (always applied at element size 4). The `SECTION_FLAG_BIT_SHUFFLED`
        // flag now records it explicitly. The batch writer sets the flag
        // whenever it compresses (it always bit-shuffles), so existing
        // test fixtures and migration scenarios round-trip unchanged. The
        // streaming writer compresses without bit-shuffling and leaves the
        // flag clear so the reader skips the un-shuffle step here.
        if bit_shuffled {
            Ok(crate::bit_shuffle::unshuffle(&decompressed, 4))
        } else {
            Ok(decompressed)
        }
    }
    #[cfg(not(feature = "compression"))]
    {
        let _ = (compressed, bit_shuffled);
        Err(RsmfError::unsupported(
            "compression not enabled".to_string(),
        ))
    }
}

pub(crate) fn validate_manifest(
    manifest: &Manifest,
    sections: &[SectionDescriptor],
    packed_section_indices: &[usize],
) -> Result<()> {
    let mut seen_names: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for t in &manifest.tensors {
        if !seen_names.insert(t.name.as_str()) {
            return Err(RsmfError::structural(format!(
                "duplicate tensor name {}",
                t.name
            )));
        }
    }
    for (i, v) in manifest.variants.iter().enumerate() {
        let section_idx = match v.section_kind {
            k if k == SectionKind::CanonicalArena.to_raw() as u8 => sections
                .iter()
                .position(|s| s.kind == SectionKind::CanonicalArena)
                .ok_or_else(|| {
                    RsmfError::structural(format!(
                        "variant[{i}] references missing canonical arena"
                    ))
                })?,
            k if k == SectionKind::PackedArena.to_raw() as u8 => {
                let idx = v.section_index as usize;
                *packed_section_indices.get(idx).ok_or_else(|| {
                    RsmfError::structural(format!(
                        "variant[{i}] references missing packed arena {idx}"
                    ))
                })?
            }
            other => {
                return Err(RsmfError::structural(format!(
                    "variant[{i}] section_kind {other} not a tensor arena"
                )));
            }
        };
        let section = &sections[section_idx];
        if !section.is_compressed() {
            let end = v
                .section_relative_offset
                .checked_add(v.length)
                .ok_or_else(|| RsmfError::structural("variant length overflow"))?;
            if end > section.length {
                return Err(RsmfError::structural(format!(
                    "variant[{i}] extends past its arena"
                )));
            }
        }

        // Enforce 64-byte alignment for canonical tensors as per BINARY_FORMAT.md
        if v.target == crate::tensor::variant::TargetTag::Canonical {
            if v.alignment < 64 {
                return Err(RsmfError::structural(format!(
                    "canonical variant[{i}] alignment {} is less than 64B requirement",
                    v.alignment
                )));
            }

            // Canonical/Raw variants must have storage_dtype matching the owning
            // tensor's logical dtype.
            if let Some(db) = v.storage_dtype.size_bytes() {
                if db > 0 && v.length % db as u64 != 0 {
                    return Err(RsmfError::structural(format!(
                        "canonical variant[{i}] length {} not multiple of dtype size {}",
                        v.length, db
                    )));
                }
            }
        }

        let align = u64::from(v.alignment);
        if align != 0 && v.section_relative_offset % align != 0 {
            return Err(RsmfError::structural(format!(
                "variant[{i}] offset {} not aligned to {}",
                v.section_relative_offset, align
            )));
        }
    }
    Ok(())
}
