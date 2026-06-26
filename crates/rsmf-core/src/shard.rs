//! Writer-side multi-file sharding.
//!
//! The sharder rewrites an existing RSMF file into a master file plus raw shard
//! buffers. The master keeps placeholder tensor arenas so the section table
//! remains structurally valid; tensor variant bytes are read from the external
//! shard whose id is recorded on the owning tensor.

use std::fs;
use std::path::{Path, PathBuf};

use crate::checksum::{CHECKSUM_LEN, digest_128};
use crate::error::{Result, RsmfError};
use crate::manifest::Manifest;
use crate::placement::PLACEMENT_SECTION_KIND;
use crate::preamble::{FORMAT_MAJOR, FORMAT_MINOR, MAGIC, PREAMBLE_LEN, Preamble};
use crate::reader::RsmfFile;
use crate::section::{SECTION_DESC_LEN, SectionDescriptor, SectionKind};
use crate::tier::{Tier, tier_intent_from_meta};

/// Built-in tensor-to-shard assignment strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Balance tensors by total variant byte length.
    Size,
    /// Group tensors by the first `tier.intent` found on their variants.
    Tier,
    /// Group tensors by MoE expert metadata (`moe.layer` + `moe.expert`).
    Expert,
}

/// Options for [`write_sharded_file`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardWriteOptions {
    /// Number of shard files to emit. Shard ids are `1..=shard_count`.
    pub shard_count: u64,
    /// Assignment strategy.
    pub strategy: ShardStrategy,
}

/// One emitted shard artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardArtifact {
    /// Shard id recorded in tensor descriptors.
    pub shard_id: u64,
    /// Path written for this shard.
    pub path: PathBuf,
    /// Number of bytes in the raw shard buffer.
    pub bytes: u64,
    /// Number of tensors assigned to this shard.
    pub tensor_count: usize,
    /// BLAKE3-128 of the shard file bytes.
    pub checksum: [u8; CHECKSUM_LEN],
}

/// Summary returned by [`write_sharded_file`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardWriteSummary {
    /// Master file path.
    pub master_path: PathBuf,
    /// Emitted shard files.
    pub shards: Vec<ShardArtifact>,
}

struct SectionPayload {
    kind: SectionKind,
    align: u16,
    flags: u32,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct TensorAssignment {
    shard_id: u64,
}

/// Emit a sharded master and raw shard files from an open source file.
///
/// The source file is read through [`RsmfFile`], so compressed source arenas are
/// materialized before being copied into raw shard buffers. The output master
/// uses uncompressed placeholder tensor arenas; `full_verify()` succeeds only
/// after the generated shard files are attached.
pub fn write_sharded_file(
    src: &RsmfFile,
    master_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &ShardWriteOptions,
) -> Result<ShardWriteSummary> {
    if options.shard_count == 0 {
        return Err(RsmfError::structural("shard_count must be > 0"));
    }

    let master_path = master_path.as_ref();
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir).map_err(|source| RsmfError::IoWithPath {
        path: out_dir.to_path_buf(),
        source,
    })?;

    let assignments = assign_tensors(src.manifest(), options)?;
    let mut manifest = src.manifest().clone();
    let mut shard_bytes = vec![Vec::<u8>::new(); options.shard_count as usize];

    for (tensor_idx, assignment) in assignments.iter().enumerate() {
        manifest.tensors[tensor_idx].shard_id = assignment.shard_id;
        let variant_indices = tensor_variant_indices(&manifest, tensor_idx);
        for variant_idx in variant_indices {
            let src_variant = &src.manifest().variants[variant_idx as usize];
            let bytes = src.variant_bytes(src_variant)?;
            let out_variant = &mut manifest.variants[variant_idx as usize];
            let shard = &mut shard_bytes[(assignment.shard_id - 1) as usize];
            pad_to_alignment(shard, u64::from(out_variant.alignment));
            let offset = shard.len() as u64;
            shard.extend_from_slice(bytes);
            out_variant.section_relative_offset = offset;
            out_variant.length = bytes.len() as u64;
            out_variant.checksum = digest_128(bytes);
        }
    }

    let payloads = build_master_payloads(src, &manifest)?;
    let master_bytes = encode_file(&manifest, payloads)?;
    fs::write(master_path, &master_bytes).map_err(|source| RsmfError::IoWithPath {
        path: master_path.to_path_buf(),
        source,
    })?;

    let mut artifacts = Vec::with_capacity(shard_bytes.len());
    for (i, bytes) in shard_bytes.iter().enumerate() {
        let shard_id = (i + 1) as u64;
        let path = out_dir.join(format!("shard-{shard_id}.bin"));
        fs::write(&path, bytes).map_err(|source| RsmfError::IoWithPath {
            path: path.clone(),
            source,
        })?;
        let tensor_count = assignments
            .iter()
            .filter(|assignment| assignment.shard_id == shard_id)
            .count();
        artifacts.push(ShardArtifact {
            shard_id,
            path,
            bytes: bytes.len() as u64,
            tensor_count,
            checksum: digest_128(bytes),
        });
    }

    Ok(ShardWriteSummary {
        master_path: master_path.to_path_buf(),
        shards: artifacts,
    })
}

fn assign_tensors(
    manifest: &Manifest,
    options: &ShardWriteOptions,
) -> Result<Vec<TensorAssignment>> {
    let mut sizes = Vec::with_capacity(manifest.tensors.len());
    for tensor_idx in 0..manifest.tensors.len() {
        let variant_bytes = tensor_variant_indices(manifest, tensor_idx)
            .iter()
            .map(|&idx| manifest.variants[idx as usize].length)
            .sum();
        sizes.push(variant_bytes);
    }

    match options.strategy {
        ShardStrategy::Size => Ok(assign_by_size(&sizes, options.shard_count)),
        ShardStrategy::Tier => assign_by_tier(manifest, &sizes, options.shard_count),
        ShardStrategy::Expert => assign_by_expert(manifest, &sizes, options.shard_count),
    }
}

fn assign_by_size(sizes: &[u64], shard_count: u64) -> Vec<TensorAssignment> {
    let mut loads = vec![0u64; shard_count as usize];
    let mut out = vec![TensorAssignment { shard_id: 1 }; sizes.len()];
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&idx| std::cmp::Reverse(sizes[idx]));
    for idx in order {
        let shard_idx = loads
            .iter()
            .enumerate()
            .min_by_key(|(_, load)| **load)
            .map(|(i, _)| i)
            .unwrap_or(0);
        loads[shard_idx] += sizes[idx];
        out[idx] = TensorAssignment {
            shard_id: (shard_idx + 1) as u64,
        };
    }
    out
}

fn assign_by_tier(
    manifest: &Manifest,
    sizes: &[u64],
    shard_count: u64,
) -> Result<Vec<TensorAssignment>> {
    let mut tier_to_shard: Vec<(Tier, u64)> = Vec::new();
    let mut loads = vec![0u64; shard_count as usize];
    let mut out = Vec::with_capacity(manifest.tensors.len());

    for (tensor_idx, &size) in sizes.iter().enumerate().take(manifest.tensors.len()) {
        let tier = first_tensor_tier(manifest, tensor_idx)?;
        let shard_id = if let Some(tier) = tier {
            if let Some((_, shard_id)) = tier_to_shard.iter().find(|(t, _)| *t == tier) {
                *shard_id
            } else {
                let shard_id = ((tier_to_shard.len() as u64) % shard_count) + 1;
                tier_to_shard.push((tier, shard_id));
                shard_id
            }
        } else {
            least_loaded_shard(&loads)
        };
        loads[(shard_id - 1) as usize] += size;
        out.push(TensorAssignment { shard_id });
    }
    Ok(out)
}

fn assign_by_expert(
    manifest: &Manifest,
    sizes: &[u64],
    shard_count: u64,
) -> Result<Vec<TensorAssignment>> {
    let mut group_to_shard: Vec<(String, u64)> = Vec::new();
    let mut loads = vec![0u64; shard_count as usize];
    let mut out = Vec::with_capacity(manifest.tensors.len());

    for (tensor_idx, tensor) in manifest.tensors.iter().enumerate() {
        let group = expert_group_key(&tensor.metadata)?;
        let shard_id = if let Some(group) = group {
            if let Some((_, shard_id)) = group_to_shard.iter().find(|(g, _)| g == &group) {
                *shard_id
            } else {
                let shard_id = ((group_to_shard.len() as u64) % shard_count) + 1;
                group_to_shard.push((group, shard_id));
                shard_id
            }
        } else {
            least_loaded_shard(&loads)
        };
        loads[(shard_id - 1) as usize] += sizes[tensor_idx];
        out.push(TensorAssignment { shard_id });
    }
    Ok(out)
}

fn least_loaded_shard(loads: &[u64]) -> u64 {
    loads
        .iter()
        .enumerate()
        .min_by_key(|(_, load)| **load)
        .map(|(i, _)| (i + 1) as u64)
        .unwrap_or(1)
}

fn first_tensor_tier(manifest: &Manifest, tensor_idx: usize) -> Result<Option<Tier>> {
    for variant_idx in tensor_variant_indices(manifest, tensor_idx) {
        let variant = &manifest.variants[variant_idx as usize];
        if let Some(tier) = tier_intent_from_meta(&variant.meta.extra)? {
            return Ok(Some(tier));
        }
    }
    Ok(None)
}

fn expert_group_key(meta: &[(String, String)]) -> Result<Option<String>> {
    let Some(layer) = lookup(meta, "moe.layer") else {
        return Ok(None);
    };
    if let Some(expert) = lookup(meta, "moe.expert") {
        return Ok(Some(format!("layer={layer};expert={expert}")));
    }
    if lookup(meta, "moe.shared") == Some("1") {
        return Ok(Some(format!("layer={layer};shared=1")));
    }
    Ok(Some(format!("layer={layer};dense")))
}

fn lookup<'a>(meta: &'a [(String, String)], key: &str) -> Option<&'a str> {
    meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn tensor_variant_indices(manifest: &Manifest, tensor_idx: usize) -> Vec<u32> {
    let tensor = &manifest.tensors[tensor_idx];
    let mut out = Vec::with_capacity(1 + tensor.packed_variants.len());
    out.push(tensor.canonical_variant);
    out.extend(tensor.packed_variants.iter().copied());
    out
}

fn build_master_payloads(src: &RsmfFile, manifest: &Manifest) -> Result<Vec<SectionPayload>> {
    let mut payloads = Vec::new();
    payloads.push(SectionPayload {
        kind: SectionKind::CanonicalArena,
        align: source_section_align(src, SectionKind::CanonicalArena, 0).unwrap_or(64),
        flags: 0,
        bytes: placeholder_arena(manifest, SectionKind::CanonicalArena, 0),
    });

    let packed_count = manifest
        .variants
        .iter()
        .filter(|v| v.section_kind == SectionKind::PackedArena.to_raw() as u8)
        .map(|v| v.section_index)
        .max()
        .map_or(0usize, |idx| idx as usize + 1);
    for packed_idx in 0..packed_count {
        payloads.push(SectionPayload {
            kind: SectionKind::PackedArena,
            align: source_section_align(src, SectionKind::PackedArena, packed_idx).unwrap_or(64),
            flags: 0,
            bytes: placeholder_arena(manifest, SectionKind::PackedArena, packed_idx as u8),
        });
    }

    if !manifest.graphs.is_empty() {
        payloads.push(SectionPayload {
            kind: SectionKind::Graph,
            align: source_section_align(src, SectionKind::Graph, 0).unwrap_or(8),
            flags: 0,
            bytes: graph_section_bytes(src)?,
        });
    }
    if !manifest.assets.is_empty() {
        payloads.push(SectionPayload {
            kind: SectionKind::Assets,
            align: source_section_align(src, SectionKind::Assets, 0).unwrap_or(8),
            flags: 0,
            bytes: assets_section_bytes(src)?,
        });
    }
    for custom in src.custom_sections()? {
        if custom.kind == PLACEMENT_SECTION_KIND {
            continue;
        }
        payloads.push(SectionPayload {
            kind: SectionKind::Custom(custom.kind),
            align: custom.alignment,
            flags: 0,
            bytes: custom.bytes,
        });
    }
    Ok(payloads)
}

fn source_section_align(src: &RsmfFile, kind: SectionKind, index: usize) -> Option<u16> {
    src.sections()
        .iter()
        .filter(|section| section.kind == kind)
        .nth(index)
        .map(|section| section.align)
}

fn placeholder_arena(manifest: &Manifest, kind: SectionKind, section_index: u8) -> Vec<u8> {
    let kind_raw = kind.to_raw() as u8;
    let len = manifest
        .variants
        .iter()
        .filter(|variant| {
            variant.section_kind == kind_raw && variant.section_index == section_index
        })
        .filter_map(|variant| variant.section_relative_offset.checked_add(variant.length))
        .max()
        .unwrap_or(0);
    vec![0u8; len as usize]
}

fn graph_section_bytes(src: &RsmfFile) -> Result<Vec<u8>> {
    let manifest = src.manifest();
    let len = manifest
        .graphs
        .iter()
        .filter_map(|graph| graph.offset.checked_add(graph.length))
        .max()
        .unwrap_or(0);
    let mut out = vec![0u8; len as usize];
    for (graph, payload) in manifest.graphs.iter().zip(src.graph_payloads()) {
        let start = graph.offset as usize;
        let end = start + payload.bytes.len();
        out[start..end].copy_from_slice(payload.bytes);
    }
    Ok(out)
}

fn assets_section_bytes(src: &RsmfFile) -> Result<Vec<u8>> {
    let manifest = src.manifest();
    let len = manifest
        .assets
        .iter()
        .filter_map(|asset| asset.offset.checked_add(asset.length))
        .max()
        .unwrap_or(0);
    let mut out = vec![0u8; len as usize];
    for asset in &manifest.assets {
        let payload = src.asset(&asset.name).ok_or_else(|| {
            RsmfError::structural(format!("asset {} missing payload", asset.name))
        })?;
        let start = asset.offset as usize;
        let end = start + payload.bytes.len();
        out[start..end].copy_from_slice(payload.bytes);
    }
    Ok(out)
}

fn encode_file(manifest: &Manifest, payloads: Vec<SectionPayload>) -> Result<Vec<u8>> {
    let manifest_bytes = manifest.encode()?;
    let section_count = payloads.len() as u64 + 1;
    let mut cursor = PREAMBLE_LEN + section_count * SECTION_DESC_LEN;

    cursor = align_up(cursor, 8);
    let manifest_off = cursor;
    let manifest_len = manifest_bytes.len() as u64;
    cursor += manifest_len;

    let mut layouts = Vec::with_capacity(payloads.len());
    for payload in &payloads {
        cursor = align_up(cursor, u64::from(payload.align));
        let offset = cursor;
        let length = payload.bytes.len() as u64;
        layouts.push((offset, length, digest_128(&payload.bytes)));
        cursor += length;
    }

    let mut section_table = Vec::with_capacity(section_count as usize);
    section_table.push(SectionDescriptor {
        kind: SectionKind::Manifest,
        align: 8,
        flags: 0,
        offset: manifest_off,
        length: manifest_len,
        checksum: digest_128(&manifest_bytes),
    });
    for (payload, (offset, length, checksum)) in payloads.iter().zip(layouts.iter()) {
        section_table.push(SectionDescriptor {
            kind: payload.kind,
            align: payload.align,
            flags: payload.flags,
            offset: *offset,
            length: *length,
            checksum: *checksum,
        });
    }

    let preamble = Preamble {
        magic: MAGIC,
        major: FORMAT_MAJOR,
        minor: FORMAT_MINOR,
        flags: 0,
        header_len: PREAMBLE_LEN,
        section_tbl_off: PREAMBLE_LEN,
        section_tbl_count: section_count,
        manifest_off,
        manifest_len,
        preamble_checksum: [0u8; 8],
    };

    let mut out = Vec::with_capacity(cursor as usize);
    out.extend_from_slice(&preamble.encode());
    for section in &section_table {
        out.extend_from_slice(&section.encode());
    }
    pad_to_file_offset(&mut out, manifest_off)?;
    out.extend_from_slice(&manifest_bytes);
    for (payload, (offset, _, _)) in payloads.iter().zip(layouts.iter()) {
        pad_to_file_offset(&mut out, *offset)?;
        out.extend_from_slice(&payload.bytes);
    }
    Ok(out)
}

fn pad_to_alignment(bytes: &mut Vec<u8>, align: u64) {
    let target = align_up(bytes.len() as u64, align.max(1));
    bytes.resize(target as usize, 0);
}

fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

fn pad_to_file_offset(bytes: &mut Vec<u8>, target: u64) -> Result<()> {
    if bytes.len() > target as usize {
        return Err(RsmfError::structural(format!(
            "writer cursor {} exceeds target offset {target}",
            bytes.len()
        )));
    }
    bytes.resize(target as usize, 0);
    Ok(())
}
