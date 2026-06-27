//! `rsmf placement` — inspect and set placement manifests.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::anyhow;
use rsmf_core::checksum::digest_128;
use rsmf_core::preamble::PREAMBLE_LEN;
use rsmf_core::section::SECTION_DESC_LEN;
use rsmf_core::{
    DeviceDescriptor, DeviceKind, MAGIC, MemoryTier, PLACEMENT_FLAG_COLD, PLACEMENT_FLAG_PIN,
    PLACEMENT_SECTION_KIND, PlacementManifest, PlacementRecord, Preamble, RsmfFile,
    SectionDescriptor, SectionKind,
};
use toml::Value;

use super::CliError;

/// Arguments to `rsmf placement`.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Placement operation.
    #[command(subcommand)]
    pub command: Command,
}

/// `rsmf placement` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Print a placement manifest.
    Inspect(InspectArgs),
    /// Add or replace a placement manifest from a TOML plan.
    Set(SetArgs),
}

/// Arguments to `rsmf placement inspect`.
#[derive(Debug, clap::Args)]
pub struct InspectArgs {
    /// Path to the RSMF file.
    pub file: PathBuf,
}

/// Arguments to `rsmf placement set`.
#[derive(Debug, clap::Args)]
pub struct SetArgs {
    /// Path to the RSMF file.
    pub file: PathBuf,
    /// TOML placement plan.
    #[arg(long)]
    pub plan: PathBuf,
    /// Optional output path. Defaults to replacing `file` atomically.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// Execute `rsmf placement`.
pub fn run(args: Args) -> Result<(), CliError> {
    match args.command {
        Command::Inspect(args) => inspect(args),
        Command::Set(args) => set(args),
    }
}

fn inspect(args: InspectArgs) -> Result<(), CliError> {
    let file = RsmfFile::open(&args.file)?;
    let Some(placement) = file.placement_manifest()? else {
        println!("PlacementManifest: (none)");
        return Ok(());
    };

    println!("PlacementManifest: version {}", placement.version);
    if !placement.metadata.is_empty() {
        println!("Metadata:");
        for (k, v) in &placement.metadata {
            println!("  {k} = {v}");
        }
    }
    println!("Devices: {}", placement.devices.len());
    for d in &placement.devices {
        println!(
            "  id={} kind={} tier={} capacity_bytes={} bandwidth_mbps={}",
            d.id,
            d.kind.name(),
            d.tier.name(),
            d.capacity_bytes,
            d.bandwidth_mbps,
        );
        for (k, v) in &d.metadata {
            println!("    {k} = {v}");
        }
    }
    println!("Placements: {}", placement.placements.len());
    for p in &placement.placements {
        let replicas: Vec<String> = p.replicas.iter().map(ToString::to_string).collect();
        println!(
            "  shard_id={} primary_device={} prefetch_priority={} flags=0x{:04x} replicas=[{}]",
            p.shard_id,
            p.primary_device,
            p.prefetch_priority,
            p.flags,
            replicas.join(","),
        );
    }
    Ok(())
}

fn set(args: SetArgs) -> Result<(), CliError> {
    let src = RsmfFile::open(&args.file)?;
    let plan = fs::read_to_string(&args.plan)
        .map_err(|e| CliError::user(anyhow!("failed to read {}: {e}", args.plan.display())))?;
    let placement = parse_plan(&plan)?;
    placement.validate_against_manifest(src.manifest())?;

    let output = args.out.as_ref().unwrap_or(&args.file);
    rewrite_with_placement(&args.file, output, &src, &placement)?;
    println!(
        "placement set: {} (devices={}, placements={})",
        output.display(),
        placement.devices.len(),
        placement.placements.len()
    );
    Ok(())
}

fn parse_plan(plan: &str) -> Result<PlacementManifest, CliError> {
    let value: Value = toml::from_str(plan.trim_start())
        .map_err(|e| CliError::user(anyhow!("invalid TOML placement plan: {e}")))?;
    let root = value
        .as_table()
        .ok_or_else(|| CliError::user(anyhow!("placement plan must be a TOML table")))?;

    let metadata = root
        .get("metadata")
        .map(parse_string_map)
        .transpose()?
        .unwrap_or_default();

    let devices_value = root
        .get("devices")
        .and_then(Value::as_array)
        .ok_or_else(|| CliError::user(anyhow!("placement plan requires [[devices]] entries")))?;
    let mut devices = Vec::with_capacity(devices_value.len());
    for (i, value) in devices_value.iter().enumerate() {
        let table = value
            .as_table()
            .ok_or_else(|| CliError::user(anyhow!("devices[{i}] must be a table")))?;
        devices.push(DeviceDescriptor {
            id: get_u32(table, "id", &format!("devices[{i}]"))?,
            kind: DeviceKind::parse(get_str(table, "kind", &format!("devices[{i}]"))?)?,
            tier: MemoryTier::parse(get_str(table, "tier", &format!("devices[{i}]"))?)?,
            capacity_bytes: get_optional_u64(table, "capacity_bytes", &format!("devices[{i}]"))?
                .unwrap_or(0),
            bandwidth_mbps: get_optional_u64(table, "bandwidth_mbps", &format!("devices[{i}]"))?
                .unwrap_or(0),
            metadata: table
                .get("metadata")
                .map(parse_string_map)
                .transpose()?
                .unwrap_or_default(),
        });
    }

    let placements_value = root
        .get("placements")
        .and_then(Value::as_array)
        .ok_or_else(|| CliError::user(anyhow!("placement plan requires [[placements]] entries")))?;
    let mut placements = Vec::with_capacity(placements_value.len());
    for (i, value) in placements_value.iter().enumerate() {
        let table = value
            .as_table()
            .ok_or_else(|| CliError::user(anyhow!("placements[{i}] must be a table")))?;
        placements.push(PlacementRecord {
            shard_id: get_u64(table, "shard_id", &format!("placements[{i}]"))?,
            primary_device: get_u32(table, "primary_device", &format!("placements[{i}]"))?,
            prefetch_priority: get_optional_u16(
                table,
                "prefetch_priority",
                &format!("placements[{i}]"),
            )?
            .unwrap_or(0),
            flags: parse_flags(table.get("flags"), &format!("placements[{i}]"))?,
            replicas: parse_replicas(table.get("replicas"), &format!("placements[{i}]"))?,
        });
    }

    let placement = PlacementManifest {
        version: rsmf_core::PLACEMENT_VERSION,
        metadata,
        devices,
        placements,
    };
    placement.encode()?;
    Ok(placement)
}

fn parse_string_map(value: &Value) -> Result<Vec<(String, String)>, CliError> {
    let table = value
        .as_table()
        .ok_or_else(|| CliError::user(anyhow!("metadata must be a table")))?;
    let mut out = Vec::with_capacity(table.len());
    for (k, v) in table {
        let s = v
            .as_str()
            .ok_or_else(|| CliError::user(anyhow!("metadata value {k:?} must be a string")))?;
        out.push((k.clone(), s.to_string()));
    }
    Ok(out)
}

fn get_str<'a>(
    table: &'a toml::map::Map<String, Value>,
    key: &str,
    scope: &str,
) -> Result<&'a str, CliError> {
    table
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| CliError::user(anyhow!("{scope}.{key} must be a string")))
}

fn get_u64(table: &toml::map::Map<String, Value>, key: &str, scope: &str) -> Result<u64, CliError> {
    let raw = table
        .get(key)
        .and_then(Value::as_integer)
        .ok_or_else(|| CliError::user(anyhow!("{scope}.{key} must be an integer")))?;
    u64::try_from(raw).map_err(|_| CliError::user(anyhow!("{scope}.{key} must be >= 0")))
}

fn get_optional_u64(
    table: &toml::map::Map<String, Value>,
    key: &str,
    scope: &str,
) -> Result<Option<u64>, CliError> {
    if table.contains_key(key) {
        Ok(Some(get_u64(table, key, scope)?))
    } else {
        Ok(None)
    }
}

fn get_u32(table: &toml::map::Map<String, Value>, key: &str, scope: &str) -> Result<u32, CliError> {
    u32::try_from(get_u64(table, key, scope)?)
        .map_err(|_| CliError::user(anyhow!("{scope}.{key} exceeds u32::MAX")))
}

fn get_optional_u16(
    table: &toml::map::Map<String, Value>,
    key: &str,
    scope: &str,
) -> Result<Option<u16>, CliError> {
    if table.contains_key(key) {
        let value = get_u64(table, key, scope)?;
        Ok(Some(u16::try_from(value).map_err(|_| {
            CliError::user(anyhow!("{scope}.{key} exceeds u16::MAX"))
        })?))
    } else {
        Ok(None)
    }
}

fn parse_flags(value: Option<&Value>, scope: &str) -> Result<u16, CliError> {
    let Some(value) = value else {
        return Ok(0);
    };
    if let Some(raw) = value.as_integer() {
        return u16::try_from(raw)
            .map_err(|_| CliError::user(anyhow!("{scope}.flags exceeds u16::MAX")));
    }
    if let Some(name) = value.as_str() {
        return flag_by_name(name, scope);
    }
    let array = value
        .as_array()
        .ok_or_else(|| CliError::user(anyhow!("{scope}.flags must be an integer/string/array")))?;
    let mut flags = 0u16;
    for item in array {
        let name = item
            .as_str()
            .ok_or_else(|| CliError::user(anyhow!("{scope}.flags entries must be strings")))?;
        flags |= flag_by_name(name, scope)?;
    }
    Ok(flags)
}

fn flag_by_name(name: &str, scope: &str) -> Result<u16, CliError> {
    Ok(match name {
        "pin" => PLACEMENT_FLAG_PIN,
        "cold" => PLACEMENT_FLAG_COLD,
        other => {
            return Err(CliError::user(anyhow!(
                "{scope}.flags contains unknown flag {other:?}"
            )));
        }
    })
}

fn parse_replicas(value: Option<&Value>, scope: &str) -> Result<Vec<u32>, CliError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let array = value
        .as_array()
        .ok_or_else(|| CliError::user(anyhow!("{scope}.replicas must be an array")))?;
    let mut out = Vec::with_capacity(array.len());
    for (i, item) in array.iter().enumerate() {
        let raw = item
            .as_integer()
            .ok_or_else(|| CliError::user(anyhow!("{scope}.replicas[{i}] must be an integer")))?;
        let raw = u64::try_from(raw)
            .map_err(|_| CliError::user(anyhow!("{scope}.replicas[{i}] must be >= 0")))?;
        out.push(
            u32::try_from(raw)
                .map_err(|_| CliError::user(anyhow!("{scope}.replicas[{i}] exceeds u32::MAX")))?,
        );
    }
    Ok(out)
}

fn rewrite_with_placement(
    input: &Path,
    output: &Path,
    src: &RsmfFile,
    placement: &PlacementManifest,
) -> Result<(), CliError> {
    let source_bytes = fs::read(input)?;
    let placement_bytes = placement.encode()?;
    let mut payloads: Vec<(SectionKind, u16, u32, Vec<u8>)> = Vec::new();
    for section in src.sections() {
        if section.kind == SectionKind::Custom(PLACEMENT_SECTION_KIND) {
            continue;
        }
        let start = section.offset as usize;
        let end = (section.offset + section.length) as usize;
        payloads.push((
            section.kind,
            section.align,
            section.flags,
            source_bytes[start..end].to_vec(),
        ));
    }
    payloads.push((
        SectionKind::Custom(PLACEMENT_SECTION_KIND),
        8,
        0,
        placement_bytes,
    ));

    let section_count = payloads.len() as u64;
    let mut cursor = PREAMBLE_LEN + section_count * SECTION_DESC_LEN;
    let mut section_table = Vec::with_capacity(payloads.len());
    let mut layouts = Vec::with_capacity(payloads.len());
    let mut manifest_off = 0u64;
    let mut manifest_len = 0u64;

    for (kind, align, flags, bytes) in &payloads {
        cursor = align_up(cursor, u64::from(*align));
        let offset = cursor;
        let length = bytes.len() as u64;
        if *kind == SectionKind::Manifest {
            manifest_off = offset;
            manifest_len = length;
        }
        section_table.push(SectionDescriptor {
            kind: *kind,
            align: *align,
            flags: *flags,
            offset,
            length,
            checksum: digest_128(bytes),
        });
        layouts.push(offset);
        cursor += length;
    }

    if manifest_len == 0 {
        return Err(CliError::user(anyhow!(
            "source file has no manifest section after rewrite planning"
        )));
    }

    let preamble = Preamble {
        magic: MAGIC,
        major: src.preamble().major,
        minor: src.preamble().minor,
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
    for (i, (_kind, _align, _flags, bytes)) in payloads.iter().enumerate() {
        pad_to_file_offset(&mut out, layouts[i])?;
        out.extend_from_slice(bytes);
    }
    debug_assert_eq!(out.len() as u64, cursor);
    write_atomic(output, &out)?;

    // Re-open the result so malformed section-table rewrites fail before
    // returning success to the user.
    let rewritten = RsmfFile::open(output)?;
    rewritten.full_verify()?;
    Ok(())
}

fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

fn pad_to_file_offset(bytes: &mut Vec<u8>, target: u64) -> Result<(), CliError> {
    let target = target as usize;
    if bytes.len() > target {
        return Err(CliError::user(anyhow!(
            "writer cursor {} exceeds target offset {}",
            bytes.len(),
            target
        )));
    }
    bytes.resize(target, 0);
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut tmp_path = parent.join(format!(".rsmf-placement-tmp-{}", std::process::id()));
    let mut suffix = 0u32;
    while tmp_path.exists() {
        suffix += 1;
        tmp_path = parent.join(format!(
            ".rsmf-placement-tmp-{}-{suffix}",
            std::process::id()
        ));
    }
    {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
    }
    fs::rename(&tmp_path, path)?;
    Ok(())
}
