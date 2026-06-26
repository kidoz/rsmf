//! Placement manifest codec and validation.
//!
//! The placement manifest is stored as a `SectionKind::Custom(128)` payload
//! so v1 readers that do not understand placement can preserve it as an
//! ancillary section. The payload itself follows the manifest codec style:
//! little-endian integers, `StringMap`, and length-prefixed arrays.

use std::collections::HashSet;

use crate::error::{Result, RsmfError};
use crate::manifest::Manifest;

/// Custom section discriminant used for the placement manifest.
pub const PLACEMENT_SECTION_KIND: u16 = 128;

/// Current placement manifest payload version.
pub const PLACEMENT_VERSION: u32 = 1;

/// Placement record flag: keep resident / pinned when possible.
pub const PLACEMENT_FLAG_PIN: u16 = 0x1;

/// Placement record flag: cold shard placement.
pub const PLACEMENT_FLAG_COLD: u16 = 0x2;

const PLACEMENT_KNOWN_FLAGS: u16 = PLACEMENT_FLAG_PIN | PLACEMENT_FLAG_COLD;

/// Device/backend kind used by placement records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeviceKind {
    /// CPU / host memory.
    Cpu = 0,
    /// NVIDIA CUDA.
    Cuda = 1,
    /// AMD ROCm / HIP.
    RocmHip = 2,
    /// Portable WGPU adapter.
    Wgpu = 3,
    /// Apple Metal.
    Metal = 4,
}

impl DeviceKind {
    /// Decode from the on-disk discriminant.
    pub fn from_raw(raw: u8) -> Result<Self> {
        Ok(match raw {
            0 => Self::Cpu,
            1 => Self::Cuda,
            2 => Self::RocmHip,
            3 => Self::Wgpu,
            4 => Self::Metal,
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown placement device kind {other}"
                )));
            }
        })
    }

    /// Parse the CLI/TOML string form.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "rocm_hip" => Self::RocmHip,
            "wgpu" => Self::Wgpu,
            "metal" => Self::Metal,
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown placement device kind {other:?}"
                )));
            }
        })
    }

    /// Human-readable stable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::RocmHip => "rocm_hip",
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
        }
    }
}

/// Memory tier where a shard is intended to reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryTier {
    /// Device VRAM.
    Vram = 0,
    /// Host RAM.
    Ram = 1,
    /// NVMe / SSD-backed tier.
    Nvme = 2,
}

impl MemoryTier {
    /// Decode from the on-disk discriminant.
    pub fn from_raw(raw: u8) -> Result<Self> {
        Ok(match raw {
            0 => Self::Vram,
            1 => Self::Ram,
            2 => Self::Nvme,
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown placement memory tier {other}"
                )));
            }
        })
    }

    /// Parse the CLI/TOML string form.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "vram" => Self::Vram,
            "ram" => Self::Ram,
            "nvme" => Self::Nvme,
            other => {
                return Err(RsmfError::structural(format!(
                    "unknown placement memory tier {other:?}"
                )));
            }
        })
    }

    /// Human-readable stable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Vram => "vram",
            Self::Ram => "ram",
            Self::Nvme => "nvme",
        }
    }
}

/// One device/tier entry referenced by placement records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDescriptor {
    /// Device id referenced by placement records.
    pub id: u32,
    /// Backend/device kind.
    pub kind: DeviceKind,
    /// Memory tier.
    pub tier: MemoryTier,
    /// Capacity in bytes. `0` means unknown.
    pub capacity_bytes: u64,
    /// Approximate bandwidth in MB/s. `0` means unknown.
    pub bandwidth_mbps: u64,
    /// Free-form metadata.
    pub metadata: Vec<(String, String)>,
}

/// Placement routing record for one physical shard id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementRecord {
    /// `TensorDescriptor.shard_id` this placement applies to.
    pub shard_id: u64,
    /// Primary device id.
    pub primary_device: u32,
    /// Lower value means lower priority; interpretation is runtime-specific.
    pub prefetch_priority: u16,
    /// Flag word. Unknown bits are rejected.
    pub flags: u16,
    /// Replica device ids.
    pub replicas: Vec<u32>,
}

/// Placement manifest payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementManifest {
    /// Payload version.
    pub version: u32,
    /// Free-form metadata.
    pub metadata: Vec<(String, String)>,
    /// Device descriptors.
    pub devices: Vec<DeviceDescriptor>,
    /// Placement records.
    pub placements: Vec<PlacementRecord>,
}

impl Default for PlacementManifest {
    fn default() -> Self {
        Self {
            version: PLACEMENT_VERSION,
            metadata: Vec::new(),
            devices: Vec::new(),
            placements: Vec::new(),
        }
    }
}

impl PlacementManifest {
    /// Encode the placement manifest payload.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate_internal()?;
        let mut w = Writer::default();
        w.u32(self.version);
        w.u32(0);
        write_string_map(&mut w, &self.metadata)?;
        w.u32(cast_u32(self.devices.len())?);
        for d in &self.devices {
            w.u32(d.id);
            w.u8(d.kind as u8);
            w.u8(d.tier as u8);
            w.u16(0);
            w.u64(d.capacity_bytes);
            w.u64(d.bandwidth_mbps);
            write_string_map(&mut w, &d.metadata)?;
        }
        w.u32(cast_u32(self.placements.len())?);
        for p in &self.placements {
            if p.flags & !PLACEMENT_KNOWN_FLAGS != 0 {
                return Err(RsmfError::structural(format!(
                    "placement for shard {} has unknown flags 0x{:04x}",
                    p.shard_id,
                    p.flags & !PLACEMENT_KNOWN_FLAGS
                )));
            }
            w.u64(p.shard_id);
            w.u32(p.primary_device);
            w.u16(p.prefetch_priority);
            w.u16(p.flags);
            w.u32(cast_u32(p.replicas.len())?);
            for &replica in &p.replicas {
                w.u32(replica);
            }
        }
        Ok(w.into_vec())
    }

    /// Decode a placement manifest payload.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let version = r.u32()?;
        if version != PLACEMENT_VERSION {
            return Err(RsmfError::structural(format!(
                "unsupported placement manifest version {version}"
            )));
        }
        let reserved = r.u32()?;
        if reserved != 0 {
            return Err(RsmfError::structural(
                "placement reserved field must be zero".to_string(),
            ));
        }
        let metadata = read_string_map(&mut r)?;
        let device_count = r.u32()? as usize;
        let mut devices = Vec::with_capacity(device_count);
        for _ in 0..device_count {
            let id = r.u32()?;
            let kind = DeviceKind::from_raw(r.u8()?)?;
            let tier = MemoryTier::from_raw(r.u8()?)?;
            let reserved = r.u16()?;
            if reserved != 0 {
                return Err(RsmfError::structural(
                    "placement device reserved field must be zero".to_string(),
                ));
            }
            let capacity_bytes = r.u64()?;
            let bandwidth_mbps = r.u64()?;
            let metadata = read_string_map(&mut r)?;
            devices.push(DeviceDescriptor {
                id,
                kind,
                tier,
                capacity_bytes,
                bandwidth_mbps,
                metadata,
            });
        }
        let placement_count = r.u32()? as usize;
        let mut placements = Vec::with_capacity(placement_count);
        for _ in 0..placement_count {
            let shard_id = r.u64()?;
            let primary_device = r.u32()?;
            let prefetch_priority = r.u16()?;
            let flags = r.u16()?;
            if flags & !PLACEMENT_KNOWN_FLAGS != 0 {
                return Err(RsmfError::structural(format!(
                    "placement for shard {shard_id} has unknown flags 0x{:04x}",
                    flags & !PLACEMENT_KNOWN_FLAGS
                )));
            }
            let replica_count = r.u32()? as usize;
            let mut replicas = Vec::with_capacity(replica_count);
            for _ in 0..replica_count {
                replicas.push(r.u32()?);
            }
            placements.push(PlacementRecord {
                shard_id,
                primary_device,
                prefetch_priority,
                flags,
                replicas,
            });
        }
        if r.remaining() != 0 {
            return Err(RsmfError::structural(format!(
                "placement manifest has {} trailing bytes",
                r.remaining()
            )));
        }
        let out = Self {
            version,
            metadata,
            devices,
            placements,
        };
        out.validate_internal()?;
        Ok(out)
    }

    /// Validate placement references against a parsed RSMF manifest.
    pub fn validate_against_manifest(&self, manifest: &Manifest) -> Result<()> {
        self.validate_internal()?;
        let shard_ids: HashSet<u64> = manifest.tensors.iter().map(|t| t.shard_id).collect();
        for p in &self.placements {
            if p.shard_id != 0 && !shard_ids.contains(&p.shard_id) {
                return Err(RsmfError::structural(format!(
                    "placement references shard_id {} with no tensor",
                    p.shard_id
                )));
            }
        }
        Ok(())
    }

    fn validate_internal(&self) -> Result<()> {
        let mut device_ids = HashSet::new();
        for d in &self.devices {
            if !device_ids.insert(d.id) {
                return Err(RsmfError::structural(format!(
                    "duplicate placement device id {}",
                    d.id
                )));
            }
        }
        for p in &self.placements {
            if !device_ids.contains(&p.primary_device) {
                return Err(RsmfError::structural(format!(
                    "placement for shard {} references missing primary_device {}",
                    p.shard_id, p.primary_device
                )));
            }
            for &replica in &p.replicas {
                if !device_ids.contains(&replica) {
                    return Err(RsmfError::structural(format!(
                        "placement for shard {} references missing replica device {}",
                        p.shard_id, replica
                    )));
                }
            }
        }
        Ok(())
    }
}

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
                "placement manifest truncated: need {n} more bytes at offset {}",
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
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.bytes[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(u32::from_le_bytes(b))
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

    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        let bs = self.bytes(len)?;
        std::str::from_utf8(bs)
            .map(str::to_string)
            .map_err(|e| RsmfError::structural(format!("invalid placement utf-8: {e}")))
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
    u32::try_from(n).map_err(|_| RsmfError::structural("placement count exceeds u32::MAX"))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::descriptor::TensorDescriptor;
    use crate::tensor::dtype::LogicalDtype;

    fn placement() -> PlacementManifest {
        PlacementManifest {
            version: PLACEMENT_VERSION,
            metadata: vec![("created_by".into(), "test".into())],
            devices: vec![
                DeviceDescriptor {
                    id: 0,
                    kind: DeviceKind::Wgpu,
                    tier: MemoryTier::Vram,
                    capacity_bytes: 1024,
                    bandwidth_mbps: 100,
                    metadata: vec![("name".into(), "gpu0".into())],
                },
                DeviceDescriptor {
                    id: 1,
                    kind: DeviceKind::Cpu,
                    tier: MemoryTier::Ram,
                    capacity_bytes: 2048,
                    bandwidth_mbps: 50,
                    metadata: vec![],
                },
            ],
            placements: vec![PlacementRecord {
                shard_id: 7,
                primary_device: 0,
                prefetch_priority: 10,
                flags: PLACEMENT_FLAG_PIN,
                replicas: vec![1],
            }],
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let p = placement();
        let bytes = p.encode().unwrap();
        let back = PlacementManifest::decode(&bytes).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn rejects_unknown_flags() {
        let mut p = placement();
        p.placements[0].flags = 0x8000;
        let err = p.encode().unwrap_err();
        assert!(matches!(err, RsmfError::Structural(_)));
    }

    #[test]
    fn validates_shard_references() {
        let manifest = Manifest {
            tensors: vec![TensorDescriptor {
                name: "w".into(),
                dtype: LogicalDtype::F32,
                shape: vec![1],
                canonical_variant: 0,
                packed_variants: vec![],
                shard_id: 7,
                metadata: vec![],
            }],
            ..Manifest::default()
        };
        placement().validate_against_manifest(&manifest).unwrap();

        let empty = Manifest::default();
        let err = placement().validate_against_manifest(&empty).unwrap_err();
        assert!(err.to_string().contains("shard_id 7"));
    }
}
