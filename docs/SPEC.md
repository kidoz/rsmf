# RSMF — Binary Format Specification

**Status:** v1.0.
**Endianness:** little-endian on disk, everywhere.
**Integer widths:** all offsets, lengths, and counts are unsigned 64-bit unless noted.

This document is authoritative for the on-disk layout. When the code
and this spec disagree, that is a bug in one or the other — file an
issue.

---

## 1. High-level layout

```
+---------------------------------------------------+
| 0  Preamble (fixed, 64 bytes)                     |
+---------------------------------------------------+
| 1  Section table (N × 64 bytes)                   |
+---------------------------------------------------+
| 2  Manifest section (binary)                      |
+---------------------------------------------------+
| 3  Canonical tensor arena                         |
+---------------------------------------------------+
| 4  Packed tensor arena(s)              (optional) |
+---------------------------------------------------+
| 5  Graph section (opaque ONNX/ORT)     (optional) |
+---------------------------------------------------+
| 6  Asset section                       (optional) |
+---------------------------------------------------+
```

The **batch writer** (`crate::RsmfWriter`) emits sections in the order
above. The **streaming writer** (`crate::StreamingRsmfWriter`) emits
them in a different order and patches the preamble at `finish()`; see
§9. Both produce files that are valid per this specification and
interchangeable from the reader's perspective.

---

## 2. Preamble (offset `0`, length `64`)

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 8 | magic | `"RSMF\0\0\0\x01"` |
| 0x08 | 2 | major | format major version (u16) |
| 0x0A | 2 | minor | format minor version (u16) |
| 0x0C | 4 | flags | reserved; MUST be zero in v1 |
| 0x10 | 8 | header_len | total preamble length (u64 = 64) |
| 0x18 | 8 | tbl_off | absolute offset of section table (u64) |
| 0x20 | 8 | tbl_count | number of sections (u64) |
| 0x28 | 8 | manifest_off| absolute offset of manifest payload (u64) |
| 0x30 | 8 | manifest_len| manifest length (u64) |
| 0x38 | 8 | checksum | BLAKE3 of bytes `[0x00..0x38]` truncated to 8 bytes |

Readers MUST verify the preamble checksum before trusting any other
field. A readable preamble with a bad checksum is `Structural` error
territory — the file is corrupted, not merely of an unknown version.

---

## 3. Section table

Array of 64-byte entries starting at `tbl_off`.

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 2 | kind | see §3.1 |
| 0x02 | 2 | align | required power-of-two alignment of `offset` |
| 0x04 | 4 | flags | see §3.2 |
| 0x08 | 8 | offset | absolute file offset of the payload (u64) |
| 0x10 | 8 | length | payload length in bytes (u64) |
| 0x18 | 16 | checksum | BLAKE3 of payload truncated to 16 bytes |
| 0x28 | 24 | reserved | zero-filled |

Entries MUST list sections in ascending `offset` order with no overlap,
and every `offset + length` MUST fit inside the file. Sections with
length zero are rejected.

### 3.1 `SectionKind` discriminants

| Discriminant | Name           | Notes |
|---|---|---|
| `1`  | `Manifest`       | Exactly one. |
| `2`  | `CanonicalArena` | Exactly one; holds raw canonical tensor bodies. |
| `3`  | `PackedArena`    | Zero or more; one per `ArenaGroup` emitted. |
| `4`  | `Graph`          | At most one; opaque ONNX / ORT bytes. |
| `5`  | `Assets`         | At most one; concatenated named assets. |
| `6..127` | *reserved*   | Rejected with `Structural` until promoted. |
| `>= 128` | `Custom(raw)` | Vendor / user-defined ancillary section; readers MUST preserve on round-trip but are not required to understand the bytes. Mirrors PNG's "ancillary chunk" convention. |

### 3.2 Section flag bits

| Bit  | Name                       | Meaning |
|---|---|---|
| `0x1` | `SECTION_FLAG_COMPRESSED`   | Payload is zstd-encoded on disk. Readers must decompress before reading. |
| `0x2` | `SECTION_FLAG_BIT_SHUFFLED` | Payload was passed through the rsmf-core bit-shuffle pre-processor (element size 4, matching f32) *before* compression. Readers must un-shuffle after decompression. Typically set together with bit 0 by the batch writer. The streaming writer never sets this bit. |

Unknown flag bits MUST be rejected with a `Structural` error — a new
writer accidentally setting a bit an old reader doesn't understand is
exactly the "fail loud" case.

---

## 4. Manifest section

Encoded as a custom binary stream. Strings and length-prefixed arrays
use `u32` for their length. Integer fields are little-endian. There is
no external schema: the decoder walks the fields in declaration order
and rejects any trailing bytes.

### 4.0 Manifest envelope

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 4 | version | manifest version (u32), currently `1` |
| 0x04 | 4 | reserved | MUST be zero |
| 0x08 | … | metadata | global `StringMap` (§4.1) |
| … | 4 | tensor_count | number of `TensorDescriptor` entries (u32) |
| … | … | tensors | `tensor_count` × `TensorDescriptor` (§4.2) |
| … | 4 | variant_count | number of `VariantDescriptor` entries (u32) |
| … | … | variants | `variant_count` × `VariantDescriptor` (§4.3) |
| … | 4 | graph_count | number of `GraphDescriptor` entries (u32) |
| … | … | graphs | `graph_count` × `GraphDescriptor` (§4.4) |
| … | 4 | asset_count | number of `AssetDescriptor` entries (u32) |
| … | … | assets | `asset_count` × `AssetDescriptor` (§4.5) |

### 4.1 `StringMap`
- `count` (u32)
- for each:
    - `key_len` (u32)
    - `key` (UTF-8 bytes)
    - `val_len` (u32)
    - `val` (UTF-8 bytes)

### 4.2 `TensorDescriptor`

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | … | name | length-prefixed UTF-8 string |
| … | 2 | dtype | `LogicalDtype` discriminant (§5) |
| … | 2 | reserved | MUST be zero |
| … | 4 | rank | number of dimensions (u32) |
| … | rank*8 | shape | array of u64 dimensions |
| … | 4 | canonical | index of the canonical `VariantDescriptor` (u32) |
| … | 4 | count | number of packed variant indices (u32) |
| … | count*4 | packed | array of packed variant indices (u32) |
| … | 8 | shard_id | physical shard identifier (u64); see §4.6 |
| … | … | metadata | `StringMap` (per-tensor) |

### 4.3 `VariantDescriptor`

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 2 | target | `TargetTag` discriminant (§7) |
| 0x02 | 2 | encoding | `EncodingKind` (§6.1) |
| 0x04 | 2 | storage | `StorageDtype` (§6) |
| 0x06 | 2 | layout | `LayoutTag` (§6.2) |
| 0x08 | 4 | alignment | payload alignment, power of two |
| 0x0C | 4 | reserved | MUST be zero |
| 0x10 | 8 | offset | relative to the owning arena section start (u64) |
| 0x18 | 8 | length | variant size in bytes (u64) |
| 0x20 | 16 | checksum | BLAKE3 of the variant bytes, truncated to 16 bytes |
| 0x30 | 1 | kind | owning arena section kind: `2=CanonicalArena`, `3=PackedArena` |
| 0x31 | 1 | index | index among sections of this kind (0-based) |
| 0x32 | 2 | reserved | MUST be zero |
| 0x34 | … | meta | `VariantMeta` (§4.3.1) |

Every tensor has exactly one canonical variant plus zero or more packed
variants. Canonical variants live in the single `CanonicalArena`; packed
variants are grouped into `PackedArena` sections by `ArenaGroup` (§7.1)
so CPU and GPU payloads stay in separate sections with independent
alignment.

#### 4.3.1 `VariantMeta`

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 2 | block_rank | `u16` number of dimensions in `block_shape` |
| 0x02 | block_rank*8 | block_shape | block-quantisation block shape (u64 × rank) |
| … | 4 | group_size | grouped-quant group size (u32); `0` means unused |
| … | 2 | scale_dtype | `StorageDtype` of scales, or `0xFFFF` sentinel for "absent" |
| … | 2 | zero_point_dtype | `StorageDtype` of zero points, or `0xFFFF` sentinel |
| … | 4 | reserved | MUST be zero |
| … | … | extra | `StringMap` of free-form per-variant metadata |

### 4.4 `GraphDescriptor`

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 1 | kind | `GraphKind`: `1=Onnx`, `2=Ort`, `3=Other` |
| 0x01 | 1 | reserved | MUST be zero |
| 0x02 | 2 | reserved | MUST be zero |
| 0x04 | 4 | alignment | payload alignment, power of two |
| 0x08 | 8 | offset | offset inside the `Graph` section payload (u64) |
| 0x10 | 8 | length | graph payload length in bytes (u64) |
| 0x18 | 16 | checksum | BLAKE3 of the graph payload, truncated to 16 bytes |
| 0x28 | … | metadata | `StringMap` |

### 4.5 `AssetDescriptor`

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | … | name | length-prefixed UTF-8 string (unique within the file) |
| … | 4 | reserved | MUST be zero |
| … | 4 | alignment | payload alignment, power of two |
| … | 8 | offset | offset inside the `Assets` section payload (u64) |
| … | 8 | length | asset payload length in bytes (u64) |
| … | 16 | checksum | BLAKE3 of the asset payload, truncated to 16 bytes |
| … | … | metadata | `StringMap` |

### 4.6 `shard_id` and multi-file sharding

`TensorDescriptor.shard_id` identifies which physical byte buffer holds
the variant bytes for that tensor:

- `shard_id == 0` — the default. Variant bytes live in the master
  file's arenas as per §3 and the variant's `section_relative_offset` is
  resolved against the owning `SectionDescriptor.offset`.
- `shard_id != 0` — variant bytes live in an external **shard file**.
  Shards are raw arena byte buffers: the variant lives at
  `shard_bytes[offset..offset + length]`, indexed by `offset` alone
  (the master's `SectionDescriptor.offset` is not applied). The master
  still carries a `SectionDescriptor` for the arena so structural
  validation passes; the bytes at that position in the master are
  placeholder and will not be read once a shard is attached.

v1 does **not** mandate or define a writer-side sharding API. Producers
wanting sharded output must manually build the master's placeholder
arena and emit each shard as a raw byte buffer. A first-class writer
API and a reference packing scheme are explicitly deferred to a future
format version.

Readers that do not support sharding MUST fail open with a typed error
on any tensor whose `shard_id != 0` when the corresponding shard has
not been attached, rather than silently falling back to the master's
placeholder bytes.

---

## 5. Logical Dtypes

| Discriminant | Name | Size (bytes) |
|---|---|---|
| 1 | `U8`   | 1 |
| 2 | `I8`   | 1 |
| 3 | `U16`  | 2 |
| 4 | `I16`  | 2 |
| 5 | `U32`  | 4 |
| 6 | `I32`  | 4 |
| 7 | `U64`  | 8 |
| 8 | `I64`  | 8 |
| 9 | `F16`  | 2 |
| 10 | `BF16` | 2 |
| 11 | `F32`  | 4 |
| 12 | `F64`  | 8 |
| 13 | `Bool` | 1 (`0 = false`, anything else is rejected) |

Unknown discriminants MUST be rejected with a `Structural` error.

---

## 6. Storage Dtypes

Discriminants `1..=13` are identical to `LogicalDtype` (§5) and name a
raw little-endian, row-major layout. Discriminants `>= 100` name
block-quantised or byte-level float formats that require a decode step
before the logical tensor can be observed.

| Discriminant | Name | Block size | Bytes per block | Notes |
|---|---|---|---|---|
| 100 | `Q4_0`    |  32 |  18 | llama.cpp legacy, 4-bit linear, `[f16 scale][16B qs]`. |
| 101 | `Q8_0`    |  32 |  34 | 8-bit linear, `[f16 scale][32B qs]`. (Older specs called this `Int8Block`.) |
| 102 | `Q3K`     | 256 | 110 | llama.cpp K-quant, 3-bit. |
| 103 | `NF4`     |  32 |  18 | QLoRA NormalFloat 4, `[f16 scale][16B qs]` with spec-defined dequant levels. |
| 104 | `Fp8E4M3` |   1 |   1 | OCP FP8 E4M3 (bias 7, no infinity, `S.1111.111` → NaN). |
| 105 | `Q4K`     | 256 | 144 | llama.cpp K-quant, 4-bit. |
| 106 | `Q5_0`    |  32 |  22 | llama.cpp legacy, 5-bit. |
| 107 | `Q5K`     | 256 | 176 | llama.cpp K-quant, 5-bit. |
| 108 | `Q6K`     | 256 | 210 | llama.cpp K-quant, 6-bit (near-lossless). |
| 109 | `Q2K`     | 256 |  84 | llama.cpp K-quant, 2-bit (extreme compression). |
| 110 | `Fp8E5M2` |   1 |   1 | OCP FP8 E5M2 (bias 15, IEEE-754-shaped, supports ±∞ and NaN). |

Unknown discriminants MUST be rejected with a `Structural` error. The
absent-dtype sentinel `0xFFFF` is reserved for `VariantMeta` and MUST
NOT appear as a storage dtype itself.

### 6.1 `EncodingKind`

| Discriminant | Name | Meaning |
|---|---|---|
| 0 | `Raw`            | Storage bytes equal logical bytes (no transformation). |
| 1 | `CastF16`        | Lossy f32 → f16 cast; decoded on load to f32. |
| 2 | `BlockQuantized` | Block-quantised layout per the `StorageDtype` above. |

Unknown kinds MUST be rejected with a `Structural` error.

### 6.2 `LayoutTag`

| Discriminant | Name | Meaning |
|---|---|---|
| 0 | `RowMajor`        | C-contiguous. |
| 1 | `Blocked`         | Block-quantisation block shape is in `VariantMeta.block_shape`. |
| 2 | `TileInterleaved` | WMMA / tensor-core / WGSL-fragment tile packing. Tile shape is in `VariantMeta.block_shape`. Consumers unable to handle the layout MUST surface `RsmfError::Unsupported` rather than re-interpret the bytes as row-major. |

Unknown tags MUST be rejected with a `Structural` error.

---

## 7. TargetTag

| Discriminant | Name          | Arena group (§7.1) |
|---|---|---|
|  0 | `Canonical`   | — (canonical arena) |
|  1 | `CpuGeneric`  | `Cpu` |
|  2 | `CpuAvx2`     | `Cpu` |
|  3 | `CpuAvx512`   | `Cpu` |
|  4 | `CpuNeon`     | `Cpu` |
|  5 | `Wgpu`        | `Wgpu` |
|  6 | `Cuda`        | `Cuda` |
|  7 | `Metal`       | `Metal` |
|  8 | `Vulkan`      | `Wgpu` (shares memory-layout expectations) |
|  9 | `RocmHip`     | `Cuda` (shares CUDA-style memory + alignment) |
| 10 | `Tpu`         | `Tpu` |
| 11 | `CpuSve`      | `Cpu` |
| 12 | `CpuRiscvV`   | `Cpu` |

The enum is additive: new values are appended at increasing indices;
old readers reject unknown values with a `Structural` error (non-silent).
This means the enum can grow without a format version bump, at the
cost of requiring matching reader updates before a producer can emit
the new tag.

### 7.1 `ArenaGroup`

Packed variants are bucketed into per-group `PackedArena` sections so
CPU and GPU payloads can live in separate sections with independent
alignment. Wire values are the group index:

| Index | Name | Members |
|---|---|---|
| 0 | `Cpu`   | CpuGeneric, CpuAvx2, CpuAvx512, CpuNeon, CpuSve, CpuRiscvV |
| 1 | `Wgpu`  | Wgpu, Vulkan |
| 2 | `Cuda`  | Cuda, RocmHip |
| 3 | `Metal` | Metal |
| 4 | `Tpu`   | Tpu |

---

## 8. Adapters

LoRA / DoRA / IA³-style adapters are stored as **regular tensors**
annotated with an `adapter.*` metadata convention. There is no
dedicated section kind or bespoke codec — a writer that follows the
convention produces a file that streams, compresses, verifies, and
round-trips through readers that don't understand adapters.

See `docs/CONVENTIONS.md` → *Adapter-level* for the full key list
(`adapter.name`, `adapter.kind`, `adapter.role`, `adapter.rank`,
`adapter.alpha`, `adapter.target`, plus the file-level
`base_model.*` keys). Readers that want to resolve a delta like
`W + (α / r) · B · A` without parsing metadata strings can walk
`RsmfFile::adapters()` / `rsmf.RsmfFile.adapters()`.

---

## 9. Writer paths and feature matrix

Two writers produce v1-conformant files. Readers cannot distinguish
them from the on-disk contents alone (the section table lists sections
in ascending offset order either way), but their capabilities differ
and the feature matrix is load-bearing for users deciding which to use.

| Feature                                  | Batch writer (`RsmfWriter`) | Streaming writer (`StreamingRsmfWriter`) |
|---|---|---|
| Multiple canonical tensors                | yes | yes |
| Packed variants                           | yes | **no** — reserved for a future release |
| Quantisation helpers (`quantize_q4_0`, …) | yes | **no** |
| `cast_f16` packed variant                 | yes | **no** |
| ONNX / ORT graph bodies                   | yes | yes |
| Assets                                    | yes | yes |
| Global manifest metadata                  | yes | yes |
| Per-tensor / per-variant / per-graph / per-asset metadata | yes | **no** — reserved for a future release |
| zstd compression of arenas / graph / assets | yes (`compression` feature) | yes (`compression` feature) |
| Bit-shuffle pre-processor for compressed f32 arenas (sets `SECTION_FLAG_BIT_SHUFFLED`) | yes | **no** — streaming path cannot bit-shuffle; flag is left clear and the reader skips the un-shuffle step |
| Content-addressable dedup of identical variant bytes | yes (opt-in) | **no** |
| Multi-file shard-aware output (`shard_id != 0`) | **no** in v1 | **no** in v1 |
| Peak RSS independent of tensor size       | no — every payload must fit in memory | **yes** — payloads stream directly through `Read` handles |
| On-disk section order                     | Manifest → Canonical → Packed → Graph → Assets | Canonical → Graph → Assets → Manifest |

The streaming writer's manifest comes last so it can record the final
(post-compression) offsets for previously-written sections. The
preamble and section table are placeholders that get patched at
`finish()` with the true values once all sections are on disk.

Both writers must produce files for which every section's `offset` is
aligned to the section's declared `align`, no two sections overlap,
and every variant's recorded checksum matches the bytes at
`section_relative_offset..section_relative_offset + length` inside its
owning arena.

Callers who need a feature the streaming writer lacks (packed variants,
quantisation, tensor-level metadata) must use the batch writer. There
is no plan in v1 to rewrite streaming output with those features via a
post-process; a later minor release may add it.
