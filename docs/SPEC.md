# RSMF — Binary Format Specification

**Status:** v1.0.
**Endianness:** little-endian on disk, everywhere.
**Integer widths:** all offsets, lengths, and counts are unsigned 64-bit unless noted.

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

---

## 2. Preamble (offset `0`, length `64`)

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 8 | magic | "RSMF\0\0\0\1" |
| 0x08 | 2 | major | format major version (u16) |
| 0x0A | 2 | minor | format minor version (u16) |
| 0x0C | 4 | flags | reserved (u32) |
| 0x10 | 8 | header_len | total preamble length (u64 = 64) |
| 0x18 | 8 | tbl_off | absolute offset of section table (u64) |
| 0x20 | 8 | tbl_count | number of sections (u64) |
| 0x28 | 8 | manifest_off| absolute offset of manifest payload (u64) |
| 0x30 | 8 | manifest_len| manifest length (u64) |
| 0x38 | 8 | checksum | BLAKE3 [0x00..0x38] truncated to 8 bytes |

---

## 3. Section table

Array of 64-byte entries starting at `tbl_off`.

| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 2 | kind | 1=Manifest, 2=Canonical, 3=Packed, 4=Graph, 5=Assets |
| 0x02 | 2 | align | required power-of-two alignment |
| 0x04 | 4 | flags | bit 0: compressed (zstd) |
| 0x08 | 8 | offset | absolute file offset (u64) |
| 0x10 | 8 | length | payload length in bytes (u64) |
| 0x18 | 16 | checksum | BLAKE3 of payload truncated to 16 bytes |
| 0x28 | 24 | reserved | zero-filled |

---

## 4. Manifest section

Encoded as a custom binary stream. Strings and arrays are prefixed with a `u32` length.

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
| 0x00 | ... | name | length-prefixed string |
| ... | 2 | dtype | LogicalDtype discriminant |
| ... | 2 | reserved | zero |
| ... | 4 | rank | number of dimensions (u32) |
| ... | rank*8| shape | array of u64 dimensions |
| ... | 4 | canonical | index of canonical variant (u32) |
| ... | 4 | count | number of packed variants (u32) |
| ... | count*4| packed | array of variant indices (u32) |
| ... | 8 | shard_id | physical file index (u64) |
| ... | ... | metadata | `StringMap` |

### 4.3 `VariantDescriptor`
| Offset | Size | Field | Description |
|---|---|---|---|
| 0x00 | 2 | target | TargetTag (Canonical=0, Wgpu=5, etc) |
| 0x02 | 2 | encoding | 0=Raw, 1=CastF16, 2=BlockQuantized |
| 0x04 | 2 | storage | StorageDtype |
| 0x06 | 2 | layout | 0=RowMajor, 1=Blocked |
| 0x08 | 4 | alignment| required power-of-two |
| 0x0C | 4 | reserved | zero |
| 0x10 | 8 | offset | relative to arena section start (u64) |
| 0x18 | 8 | length | variant size in bytes (u64) |
| 0x20 | 16 | checksum | BLAKE3 of variant bytes truncated to 16 bytes |
| 0x30 | 1 | kind | arena kind (1=Canonical, 2=Packed) |
| 0x31 | 1 | index | section index within that kind |
| 0x32 | 2 | reserved | zero |
| 0x34 | ... | meta | `VariantMeta` |

---

## 5. Logical Dtypes
1=U8, 2=I8, 3=U16, 4=I16, 5=U32, 6=I32, 7=U64, 8=I64, 9=F16, 10=BF16, 11=F32, 12=F64, 13=Bool.

## 6. Storage Dtypes
All logical dtypes (1-13) plus 100=Q4_0, 101=Int8Block.
