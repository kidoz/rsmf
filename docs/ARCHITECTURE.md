# RSMF — Architecture

## Crate graph

```
rsmf-core  (library)           ← format, reader, writer, validator, selection
   │
   ├── rsmf-cli   (bin)         ← `rsmf` command-line interface
   ├── rsmf-wgpu  (library)     ← portable WGPU upload path (excluded from default-members)
   ├── rsmf-cuda  (library)     ← native CUDA zero-copy path (excluded from default-members)
   ├── rsmf-metal (library)     ← native Metal zero-copy path (excluded from default-members)
   ├── rsmf-runtime (library)   ← high-level inference engine (ONNX Runtime / ort)
   ├── rsmf-python (library)    ← PyO3 bindings for Python / NumPy
   └── rsmf-bench (library+bench) ← criterion benchmarks
```

- `rsmf-core` has zero GPU dependencies. It handles the core format, zero-copy mmap reader, writer, and quantization logic.
- `rsmf-cli` provides the `rsmf` binary with `pack`, `import` (from HF), `inspect`, and `verify` commands.
- `rsmf-wgpu`, `rsmf-cuda`, and `rsmf-metal` provide materialization paths for different GPU backends. They are excluded from `default-members` for fast, dependency-free core builds.
- `rsmf-runtime` integrates the `ort` v2 crate to run inference on embedded graphs using memory-mapped tensors.
- `rsmf-python` enables high-performance access to RSMF models from Python. See the "Python surface" section below.

## Module map inside `rsmf-core`

| Module | Role |
| --- | --- |
| `preamble` | fixed 64-byte header: magic, version, pointers, self-checksum |
| `section` | `SectionKind` enum + 64-byte `SectionDescriptor` codec |
| `manifest` | versioned binary manifest: tensors / variants / graphs / assets |
| `tensor::dtype` | `LogicalDtype` and `StorageDtype` (including Q4_0, INT8) |
| `tensor::variant` | `TargetTag`, `EncodingKind`, `LayoutTag`, `VariantDescriptor`, `VariantMeta` |
| `tensor::descriptor` | logical `TensorDescriptor` with `shard_id` support |
| `tensor::view` | mmap-backed typed view with SIMD dequantization |
| `writer` | `RsmfWriter` builder with bit-shuffling compression |
| `reader` | `RsmfFile` mmap reader with multi-file sharding support |
| `bit_shuffle` | custom bit-shuffling pre-processor for improved compression |
| `safetensors_convert` | optional safetensors → RSMF converter |
| `gguf_convert` | optional GGUF → RSMF converter |
| `npy_convert` | optional NumPy (.npy) → RSMF converter |

## Data flow — write path

```
[TensorInput]      [GraphInput]     [AssetInput]
       \                |                /
        ▼               ▼               ▼
    RsmfWriter.encode()
       │
       │ 1. Quantize variants (INT8, Q4_0) if requested
       │ 2. Shuffle bits for floating-point data blocks
       │ 3. Zstd compress arenas and payloads
       │ 4. Build Manifest with shard-aware descriptors
       │ 5. Encode manifest and compute section checksums
       │ 6. Emit: preamble | section_table | manifest | arenas | graphs | assets
       ▼
   Atomic write: tempfile + rename
```

## Data flow — read path

```
   RsmfFile::open(path)
       │
       │ 1. open(path) + Mmap::map master file
       │ 2. Preamble & Section Table decode + validate
       │ 3. Manifest decode (tensors, variants, graphs, assets)
       │ 4. Optional: with_shard(id, mmap) to attach external weight files
       ▼
   RsmfFile {
     master_mmap, shard_mmaps,
     preamble, sections, manifest,
     caches for lazy decompression,
   }

   ───────────────────────────────────────────────────────────
   Accessors:
     tensor_view(name)      → TensorView (mmap-backed, SIMD dequantized)
     graph_payloads()       → Vec<GraphPayload> (lazily decompressed)
     select_variants(...)   → TensorPlan
     full_verify()          → Result<()> (re-hashes all shards and sections)
```

## Advanced Features

- **Quantization**: Native INT8 and Q4_0 (4-bit) block quantization.
- **Compression**: Bit-shuffled Zstd compression for maximum storage efficiency.
- **Sharding**: Support for massive models spanning multiple physical files.
- **Interop**: First-class support for Safetensors, GGUF, and NumPy.
- **Bindings**: High-performance Python access via PyO3.

## Python surface (`rsmf-python`)

Built with PyO3 v0.22 + `numpy` v0.22, `abi3-py38` so a single wheel works across Python ≥ 3.8.

The Rust `RsmfFile` is exposed as `rsmf.RsmfFile`. Available methods:

| Method | Returns | Notes |
|---|---|---|
| `RsmfFile(path)` | constructor | opens + structural-validates the file |
| `file_info()` | `dict` | `file_size`, `format_major`, `format_minor`, `section_count`, `tensor_count`, `variant_count`, `graph_count`, `asset_count` |
| `metadata()` | `dict[str, str]` | global manifest metadata |
| `verify()` | `None` / raises | full BLAKE3 pass across preamble, sections, variants, graphs, assets |
| `tensor_names()` | `list[str]` | |
| `get_tensor(name)` | `np.ndarray` | canonical variant; dequantizes quantized storage to `f32` |
| `tensor_variants(name)` | `list[dict]` | canonical first, then packed; each dict has `variant_idx`, `target`, `encoding`, `storage_dtype`, `layout`, `length` |
| `get_tensor_variant(name, variant_idx)` | `np.ndarray` | load a specific variant by its global index |
| `asset_names()` | `list[str]` | |
| `get_asset(name)` | `bytes` / `None` | |
| `graph_count()` | `int` | |
| `graph_kind(idx)` | `"onnx"` / `"ort"` / `"other"` | |
| `get_graph(idx)` | `bytes` | |

`F16` and `BF16` tensors surface as `float32` arrays because NumPy has no stable `float16` / `bfloat16` dtype. All other logical dtypes (F32, F64, I8/I16/I32/I64, U8/U16/U32/U64) map 1-1 to native NumPy dtypes.

## Integrity model

- **Structural**: O(manifest size). Every public `RsmfFile::open` performs it.
- **Full**: O(file size). Opt-in via `RsmfFile::full_verify` (Rust) or `RsmfFile.verify()` (Python). Validates all shards.

## Unsafe surface

The only `unsafe` blocks live at the mmap boundary. The SAFETY comments explain that we never offer write access to the mapping and use slice indexing for memory safety.
