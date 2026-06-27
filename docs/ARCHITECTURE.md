# RSMF — Architecture

## Crate graph

```
rsmf-core  (library)           ← format, reader, writer, validator, selection
   │
   ├── rsmf-cli   (bin)         ← `rsmf` command-line interface
   ├── rsmf-wgpu  (library)     ← portable WGPU chunked-staging upload path (excluded from default-members)
   ├── rsmf-cuda  (library)     ← synchronous CUDA host→device upload helper (excluded from default-members)
   ├── rsmf-metal (library)     ← synchronous Metal host→GPU upload helper (excluded from default-members)
   ├── rsmf-moe-runtime (library) ← experimental placement-aware MoE runtime PoC (excluded from default-members)
   ├── rsmf-runtime (library)   ← production-oriented inference engine (ONNX Runtime / ort)
   ├── rsmf-python (library)    ← PyO3 bindings for Python / NumPy
   └── rsmf-bench (library+bench) ← criterion benchmarks
```

- `rsmf-core` has zero GPU dependencies. It handles the core format, mmap reader, writer, and quantization logic.
- `rsmf-cli` provides the `rsmf` binary with `pack`, `import` (from HF), `inspect`, and `verify` commands.
- `rsmf-wgpu` performs chunked, alignment-aware staging uploads through
  `wgpu::Queue::write_buffer`. `rsmf-cuda` and `rsmf-metal` are deliberately
  thin: they wrap a single blocking host→device byte copy
  (`cudarc::htod_sync_copy`, `MTLBuffer::new_buffer_with_data`) and do **not**
  perform zero-copy, streaming, or backend-specific layout translation. All
  three are excluded from `default-members` for fast, dependency-free core
  builds. Real vendor zero-copy paths remain future work.
- `rsmf-runtime` integrates the `ort` v2 crate to run embedded ONNX / ORT graph
  payloads with explicit graph selection, configurable session options, CPU as
  the portable default execution provider, cached sessions, graph
  input/output metadata, typed owned input/output tensors, and explicit
  ONNX-initializer-to-RSMF-tensor bindings for CPU residency. Bindings default
  to canonical tensors and may select a specific global RSMF variant index.
  The initializer path supports raw row-major `F32`, `F64`, `I64`, `I32`,
  `U8`, `I8`, and `Bool` tensors. ONNX initializer dtype/shape metadata is
  preflighted before ORT session creation. `SessionHandle::memory_report()`
  exposes graph payload bytes and per-initializer materialized bytes. The
  current initializer path avoids graph-embedded duplicate weight bytes but
  materializes an ORT-owned value at session build time; mmap/device zero-copy
  remains future work. `RuntimeExecutor` adds the first runtime-server control
  layer: a bounded in-process priority queue around `Engine::run`, with FIFO
  ordering within a priority level, pre-dispatch deadline expiry, timeout
  helpers, queued cancellation, typed error propagation, per-request queue/run
  timings, and opt-in continuous batching on the leading tensor dimension. It
  reports live queue depth, queued input bytes, active runtime invocations,
  active requests, batch pressure, scheduler flush reasons, and cumulative
  metrics, with optional queued tensor byte admission, soft/hard queued-memory
  pressure policy, and per-tenant queued request/byte quotas. Already-running
  ORT calls receive a best-effort `RunOptions::terminate` signal through request
  cancellation tokens.
  `RuntimeNetworkServer` exposes a std-library HTTP/1.1 JSON wrapper for
  `/health`, `/metrics`, `POST /v1/run`, `GET /v1/requests/{id}`, and
  `DELETE /v1/requests/{id}` without adding a web framework dependency. It
  supports protocol version fields, configurable header/body/response size
  limits, connection timeouts, sanitized error responses, and JSON `tenant_id`
  propagation for executor quota accounting. It stays graph-runtime agnostic for
  later native decoder paths. The native decoder track begins with
  `Engine::native_decoder_contract()`, which validates a LLaMA-style
  `config.json`, required `tokenizer.json`, optional `generation_config.json`,
  and the expected ordinary RSMF tensor names, shapes, and weight dtypes without
  adding a graph IR. The CPU reference decoder block path implements RMSNorm,
  row-major projections, RoPE, causal grouped-query attention, SwiGLU MLP, and
  residual additions over supplied f32 buffers. `Engine::native_decoder_weights()`
  loads validated RSMF tensor variants into owned f32 buffers, and
  `Engine::native_decoder_greedy_decode()` provides an owned KV cache, logits,
  EOS-aware generation, deterministic sampling controls, and a backend selector.
  `auto` resolves to CPU reference, while `accelerated` currently dispatches to
  the in-tree threaded CPU logits path. `Engine::native_decoder_check_reference_logits()`
  compares runtime logits against local or exported references, and the first
  performance slice adds page-sized KV-cache allocation accounting plus threaded
  final projection without adding GPU dependencies.
- `rsmf-moe-runtime` is a proof-of-concept runtime for one MoE layer: host-side
  top-1 gating, token batching by destination expert, placement-aware expert
  shard lookup, WGPU expert matmuls when available, and a CPU reference path.
  Its optional `wgpu` feature reports CPU fallback when no adapter is available.
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
