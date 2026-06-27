# rsmf — Rust Split Model Format

[![CI](https://github.com/kidoz/rsmf/actions/workflows/ci.yml/badge.svg)](https://github.com/kidoz/rsmf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![Edition: 2024](https://img.shields.io/badge/edition-2024-dea584.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![Python: 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![Format: v1.0](https://img.shields.io/badge/format-v1.0-brightgreen.svg)](docs/SPEC.md)

A Rust-native binary container for machine-learning model tensors with:

- one canonical little-endian, row-major, uncompressed tensor variant per logical tensor,
- zero or more optional packed variants tagged by backend (`cpu_generic`, `cpu_avx2`, `cpu_avx512`, `cpu_neon`, `wgpu`, `cuda`, `metal`),
- mmap-backed CPU loading for low peak RSS,
- a portable WGPU upload path,
- opaque ONNX / ORT graph payload (no new graph IR),
- BLAKE3 integrity checks, quick or full,
- a CLI for inspect / verify / pack / extract / select / placement,
- PyO3 bindings with NumPy-friendly tensor access (dequantizes Q4_0 / Q8_0 / NF4 / F16 / BF16 to `float32`).

Format version **1.0**. Full specification in [`docs/SPEC.md`](docs/SPEC.md); architecture in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); format decisions in
[`docs/FORMAT_DECISIONS.md`](docs/FORMAT_DECISIONS.md); benchmark methodology in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md); assumptions in
[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

## Workspace layout

| Crate | Role | In `default-members` |
|---|---|---|
| `rsmf-core` | library — format, reader, writer, validator, selection | yes |
| `rsmf-cli` | `rsmf` command-line binary | yes |
| `rsmf-bench` | criterion benchmarks | yes |
| `rsmf-wgpu` | portable WGPU chunked-staging upload path | no |
| `rsmf-cuda` | synchronous CUDA host→device upload helper | no |
| `rsmf-metal` | synchronous Metal host→GPU upload helper | no |
| `rsmf-moe-runtime` | experimental placement-aware MoE runtime PoC | no |
| `rsmf-runtime` | production-oriented ONNX Runtime inference engine | no |
| `rsmf-python` | PyO3 bindings | no |

GPU / runtime / Python crates are excluded from `default-members` so `cargo build` and `cargo test` stay fast and hardware-free.

## Quick start (Rust)

Build and test the default members (no GPU, no runtime, no Python). These
are the three crates listed under `default-members` in the root
`Cargo.toml` — `rsmf-core`, `rsmf-cli`, and `rsmf-bench`:

```sh
cargo build
cargo test
```

To exercise the full workspace including the GPU / runtime / Python
crates (each of which pulls in its own SDK — `wgpu`, `cudarc`, `metal`,
`ort`, `pyo3`), run:

```sh
cargo build --workspace
cargo test  --workspace
```

Install the `rsmf` CLI locally:

```sh
cargo install --path crates/rsmf-cli
```

### Pack from safetensors, GGUF, NumPy, or PyTorch

```sh
rsmf pack --from-safetensors model.safetensors --out model.rsmf
rsmf pack --from-gguf        model.gguf        --out model.rsmf
rsmf pack --from-npy         embeddings.npy    --out embeddings.rsmf
rsmf pack --from-torch       checkpoint.pt     --out model.rsmf
```

`--from-gguf` imports GGUF tensors byte-for-byte: raw dtypes
(`F32/F16/BF16/F64/I8/I16/I32/I64`) are stored with matching
`LogicalDtype`, and the standard quantised formats
(`Q4_0`, `Q5_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`) are
stored as canonical `BlockQuantized` variants with their original
bytes and dequantised to `float32` on demand via `get_tensor()`.
The legacy (`Q4_1`, `Q5_1`, `Q8_1`) and IQ* quantisations have no RSMF
decoder yet and fail pack with a typed error rather than being
silently dropped.

`--from-torch` shells out to `python3` with `torch` and `safetensors`
installed; it uses `torch.load(..., weights_only=True)` so arbitrary-code
execution on pickle load is blocked by default. Override the interpreter
with `RSMF_PYTHON_BIN=/path/to/python3`; set `RSMF_ALLOW_UNSAFE_PICKLE=1`
only for checkpoints you trust that the safe loader rejects.

### Stream multi-hundred-GB checkpoints without buffering

```sh
rsmf pack --stream --from-safetensors huge_model.safetensors --out huge_model.rsmf
```

`--stream` bypasses the buffered pack path and pipes tensor bytes
directly to disk as each tensor is read. Peak RSS stays in the low-MB
range regardless of source size, so a 70B-parameter (~140 GB) checkpoint
packs on a laptop with tens of MB of free RAM. `--graph`, `--asset`,
and `--compress-*` all work under `--stream` — a full model bundle
(weights + ONNX graph + tokenizer + configs, optionally zstd-compressed)
is packable in one pass without ever holding a section in memory.
The streaming path skips the bit-shuffle pre-processor the batch writer
uses, so compression ratios are slightly lower; in exchange, packing
scales to arbitrary source size. Not yet compatible with `--quantize-*`
or `--cast-f16`.

```sh
rsmf pack --stream --from-safetensors model.safetensors \
          --graph intent_classifier.onnx \
          --asset tokenizer.json \
          --asset config.json \
          --out bundle.rsmf
```

### Add packed variants during pack

```sh
# Add an NF4 / Q4_0 / Q8_0 / F16 / FP8 variant tagged by backend.
rsmf pack --from-npy embeddings.npy \
          --quantize-q4_0  cpu_generic \
          --compress-tensors \
          --out embeddings.rsmf
```

### Inspect / verify / select / extract

```sh
rsmf inspect  model.rsmf
rsmf inspect  model.rsmf --moe --prefetch
rsmf verify   model.rsmf --full
rsmf select   model.rsmf --mode cpu
rsmf select   model.rsmf --mode gpu --tier nvme --assume-wgpu
rsmf extract  model.rsmf --tensor embedding.weight out.bin
rsmf extract-asset model.rsmf --name tokenizer.json tokenizer.json
rsmf export safetensors model.rsmf model.safetensors
rsmf export onnx bundle.rsmf graph.onnx
```

Use `--target <tag>` to export a packed variant instead of the canonical
variant, and `--decode-f32` when that variant is stored in a packed /
quantized representation. Raw row-major tensors export byte-for-byte; decoded
tensors are materialized as F32 safetensors payloads.

### Shard a packed model

```sh
rsmf shard model.rsmf --by size --shards 2 --out-dir ./sharded
rsmf verify ./sharded/master.rsmf --full \
            --shard 1=./sharded/shard-1.bin \
            --shard 2=./sharded/shard-2.bin
```

`--by expert` groups tensors by `moe.layer` / `moe.expert`, `--by tier` groups
by the first `tier.intent` found on a tensor's variants, and `--by size`
greedily balances total variant bytes. The master keeps placeholder arena bytes;
full checksum verification needs every shard attached.

Per-variant `prefetch.group` / `prefetch.affinity` metadata is inspectable with
`rsmf inspect --prefetch`. These hints are metadata-only in v1 and are intended
for later runtimes that want to speculatively co-load expert or shard-local
variants.

### Placement manifests

Placement metadata is an optional `Custom(128)` section that tells runtimes
where shard ids should live. Add or replace it from a TOML plan:

```toml
[[devices]]
id = 0
kind = "cuda"
tier = "vram"
capacity_bytes = 25769803776

[[devices]]
id = 1
kind = "cpu"
tier = "ram"

[[placements]]
shard_id = 0
primary_device = 0
prefetch_priority = 10
flags = ["pin"]
replicas = [1]
```

```sh
rsmf placement set model.rsmf --plan placement.toml
rsmf placement inspect model.rsmf
```

### Production ORT runtime

`rsmf-runtime` is the first production-facing general inference engine for
embedded ONNX / ORT graph payloads. It keeps RSMF as a storage/container format
and owns runtime concerns: graph selection, ORT session options, CPU execution
provider defaults, cached sessions, graph input/output metadata, and typed
owned input/output tensors.

It adds explicit initializer bindings through `SessionOptions.initializers`,
mapping ONNX initializer names to RSMF tensor names. Bindings use canonical
tensors by default and can opt into a specific global RSMF variant index with
`InitializerBinding::with_variant`. The CPU path supports raw row-major `F32`,
`F64`, `I64`, `I32`, `U8`, `I8`, and `Bool` initializers, with ONNX
initializer dtype/shape preflight before ORT session creation. Unsupported
layouts or encodings fail with typed runtime errors. It avoids embedding
duplicate weight bytes in the graph payload, while the pinned ORT Rust API
still materializes an ORT-owned initializer value during session build.
`SessionHandle::memory_report` exposes graph payload bytes and
per-initializer materialized bytes.

The runtime server layer builds on `RuntimeExecutor`: a bounded
in-process priority queue around `Engine::run`. It supports FIFO ordering
within a priority level, higher-priority dispatch, pre-dispatch deadline
expiry, timeout helpers, queued cancellation, typed runtime error propagation,
opt-in dynamic batching on the leading tensor dimension, per-request queue/run
timings, queued tensor byte admission, per-tenant queued request/byte quotas,
live queue/active-runtime pressure metrics, continuous-batching flush metrics,
cumulative executor metrics, and best-effort interruption of running ORT calls
through `RunOptions::terminate`. A dependency-light HTTP/1.1 JSON serving
wrapper exposes health, metrics, synchronous inference, in-flight request
status, request cancellation, and tenant id propagation. It is intentionally
graph-runtime agnostic so native decoder execution can share the same control
plane later.

Build it explicitly:

```sh
cargo test -p rsmf-runtime
```

Provider-specific device I/O binding and true mmap/device zero-copy residency
are future runtime milestones.

### Minimal MoE runtime PoC

`rsmf-moe-runtime` is an experimental, non-default crate that validates the
MoE/sharding/placement/prefetch slice end to end. It runs host-side top-1
gating, batches tokens by destination expert, resolves each expert shard through
`PlacementManifest`, and compares the batched path against a single-device CPU
reference. Build it explicitly:

```sh
cargo test -p rsmf-moe-runtime
cargo test -p rsmf-moe-runtime --features wgpu
```

The `wgpu` feature runs expert matmuls through WGPU when an adapter is
available and falls back cleanly to CPU otherwise. `MoeRuntimeOptions::limits`
provides finite default guardrails for token batches, decoded tensors, and
output allocations.

### Rewrite: ship a smaller artifact by stripping dev-only variants / assets

```sh
# Drop the dev-time cpu_generic + wgpu variants before shipping.
rsmf rewrite dev.rsmf prod.rsmf \
             --strip-variants cpu_generic \
             --strip-variants wgpu

# Keep only canonical tensors — every packed variant goes away.
rsmf rewrite dev.rsmf canonical_only.rsmf --keep-only-canonical

# Drop bundled graph + a named asset + re-compress.
rsmf rewrite dev.rsmf small.rsmf \
             --strip-graphs \
             --strip-asset config.json \
             --compress-tensors --compress-assets
```

`rewrite` reads the source via the batch reader and writes through the
batch writer by default, so every byte passes through RAM once.
Use `--stream` to copy tensors, graphs, and assets directly to disk
without buffering them in RAM (implies `--keep-only-canonical`).

### Bundle weights, graph, and assets into one file

```sh
rsmf pack --from-npy embeddings.npy \
          --graph intent_classifier.onnx \
          --asset tokenizer.json \
          --asset config.json \
          --compress-tensors --compress-assets \
          --out bundle.rsmf
```

## Python bindings

Build the extension into your active environment:

```sh
pip install maturin
maturin develop -m crates/rsmf-python/Cargo.toml --release
```

### Usage

```python
import rsmf

f = rsmf.RsmfFile("model.rsmf")
f.verify()                              # full BLAKE3 pass; raises on mismatch

info = f.file_info()                    # dict: file_size, format_major/minor,
                                        # section_count, tensor_count,
                                        # variant_count, graph_count, asset_count
print(info)

# Tensor access — dequantizes Q4_0/Q8_0/NF4/F16/BF16 → f32 numpy array
weights = f.get_tensor("embeddings_cache")        # canonical variant
variants = f.tensor_variants("embeddings_cache")  # list of dicts with
                                                  # variant_idx, target, encoding,
                                                  # storage_dtype, layout, length
q4 = f.get_tensor_variant("embeddings_cache", variants[1]["variant_idx"])

# Graphs and assets
onnx_bytes = f.get_graph(0)                       # -> bytes
records    = f.get_asset("processed_cpes.json")   # -> bytes | None
```

F16 and BF16 tensors dequantize to `float32` NumPy arrays (no stable NumPy dtype for f16/bf16). All other logical dtypes (F32, F64, I8/I16/I32/I64, U8/U16/U32/U64) map to their native NumPy dtype.

## GPU backends

Build the WGPU upload crate (and its Criterion benchmark) when a GPU adapter is available:

```sh
cargo build -p rsmf-wgpu
cargo bench -p rsmf-bench --features wgpu   # skips gracefully if no GPU
```

CUDA (`rsmf-cuda`) and Metal (`rsmf-metal`) require their respective SDKs and are not built by default.

## Benchmarks

```sh
cargo bench -p rsmf-bench                   # default CPU sizes, laptop-friendly
cargo bench -p rsmf-bench --features large  # larger fixtures, opt-in
```

## Development tasks

A [`justfile`](justfile) bundles common workflows: `just fmt`, `just lint`, `just test`, `just check`, `just build`, `just bench`.

## License

MIT
