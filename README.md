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
- a CLI for inspect / verify / pack / extract / select,
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
| `rsmf-wgpu` | portable WGPU upload path | no |
| `rsmf-cuda` | native CUDA zero-copy path | no |
| `rsmf-metal` | native Metal zero-copy path | no |
| `rsmf-runtime` | ONNX Runtime inference engine | no |
| `rsmf-python` | PyO3 bindings | no |

GPU / runtime / Python crates are excluded from `default-members` so `cargo build` and `cargo test` stay fast and hardware-free.

## Quick start (Rust)

Build and test the default workspace (no GPU, no Python):

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
rsmf verify   model.rsmf --full
rsmf select   model.rsmf --mode cpu
rsmf extract  model.rsmf --tensor embedding.weight out.bin
rsmf extract-asset model.rsmf --name tokenizer.json tokenizer.json
```

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
