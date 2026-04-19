# RSMF — Assumptions

Every assumption that the RSMF ecosystem takes as given.

## Tensor data

- **Logical dtypes:** `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `f16`, `bf16`, `f32`, `f64`, `bool`.
- **Storage dtypes:** Includes all logical dtypes plus specialized quantized formats like `Q4_0`.
- **Sharding:** Supported via `shard_id`. Tensors can live in external shard files.
- **Compression:** Supported via Zstd and bit-shuffling.

## Graph payload

- Graph bytes are opaque. Multiple graphs can be stored per file.
- `rsmf-runtime` provides a reference implementation for executing these graphs via ONNX Runtime.

## Encodings

- `Raw`, `CastF16`, and `BlockQuantized` (INT8, Q4_0) are supported for end-to-end decoding.

## GPU backends

- **Native acceleration**: WGPU, CUDA, and Metal are supported with dedicated crates.

## Python bindings

- The `rsmf` Python module is built with PyO3 `v0.22` and `numpy` `v0.22`, using
  `abi3-py38` — a single compiled extension works on every Python ≥ 3.8.
- NumPy has no stable `float16` / `bfloat16` dtype on CPU, so `F16` and `BF16`
  tensors surface as `float32` arrays via the SIMD-accelerated dequant path.
- Building the extension requires `maturin`:
  `maturin develop -m crates/rsmf-python/Cargo.toml --release`.
- The bindings are not built as part of the default workspace (they live
  outside `default-members`) to keep pure-Rust workflows hardware- and
  Python-free.

## Testing and CI

- Every test fixture is synthesised at test time.
- The project maintains a `justfile` for standardized development tasks.
