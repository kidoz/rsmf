# RSMF — Format Decisions

Running record of design decisions.

## D1 — Versioned custom binary manifest
Deterministic little-endian byte stream.

## D2 — BLAKE3 truncations
16-byte for sections/variants, 8-byte for preamble. Sufficient for integrity checks.

## D3 — Compression and Sharding in v1
v1 formally supports zstd compression on all sections and multi-file sharding via `shard_id`. This enables deployment of massive (70B+) models and reduces distribution size.

## D4 — SIMD Acceleration
CPU-based dequantization is accelerated via the `wide` crate (f32x8/f32x4), ensuring high-performance fallback for non-GPU environments.

## D5 — Bit-Shuffling
A pre-processor is applied to floating-point data before Zstd compression to improve ratios by grouping similar bits.

## D6 — Unified Artifacts
RSMF prioritizes being a "one-file" (or one-master + shards) solution bundling weights, graphs (ONNX), and assets (tokenizer.json).

## D7 — Python binding surface
The `rsmf-python` crate exposes `RsmfFile` with two tensor-access paths:

- `get_tensor(name)` always returns the **canonical** variant. Quantized storage (`Q4_0`, `Q8_0`, `NF4`, `F16`, `BF16`) is dequantized to `float32`, matching the numeric behaviour of `TensorView::decode_f32`.
- `tensor_variants(name)` + `get_tensor_variant(name, variant_idx)` give callers explicit control over which variant to load (e.g. pick the `cpu_generic` Q4_0 variant to halve peak RSS).

Graph and asset accessors return `bytes`. No attempt is made to parse ONNX or to wrap it in a higher-level runtime object; that belongs to `rsmf-runtime` or the caller.

PyO3 is pinned to `v0.22` with `abi3-py38`, producing a single wheel that loads on Python ≥ 3.8. Build via `maturin develop -m crates/rsmf-python/Cargo.toml`.
