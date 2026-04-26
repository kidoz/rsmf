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

## D9 — SectionKind custom range + flag bit 1

Two format-level extensibility reservations that do not bump the
version:

- **`SectionKind::Custom(u16)`** maps any on-disk discriminant ≥ 128
  to a Custom variant that readers MUST preserve on round-trip. Values
  1–5 stay reserved for the standard's own sections; values 6–127 stay
  rejected with a structural error until the standard promotes one.
  Analogous to PNG's ancillary-chunk case-bit convention.
- **`SECTION_FLAG_BIT_SHUFFLED = 0x2`** on `SectionDescriptor.flags`
  explicitly records that the payload was passed through the
  bit-shuffle pre-processor before any further encoding / compression.
  The format has always done this inside compressed F32 sections; the
  flag makes the contract visible so consumers that want the
  un-shuffled bytes can opt out.

Both changes are additive. Old readers reject any non-zero unknown
reserved bit — so a new writer setting these bits on a file read by an
old reader fails loudly rather than silently misinterpreting.

## D10 — LayoutTag::TileInterleaved

`LayoutTag::TileInterleaved = 2` reserves space for WMMA / tensor-core
/ WGSL-fragment-friendly tile-packed layouts. Tile shape is carried in
`VariantMeta::block_shape`. Decoders that can't handle the layout MUST
return `RsmfError::Unsupported` rather than re-interpret the bytes as
row-major; the zero-copy Python path already guards on `RowMajor` and
is correct by construction.

## D8 — Source-format conversion priorities

The `rsmf pack` and `rsmf import` CLIs ingest from a fixed priority-ordered
list of source formats. The order was chosen against four criteria — reach,
quality/security win vs. the source, Rust dep cost, and fit with rsmf's
tensor/graph/asset model.

- **Currently shipped:** safetensors (`--from-safetensors`, used also by
  `rsmf import` on HuggingFace Hub), GGUF (`--from-gguf`), NumPy
  (`--from-npy`), PyTorch checkpoints (`--from-torch`), and ONNX
  (`--from-onnx`).
- **PyTorch `.pt` / `.pth` / `.bin`** uses a `python3` subprocess that
  calls `torch.load(..., weights_only=True)` and
  `safetensors.torch.save_file` into a temp file, then delegates to the
  existing safetensors pipeline. The safe loader blocks arbitrary code
  execution; `RSMF_ALLOW_UNSAFE_PICKLE=1` opts back in for trusted files
  the safe loader refuses. Pure-Rust pickle parsing is a follow-up.
- **ONNX `.onnx`** uses a `python3` subprocess with the `onnx` package to extract
  initializers (tensors) from the graph and save them to a safetensors
  temp file, which is then ingested.
- **HuggingFace imports** auto-bundle `config.json`, `generation_config.json`,
  `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
  `vocab.json`, `merges.txt`, `added_tokens.json`, `preprocessor_config.json`,
  and `chat_template.json` as rsmf assets when the remote repo carries them.
- **Next up**, in order: `--from-tflite`,
  `--from-h5`, and export paths (`rsmf export safetensors` / `gguf` / `onnx`).
  Tier-C formats (TensorRT engines, Qualcomm SNPE, NCNN, MNN, GGML legacy)
  are explicitly out of scope because their bytes are hardware- or
  driver-bound and cannot be portably re-emitted.
