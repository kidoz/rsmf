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

## D11 — MoE metadata convention

Mixture-of-Experts identity is represented with `moe.*` metadata on regular
tensors, plus manifest-level `moe.n_experts`, `moe.top_k`, `moe.n_shared`, and
`model.arch` hints. This follows the adapter convention: no new section kind,
no manifest version change, and no computation graph semantics. Existing v1
readers preserve the keys and continue to read tensor bytes unchanged.

The typed `RsmfFile::moe_experts()` accessor validates decimal fields and
groups tensors by layer, expert id, shared flag, and role for runtimes that
need expert routing metadata.

Tensor-parallel MoE planning extends the same convention with optional
per-tensor keys (`moe.parallel=tensor`, partition axis/index/count, and
`moe.collective`). These keys declare tensor partitions and required
collectives without introducing graph semantics or changing the binary format.
Runtimes that cannot execute tensor-sliced experts must fail explicitly rather
than silently treating partitions as full expert tensors.

## D12 — PlacementManifest as Custom(128)

Shard/device placement is represented by an optional
`SectionKind::Custom(128)` payload rather than a promoted standard section kind.
The section carries a typed binary `PlacementManifest` with device descriptors,
per-shard placement records, and small runtime hints such as pin/cold flags.

This is intentionally additive: it does not change tensor bytes, manifest
descriptors, variant selection, or the format version. Readers that understand
the section validate it against `TensorDescriptor.shard_id`; readers that do
not understand it can preserve the custom payload and still read the model.
The batch writer can emit it directly, and `rsmf placement set` can append or
replace it without requiring external shard files.

## D13 — Tier intent as per-variant metadata

Tier / precision intent uses `VariantMeta.extra` keys instead of adding fields
to `VariantDescriptor`: `tier.intent` names the residency tier (`vram`, `ram`,
`nvme`) and `tier.class` carries an optional hot/warm/cold-style label. This
keeps the binary layout unchanged and preserves v1 compatibility.

The selector treats a requested tier as an additional preference layer over the
existing backend/mode score. If matching tier-tagged variants exist for a
tensor, the normal scorer runs over that subset; otherwise it falls back to the
pre-existing backend-only behavior. Files without tier metadata therefore select
exactly as before.

## D14 — Reference writer-side sharding scheme

Writer-side sharding is implemented as a reference repacker over the existing
`TensorDescriptor.shard_id` semantics, not as a new wire format. The master file
keeps structurally valid zero-filled tensor arenas, while variant descriptors for
sharded tensors point at shard-local offsets and keep checksums of the external
shard bytes. Full verification therefore requires attaching the shard files.

The shard-local offset space is shared across all variants assigned to a shard,
including canonical and packed variants. This avoids collisions that would occur
if each arena independently started at offset zero while the reader indexes raw
shard files by `section_relative_offset` alone. The existing `section_kind` and
`section_index` fields are preserved for structural validation and unsharded
fallback semantics.

No format minor bump is required: this promotes the deferred writer API for the
already-specified `shard_id != 0` path and preserves the existing reader
contract. The reference CLI provides `--by size`, `--by tier`, and `--by expert`
assignment strategies.

## D15 — Prefetch hints as per-variant metadata

Prefetch / locality hints use `VariantMeta.extra` keys instead of a runtime
section or descriptor fields: `prefetch.group` names an opaque co-loading /
co-eviction group, and `prefetch.affinity` carries comma-separated shard,
expert, tier, or writer-defined labels commonly co-active with the variant.

This keeps the format metadata-only. The reader exposes a typed
`prefetch_hints()` index for runtimes to consume later, but the v1 reader does
not prefetch, evict, transport, or schedule anything by itself. Existing readers
that ignore the keys continue to read variant bytes unchanged.

## D16 — Minimal MoE runtime remains outside the file format

The expert-parallel runtime is a separate, non-default crate
(`rsmf-moe-runtime`) rather than a new section, graph IR, or manifest field. It
consumes existing `moe.*`, `prefetch.*`, `tier.*`, `shard_id`, and
`PlacementManifest` data to validate the format slice end to end.

The proof of concept runs host-side top-1 gating, batches tokens by destination
expert, resolves expert shards through placement records, runs expert matmuls on
WGPU when the optional feature and an adapter are available, executes CPU
row-gather `down` projection partitions for the narrow tensor-sliced expert
prototype, and compares against a single-device CPU reference. WGPU support
falls back to CPU when adapters are unavailable. No format version bump is
required because the runtime adds no on-disk semantics.

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
  the safe loader refuses. Artifacts are marked with `source=torch`,
  `rsmf.source_format=torch`, `rsmf.intermediate_format=safetensors`,
  and `torch.*` provenance metadata. Pure-Rust pickle parsing is a follow-up.
- **ONNX `.onnx`** uses a `python3` subprocess with the `onnx` package to extract
  initializers (tensors) from the graph and save them to a safetensors
  temp file, which is then ingested.
- **HuggingFace imports** auto-bundle `config.json`, `generation_config.json`,
  `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
  `vocab.json`, `merges.txt`, `added_tokens.json`, `preprocessor_config.json`,
  and `chat_template.json` as rsmf assets when the remote repo carries them.
- **Currently shipped export paths:** `rsmf export safetensors` for canonical
  tensor export, plus `--target <tag>` for exporting a packed variant; and
  `rsmf export onnx` for byte-exact export of an embedded ONNX graph payload.
  Raw row-major tensors are exported byte-for-byte; packed, quantized, cast,
  and FP8 logical-F32 tensors require `--decode-f32` and are materialized as
  raw F32 safetensors tensors.
- **Next up**, in order: `--from-tflite`,
  `--from-h5`, and `rsmf export gguf`.
  Tier-C formats (TensorRT engines, Qualcomm SNPE, NCNN, MNN, GGML legacy)
  are explicitly out of scope because their bytes are hardware- or
  driver-bound and cannot be portably re-emitted.
