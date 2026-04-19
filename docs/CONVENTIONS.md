# RSMF — Conventions

Every RSMF struct that carries a free-form string map (manifest, tensor,
variant, graph, asset) is authoritative for any key it contains. This
document defines **conventions** — key namespaces and expected values
that tools agree on so they can interoperate without extending the
binary format.

None of these conventions require a format version bump. A writer that
follows them emits richer files; a writer that doesn't still produces
valid RSMF. Readers are expected to tolerate missing keys, unknown keys,
and unrecognised values.

Updates to this document never bump the format version. Updates to the
binary layout always do.

---

## Namespaces

Keys group under a prefix plus dot-separated segments. Keys without a
namespace are writer-specific and should be avoided in published files.

| Prefix | Owner | Where it appears |
|---|---|---|
| `rsmf.*` | The RSMF standard itself | Manifest metadata (global) |
| `quant.*` | Quantization schemes | Per-tensor, per-variant metadata |
| `weight.*` | Weight semantics | Per-tensor metadata |
| `shape.*` | Logical-vs-physical shape info | Per-tensor metadata |
| `variant.*` | Per-variant annotations | Per-variant metadata |
| `graph.*` | Graph-payload hints | Per-graph metadata |
| `asset.*` | Asset-payload hints | Per-asset metadata |
| `bench.*` | Measured performance | Per-variant metadata |
| `safetensors.*` | Verbatim safetensors metadata (import only) | Manifest metadata |
| `x-*` / `vendor.<name>.*` | Vendor extensions | Any; collision-free namespace |

---

## Manifest-level (`rsmf.*`)

| Key | Value | Example |
|---|---|---|
| `rsmf.creator` | Tool name + version that wrote this file. | `rsmf-cli 0.1.0` |
| `rsmf.source_format` | Original source this was converted from. | `safetensors`, `gguf`, `npy`, `torch`, `onnx`, `tflite` |
| `rsmf.source_sha256` | Hex-encoded SHA-256 of the source file. Provenance. | `1a2b3c...` |
| `rsmf.created_at` | RFC 3339 timestamp. | `2026-04-20T12:34:56Z` |
| `rsmf.license` | SPDX identifier. | `MIT`, `Apache-2.0`, `LLAMA2`, `other-custom` |
| `rsmf.model_architecture` | HF-style class name. | `LlamaForCausalLM`, `BertModel` |
| `rsmf.model_family` | Coarser bucket. | `llama`, `mistral`, `bert`, `sentence-transformer` |
| `rsmf.parameter_count` | Decimal string (total params). | `7241748480` |
| `rsmf.hash_algorithm` | Which truncated BLAKE3 (reserved for future). | `blake3-128` |

---

## Tensor-level (`quant.*`, `weight.*`, `shape.*`)

### `quant.*`

Applied when the tensor's bytes are quantized (INT8, NF4, AWQ, GPTQ, …).
Keys can also appear on variant metadata when quantization is per-variant.

| Key | Value |
|---|---|
| `quant.scheme` | `awq`, `gptq`, `bnb-nf4`, `bnb-int8`, `rtn`, `gptq-marlin`, `bitsandbytes`, `autoawq` |
| `quant.bits` | Decimal string (`4`, `5`, `6`, `8`). |
| `quant.group_size` | Decimal string. `-1` means channel-wise. |
| `quant.calibration_dataset` | `c4`, `wikitext-2`, `pile`, … |
| `quant.symmetric` | `true` / `false`. |
| `quant.per_channel_axis` | Integer axis index as decimal string. |
| `quant.version` | Scheme-specific version string. |

### `weight.*`

| Key | Value |
|---|---|
| `weight.role` | HF-style role. `attention.q_proj`, `attention.k_proj`, `attention.v_proj`, `attention.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `embeddings`, `lm_head`, `layernorm`, `layer_norm.weight`, `rotary_emb.inv_freq`. |
| `weight.layer_index` | Decimal string. |
| `weight.tied_to` | Name of another tensor whose bytes this shares. |

### `shape.*`

| Key | Value |
|---|---|
| `shape.original` | JSON array as string, the pre-fusion/pre-reshape shape. `[768,2304]` |
| `shape.block_major` | `true` if physical layout is blocked but logical row-major. |
| `shape.fused_from` | Comma-separated original tensor names that were fused. |

---

## Variant-level (`variant.*`, `bench.*`)

| Key | Value |
|---|---|
| `variant.created_with` | Specific tool + version that emitted this variant. |
| `variant.calibration_error` | Decimal string. Scheme-specific metric (L2, KL, SNR). |
| `bench.decode_ns_per_elem` | Measured decode speed. |
| `bench.env` | Free-form hardware/software fingerprint. `M2-Max macOS-15 rustc-1.85 release`. |

Per-variant `quant.*` keys may appear too (see above) when the variant
is a new quantization rather than a raw reinterpretation.

---

## Graph-level (`graph.*`)

| Key | Value |
|---|---|
| `graph.opset_version` | ONNX opset. `17`, `18`, `20`. |
| `graph.input_names` | Comma-separated input tensor names. |
| `graph.output_names` | Comma-separated output tensor names. |
| `graph.framework_version` | `onnx 1.17.0`, `ort 2.0.0-rc.12`. |
| `graph.exporter` | `torch.onnx.export`, `optimum`, `tf2onnx`. |

---

## Asset-level (`asset.*`)

| Key | Value |
|---|---|
| `asset.content_type` | MIME type. `application/json`, `text/markdown`, `application/x-jinja`, `text/plain`. |
| `asset.role` | `tokenizer`, `config`, `generation_config`, `chat_template`, `special_tokens_map`, `vocab`, `merges`, `preprocessor_config`, `license`, `model_card`, `other`. |
| `asset.sha256` | Hex-encoded content hash, for bundle-level verification. |
| `asset.encoding` | `utf-8`, `latin-1`, `binary`. |

---

## Recommended assets by role

Writers that emit HuggingFace-derived bundles SHOULD use these asset
names for cross-tool interop:

| Asset name | `asset.role` | `asset.content_type` |
|---|---|---|
| `config.json` | `config` | `application/json` |
| `generation_config.json` | `generation_config` | `application/json` |
| `tokenizer.json` | `tokenizer` | `application/json` |
| `tokenizer_config.json` | `tokenizer` | `application/json` |
| `special_tokens_map.json` | `special_tokens_map` | `application/json` |
| `vocab.json` | `vocab` | `application/json` |
| `merges.txt` | `merges` | `text/plain` |
| `added_tokens.json` | `tokenizer` | `application/json` |
| `preprocessor_config.json` | `preprocessor_config` | `application/json` |
| `chat_template.jinja` | `chat_template` | `application/x-jinja` |
| `license.txt` or `LICENSE` | `license` | `text/plain` |
| `README.md` | `model_card` | `text/markdown` |

---

## Vendor extensions (`x-*`, `vendor.<name>.*`)

Vendors wanting to attach proprietary metadata MUST either:

- Use a key prefix `x-<vendor>.*` (e.g. `x-huggingface.source_url`), or
- Use `vendor.<name>.*` (e.g. `vendor.kidoz.internal_sha256`).

Readers outside the vendor MUST preserve unknown vendor keys on
round-trip (do not drop them when re-serialising).

---

## Unknown-value behaviour

The RSMF reader preserves every string map verbatim. When a
*format-level* enum (e.g. `StorageDtype`, `TargetTag`) encounters an
unknown discriminant it returns `RsmfError::Structural` — the format
is strict about enum growth. When a *convention-level* metadata value
is unknown (e.g. `asset.role = "tokenizer_v2"`) readers MUST accept
the string unchanged and pass it through.

Format-level growth requires ADRs (`docs/FORMAT_DECISIONS.md` +
`.agents/contexts/decisions/`). Convention-level growth requires only
this document to be updated.

---

## Version and evolution

This document is versioned by its latest `last_updated` date in the
header of each section below; no monotonic version number is assigned.
Additive changes (new keys, new recommended values for an existing key)
do not require writer updates — writers that don't emit the new keys
still produce valid files. Breaking changes (renaming a key, changing
the semantics of an existing value) require a bump of the conventions
document version and a parallel support period documented here.

Changes to the key/value tables above MUST be paired with:
- a commit updating `docs/CONVENTIONS.md`,
- a note in `docs/FORMAT_DECISIONS.md` when the change meaningfully
  affects downstream consumers,
- no changes to `docs/SPEC.md` — the binary format is conventions-blind.
