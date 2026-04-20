"""Typed views over the dicts returned by the native extension.

These are :class:`typing.TypedDict` declarations — at runtime the values
are plain ``dict`` instances, but type-checkers see the named fields.
"""

from __future__ import annotations

from typing import TypedDict


class FileInfo(TypedDict):
    """Return type of :meth:`rsmf.RsmfFile.file_info`."""

    file_size: int
    format_major: int
    format_minor: int
    section_count: int
    tensor_count: int
    variant_count: int
    graph_count: int
    asset_count: int


class TensorInfo(TypedDict):
    """Return type of :meth:`rsmf.RsmfFile.tensor_info`."""

    name: str
    dtype: str
    shape: list[int]
    element_count: int
    shard_id: int
    canonical_variant_idx: int
    packed_variant_idxs: list[int]
    metadata: dict[str, str]


class VariantInfo(TypedDict):
    """An entry in the list returned by
    :meth:`rsmf.RsmfFile.tensor_variants`."""

    variant_idx: int
    local_idx: int
    is_canonical: bool
    target: str  # "canonical" | "cpu_generic" | "cpu_avx2" | "cpu_avx512"
    #          | "cpu_neon" | "wgpu" | "cuda" | "metal"
    encoding: str  # "raw" | "cast_f16" | "block_quantized"
    storage_dtype: str  # e.g. "f32", "f16", "q4_0", "q8_0", "nf4", "q3_k"
    layout: str  # "row_major" | "blocked"
    length: int
    alignment: int


class AdapterEntry(TypedDict):
    """One tensor that participates in an adapter."""

    tensor_name: str
    role: str  # "lora_a" | "lora_b" | "magnitude" | "scale" | "base_weight" | other
    target: str | None  # tensor in the base model this adapter modifies


class Adapter(TypedDict):
    """A single named adapter (LoRA / DoRA / IA³)."""

    name: str
    kind: str  # "lora" | "lora_plus" | "dora" | "ia3" | other
    rank: int | None
    alpha: float | None
    effective_scale: float | None  # alpha / rank, if both are set
    entries: list[AdapterEntry]


class AdapterIndex(TypedDict):
    """Return type of :meth:`rsmf.RsmfFile.adapters`."""

    base_model_name: str | None
    base_model_sha256: str | None
    adapters: list[Adapter]
