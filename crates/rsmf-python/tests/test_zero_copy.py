"""Verify the zero-copy raw-tensor path.

The rsmf-python extension routes `get_tensor` / `get_tensor_variant`
through two paths depending on the variant:

- **zero-copy** for `encoding=Raw`, `layout=RowMajor`, aligned, non-empty
  tensors with a dtype that has a native NumPy typestr. The returned
  NumPy array is **read-only** and its `.base` chain holds a reference
  to the native `RsmfFile` so the mmap stays alive.
- **copy** for everything else (quantized variants, BF16/F16 logical
  dtype, misaligned payloads, empty tensors, blocked layouts). The
  returned array is writable and owns its memory.

These tests pin both contracts so a future refactor that accidentally
copies the raw path, or accidentally zero-copies the quantized path,
fails CI immediately.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest

import rsmf


# -- uncompressed fixture: pointer lives inside the mmap ---------------


def test_raw_canonical_is_read_only(sample_rsmf_uncompressed: Path) -> None:
    with rsmf.RsmfFile(sample_rsmf_uncompressed) as model:
        arr = model.get_tensor("weights")
        assert arr.flags.writeable is False, (
            "zero-copy raw path must return a read-only array; "
            "writable=True means the extension silently copied."
        )


def test_raw_canonical_has_base_chain(sample_rsmf_uncompressed: Path) -> None:
    with rsmf.RsmfFile(sample_rsmf_uncompressed) as model:
        arr = model.get_tensor("weights")
        # NumPy's asarray sets `.base` to the object that supplied the
        # memory — in our case the internal ZeroCopyView. The chain must
        # be non-None or the array is a detached copy.
        assert arr.base is not None, (
            "zero-copy raw path must expose a .base reference; "
            "base=None means the array is a detached copy."
        )


def test_raw_canonical_values_match_source(
    sample_rsmf_uncompressed: Path,
) -> None:
    expected = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    with rsmf.RsmfFile(sample_rsmf_uncompressed) as model:
        arr = model.get_tensor("weights")
        np.testing.assert_array_equal(arr, expected)


def test_raw_canonical_write_raises(sample_rsmf_uncompressed: Path) -> None:
    with rsmf.RsmfFile(sample_rsmf_uncompressed) as model:
        arr = model.get_tensor("weights")
        with pytest.raises(ValueError, match="read-only"):
            arr[0, 0] = 42.0


def test_raw_array_outlives_wrapper_close(sample_rsmf_uncompressed: Path) -> None:
    """The Python wrapper's close() drops its reference, but the NumPy
    array we already produced holds the native RsmfFile alive through its
    `.base` chain, so the mmap stays mapped and the array stays valid."""
    model = rsmf.RsmfFile(sample_rsmf_uncompressed)
    arr = model.get_tensor("weights")
    expected = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)

    model.close()
    assert model.closed
    # Force GC so any orphan references are collected — the array must
    # still read correctly.
    gc.collect()
    np.testing.assert_array_equal(arr, expected)


# -- compressed fixture: pointer lives inside the decompressed cache ---


def test_compressed_raw_canonical_still_zero_copies(
    sample_rsmf: Path,
) -> None:
    """Even with `--compress-tensors`, the canonical F32 payload becomes
    a Vec<u8> in the reader's OnceLock on first access. OnceLock is
    single-init, so the Vec's pointer is stable for the lifetime of the
    RsmfFile. The zero-copy path must detect that and still hand out a
    read-only view."""
    with rsmf.RsmfFile(sample_rsmf) as model:
        arr = model.get_tensor("weights")
        assert arr.flags.writeable is False
        assert arr.base is not None


# -- copy path assertions ---------------------------------------------


def test_q4_0_variant_takes_copy_path(sample_rsmf: Path) -> None:
    """`encoding=block_quantized` cannot be zero-copied because the bytes
    need dequantization. The returned array must be a fresh writable copy
    so callers can process it in place.

    `owndata` is not a useful distinguisher here — `IntoPyArray`-produced
    arrays report `owndata=False` even though they're a fresh copy (NumPy
    is wrapping a Rust-allocated buffer). The writability flag is the
    clean signal: `__array_interface__` with `readonly=True` gives the
    user a read-only view; `IntoPyArray` gives a writable copy.
    """
    with rsmf.RsmfFile(sample_rsmf) as model:
        arr = model.get_tensor("weights", target="cpu_generic")
        assert arr.dtype == np.float32
        assert arr.flags.writeable is True, (
            "quantized variant must take the decode/copy path; "
            "writable=False means a raw reinterpretation slipped through."
        )


# -- contents survive a round-trip and correctness of both paths ------


def test_both_paths_produce_same_shape_and_dtype(
    sample_rsmf: Path, sample_rsmf_uncompressed: Path
) -> None:
    with rsmf.RsmfFile(sample_rsmf_uncompressed) as model_raw:
        raw = model_raw.get_tensor("weights")
    with rsmf.RsmfFile(sample_rsmf) as model_copy:
        q4 = model_copy.get_tensor("weights", target="cpu_generic")

    assert raw.dtype == q4.dtype == np.float32
    assert raw.shape == q4.shape == (16, 8)
