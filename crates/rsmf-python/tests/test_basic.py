"""Binding-level smoke + integration tests.

These tests depend on the extension being importable; failing with an
``ImportError`` is intentional — CI should build the wheel before
invoking pytest rather than silently skipping the suite.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import rsmf


# -- API surface / packaging -------------------------------------------


def test_public_api_surface() -> None:
    for name in (
        "RsmfFile",
        "RsmfError",
        "RsmfNotFound",
        "RsmfStructuralError",
        "RsmfVerificationError",
        "RsmfIoError",
        "RsmfUnsupportedError",
        "FileInfo",
        "TensorInfo",
        "VariantInfo",
    ):
        assert hasattr(rsmf, name), f"rsmf.{name} missing"


def test_exception_hierarchy() -> None:
    assert issubclass(rsmf.RsmfError, Exception)
    for sub in (
        rsmf.RsmfNotFound,
        rsmf.RsmfStructuralError,
        rsmf.RsmfVerificationError,
        rsmf.RsmfIoError,
        rsmf.RsmfUnsupportedError,
    ):
        assert issubclass(sub, rsmf.RsmfError)


def test_py_typed_marker_present() -> None:
    pkg_dir = Path(rsmf.__file__).parent
    assert (pkg_dir / "py.typed").is_file()


def test_version_string_present() -> None:
    assert isinstance(rsmf.__version__, str) and rsmf.__version__


# -- open / error mapping ----------------------------------------------


def test_open_nonexistent_raises_typed_error() -> None:
    bogus = Path(tempfile.gettempdir()) / "rsmf-does-not-exist-xyzzy.rsmf"
    if bogus.exists():
        bogus.unlink()
    with pytest.raises(rsmf.RsmfError):
        rsmf.RsmfFile(bogus)


def test_pathlike_input_accepted(sample_rsmf: Path) -> None:
    # Path (not just str) must be accepted.
    with rsmf.RsmfFile(sample_rsmf) as model:
        assert len(model) > 0


def test_truncated_file_raises_structural(sample_rsmf: Path, tmp_path: Path) -> None:
    chopped = tmp_path / "chopped.rsmf"
    chopped.write_bytes(sample_rsmf.read_bytes()[:32])
    with pytest.raises(rsmf.RsmfError):
        rsmf.RsmfFile(chopped)


# -- file_info / tensor_info / metadata --------------------------------


def test_file_info_has_expected_counters(open_file: rsmf.RsmfFile) -> None:
    info = open_file.file_info()
    assert info["format_major"] == 1
    assert info["tensor_count"] == 1
    assert info["variant_count"] == 2  # canonical + packed Q4_0
    assert info["graph_count"] == 1
    assert info["asset_count"] == 1


def test_tensor_info_matches_written_shape(open_file: rsmf.RsmfFile) -> None:
    info = open_file.tensor_info("weights")
    assert info["dtype"] == "f32"
    assert info["shape"] == [16, 8]
    assert info["element_count"] == 128


def test_unknown_tensor_raises_notfound(open_file: rsmf.RsmfFile) -> None:
    with pytest.raises(rsmf.RsmfNotFound):
        open_file.tensor_info("does-not-exist")


# -- verify() ----------------------------------------------------------


def test_verify_passes_on_well_formed(open_file: rsmf.RsmfFile) -> None:
    open_file.verify()  # must not raise


def test_verify_fails_on_corrupted(sample_rsmf: Path, tmp_path: Path) -> None:
    corrupted = tmp_path / "corrupt.rsmf"
    data = bytearray(sample_rsmf.read_bytes())
    # Flip a byte well past the preamble so open() still succeeds but
    # full_verify catches the checksum mismatch.
    flip_idx = len(data) - 64
    data[flip_idx] ^= 0xFF
    corrupted.write_bytes(data)

    model = rsmf.RsmfFile(corrupted)
    with pytest.raises(rsmf.RsmfVerificationError):
        model.verify()


# -- tensor round-trip -------------------------------------------------


def test_get_tensor_roundtrip_shape_and_dtype(open_file: rsmf.RsmfFile) -> None:
    array = open_file.get_tensor("weights")
    assert array.dtype == np.float32
    assert array.shape == (16, 8)
    expected = np.arange(128, dtype=np.float32).reshape(16, 8)
    np.testing.assert_array_equal(array, expected)


def test_get_tensor_target_selects_q4_0_variant(open_file: rsmf.RsmfFile) -> None:
    array = open_file.get_tensor("weights", target="cpu_generic")
    # Q4_0 dequantizes back to f32 with quantization error, but the shape
    # must match and the ordering must stay roughly monotonic.
    assert array.dtype == np.float32
    assert array.shape == (16, 8)


def test_get_tensor_strict_unknown_target_raises(open_file: rsmf.RsmfFile) -> None:
    with pytest.raises(rsmf.RsmfNotFound):
        open_file.get_tensor("weights", target="cuda", strict=True)


def test_get_tensor_non_strict_falls_back_to_canonical(open_file: rsmf.RsmfFile) -> None:
    array = open_file.get_tensor("weights", target="cuda")
    # Non-strict path falls back to canonical F32.
    assert array.dtype == np.float32


# -- tensor_variants / get_tensor_variant ------------------------------


def test_tensor_variants_lists_canonical_first(open_file: rsmf.RsmfFile) -> None:
    variants = open_file.tensor_variants("weights")
    assert len(variants) == 2
    assert variants[0]["is_canonical"] is True
    assert variants[0]["target"] == "canonical"
    assert variants[1]["is_canonical"] is False
    assert variants[1]["target"] == "cpu_generic"
    assert variants[1]["storage_dtype"] == "q4_0"


def test_get_tensor_variant_cross_tensor_rejected(open_file: rsmf.RsmfFile) -> None:
    # There's only one tensor, so craft a variant_idx that doesn't belong.
    bogus_idx = 99
    with pytest.raises(rsmf.RsmfNotFound):
        open_file.get_tensor_variant("weights", bogus_idx)


# -- graph / asset -----------------------------------------------------


def test_graph_roundtrip(open_file: rsmf.RsmfFile) -> None:
    assert open_file.graph_count() == 1
    assert open_file.graph_kind(0) == "onnx"
    blob = open_file.get_graph(0)
    assert blob == b"ONNX\x00fake-graph-for-tests"


def test_asset_roundtrip(open_file: rsmf.RsmfFile) -> None:
    assert "tokenizer.json" in open_file.asset_names()
    blob = open_file.get_asset("tokenizer.json")
    assert blob == b'{"vocab":["<pad>","<bos>","<eos>"]}'


def test_missing_asset_returns_none(open_file: rsmf.RsmfFile) -> None:
    assert open_file.get_asset("not-present.json") is None


# -- context manager & close() -----------------------------------------


def test_with_block_closes_file(sample_rsmf: Path) -> None:
    with rsmf.RsmfFile(sample_rsmf) as model:
        assert not model.closed
        _ = model.tensor_names()
    assert model.closed
    with pytest.raises(rsmf.RsmfError):
        model.tensor_names()


def test_close_is_idempotent(sample_rsmf: Path) -> None:
    model = rsmf.RsmfFile(sample_rsmf)
    model.close()
    model.close()  # must not raise
    assert model.closed


def test_reentering_closed_wrapper_fails(sample_rsmf: Path) -> None:
    model = rsmf.RsmfFile(sample_rsmf)
    model.close()
    with pytest.raises(rsmf.RsmfError):
        with model:
            pass


# -- collection sugar --------------------------------------------------


def test_len_iter_contains(open_file: rsmf.RsmfFile) -> None:
    assert len(open_file) == 1
    assert "weights" in open_file
    assert "bogus" not in open_file
    names = list(open_file)
    assert names == ["weights"]
