"""Edge cases and extended API tests for rsmf-python."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

import rsmf


def test_metadata_returns_dict(open_file: rsmf.RsmfFile) -> None:
    md = open_file.metadata()
    assert isinstance(md, dict)


def test_repr_open_and_closed(open_file: rsmf.RsmfFile) -> None:
    r = repr(open_file)
    assert "<RsmfFile tensors=" in r
    assert "variants=" in r
    
    open_file.close()
    r_closed = repr(open_file)
    assert r_closed == "<RsmfFile closed>"


def test_contains_non_string(open_file: rsmf.RsmfFile) -> None:
    assert 123 not in open_file
    assert None not in open_file


def test_out_of_bounds_graph_kind(open_file: rsmf.RsmfFile) -> None:
    count = open_file.graph_count()
    with pytest.raises(IndexError):
        open_file.graph_kind(count + 5)


def test_out_of_bounds_get_graph(open_file: rsmf.RsmfFile) -> None:
    count = open_file.graph_count()
    with pytest.raises(IndexError):
        open_file.get_graph(count + 5)


def test_asset_names_contains_tokenizer(open_file: rsmf.RsmfFile) -> None:
    names = open_file.asset_names()
    assert isinstance(names, list)
    assert "tokenizer.json" in names


def test_tensor_names_returns_list(open_file: rsmf.RsmfFile) -> None:
    names = open_file.tensor_names()
    assert isinstance(names, list)
    assert "weights" in names


def test_get_tensor_variant_valid_idx(open_file: rsmf.RsmfFile) -> None:
    variants = open_file.tensor_variants("weights")
    assert len(variants) >= 1
    
    canonical_idx = variants[0]["variant_idx"]
    arr = open_file.get_tensor_variant("weights", canonical_idx)
    assert arr.dtype == np.float32
    assert arr.shape == (16, 8)
    
    if len(variants) > 1:
        packed_idx = variants[1]["variant_idx"]
        arr2 = open_file.get_tensor_variant("weights", packed_idx)
        assert arr2.dtype == np.float32
        assert arr2.shape == (16, 8)


def test_all_methods_raise_when_closed(sample_rsmf: Path) -> None:
    model = rsmf.RsmfFile(sample_rsmf)
    model.close()
    
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.file_info()
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.tensor_info("weights")
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.tensor_variants("weights")
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.metadata()
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.tensor_names()
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.asset_names()
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.graph_count()
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.graph_kind(0)
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.get_graph(0)
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.get_asset("tokenizer.json")
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.verify()
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.get_tensor("weights")
    with pytest.raises(rsmf.RsmfError, match="closed"):
        model.get_tensor_variant("weights", 0)

    # Magic methods
    with pytest.raises(rsmf.RsmfError, match="closed"):
        len(model)
    with pytest.raises(rsmf.RsmfError, match="closed"):
        list(model) # iter
    with pytest.raises(rsmf.RsmfError, match="closed"):
        "weights" in model
