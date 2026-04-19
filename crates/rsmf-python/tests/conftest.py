"""Pytest fixtures for rsmf-python.

The ``sample_rsmf`` fixture synthesises a small `.rsmf` on disk via the
``rsmf`` CLI so every test runs against a real file with all of the
format's sections in play:

    - one F32 tensor (``weights``) with a ``cpu_generic`` Q4_0 packed variant,
    - an ONNX-kinded graph payload,
    - a named asset.

The CLI binary is resolved in this order:

    1. ``RSMF_BIN`` env var if set,
    2. the installed binary on ``PATH``,
    3. the debug build at ``<workspace>/target/debug/rsmf``,
    4. building via ``cargo build -p rsmf-cli`` as a fallback.

If none of those produce a working CLI the fixture emits a loud failure
rather than skipping, so CI catches regressions in the binding layer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _workspace_root() -> Path:
    # tests/conftest.py -> tests/ -> rsmf-python -> crates -> <workspace>
    return Path(__file__).resolve().parents[3]


def _resolve_rsmf_binary() -> str:
    explicit = os.getenv("RSMF_BIN")
    if explicit and Path(explicit).exists():
        return explicit
    on_path = shutil.which("rsmf")
    if on_path:
        return on_path
    ws = _workspace_root()
    for candidate in (ws / "target" / "release" / "rsmf", ws / "target" / "debug" / "rsmf"):
        if candidate.exists():
            return str(candidate)
    # Last resort: build it. Slow on first run, cached afterwards.
    subprocess.run(
        ["cargo", "build", "-p", "rsmf-cli"],
        cwd=ws,
        check=True,
    )
    built = ws / "target" / "debug" / "rsmf"
    if not built.exists():
        raise RuntimeError(
            "rsmf CLI could not be located or built; "
            "set RSMF_BIN to an installed binary."
        )
    return str(built)


def _make_graph_bytes() -> bytes:
    # Not a valid ONNX graph; rsmf stores graph payloads opaquely so any
    # non-empty byte string round-trips. Using `ONNX\0` keeps the intent
    # obvious when eyeballing hexdumps.
    return b"ONNX\x00fake-graph-for-tests"


@pytest.fixture(scope="session")
def rsmf_binary() -> str:
    return _resolve_rsmf_binary()


@pytest.fixture(scope="session")
def sample_rsmf(tmp_path_factory: pytest.TempPathFactory, rsmf_binary: str) -> Path:
    """Synthesise a small test `.rsmf` with tensors, a graph, and an asset."""
    work = tmp_path_factory.mktemp("rsmf-fixture")

    # 16x8 F32 matrix -> 512 bytes payload; big enough for Q4_0 (block=32)
    # to produce a valid packed variant.
    array = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    npy_path = work / "weights.npy"
    np.save(npy_path, array)

    graph_path = work / "graph.onnx"
    graph_path.write_bytes(_make_graph_bytes())

    asset_path = work / "tokenizer.json"
    asset_path.write_bytes(b'{"vocab":["<pad>","<bos>","<eos>"]}')

    out_path = work / "sample.rsmf"

    subprocess.run(
        [
            rsmf_binary,
            "pack",
            "--from-npy",
            str(npy_path),
            "--quantize-q4_0",
            "cpu_generic",
            "--compress-tensors",
            "--compress-graph",
            "--compress-assets",
            "--graph",
            str(graph_path),
            "--asset",
            str(asset_path),
            "--out",
            str(out_path),
        ],
        check=True,
    )
    return out_path


@pytest.fixture
def open_file(sample_rsmf: Path):
    """Open the session fixture fresh per-test so lifecycle tests don't
    affect other cases."""
    import rsmf

    return rsmf.RsmfFile(sample_rsmf)
