"""End-to-end test for `rsmf pack --from-torch`.

Skips cleanly when `torch` is not installed in the test interpreter.
When it is, it drives the full path:

    torch.save  →  rsmf CLI subprocess with RSMF_PYTHON_BIN=sys.executable
               →  rsmf.RsmfFile round-trip  →  numpy equality check

Forwarding `RSMF_PYTHON_BIN=sys.executable` to the CLI is deliberate —
the CLI's default is plain `python3`, which on CI hosts may point at a
system interpreter without torch installed even when the test venv
has it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import rsmf  # noqa: E402  (after importorskip so unrelated suites still collect)


@pytest.fixture(scope="module")
def sample_pt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("rsmf-torch-roundtrip")
    state = {
        "encoder.weight": torch.arange(24, dtype=torch.float32).reshape(4, 6),
        "encoder.bias": torch.zeros(4, dtype=torch.float32),
    }
    pt_path = work / "sample.pt"
    torch.save(state, pt_path)
    return pt_path


def _run_pack(rsmf_binary: str, pt_path: Path, out_path: Path, *extra: str) -> None:
    env = {**os.environ, "RSMF_PYTHON_BIN": sys.executable}
    cmd = [
        rsmf_binary,
        "pack",
        "--from-torch",
        str(pt_path),
        "--out",
        str(out_path),
        *extra,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"rsmf pack --from-torch exited with {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_from_torch_roundtrip_tensor_values(
    sample_pt: Path, rsmf_binary: str, tmp_path: Path
) -> None:
    out_path = tmp_path / "sample.rsmf"
    _run_pack(rsmf_binary, sample_pt, out_path)

    with rsmf.RsmfFile(out_path) as model:
        model.verify()
        assert set(model.tensor_names()) == {"encoder.weight", "encoder.bias"}

        weight = model.get_tensor("encoder.weight")
        bias = model.get_tensor("encoder.bias")

        assert weight.dtype == np.float32
        assert weight.shape == (4, 6)
        np.testing.assert_array_equal(
            weight, np.arange(24, dtype=np.float32).reshape(4, 6)
        )

        assert bias.dtype == np.float32
        assert bias.shape == (4,)
        np.testing.assert_array_equal(bias, np.zeros(4, dtype=np.float32))


def test_from_torch_composes_with_quantization(
    sample_pt: Path, rsmf_binary: str, tmp_path: Path
) -> None:
    out_path = tmp_path / "sample_q4.rsmf"
    _run_pack(
        rsmf_binary,
        sample_pt,
        out_path,
        "--quantize-q4_0",
        "cpu_generic",
        "--compress-tensors",
    )

    with rsmf.RsmfFile(out_path) as model:
        model.verify()
        variants = model.tensor_variants("encoder.weight")
        # canonical F32 plus one cpu_generic Q4_0.
        assert len(variants) == 2
        assert variants[0]["is_canonical"] is True
        assert variants[0]["storage_dtype"] == "f32"
        assert variants[1]["target"] == "cpu_generic"
        assert variants[1]["storage_dtype"] == "q4_0"


def test_from_torch_file_carries_safetensors_source_marker(
    sample_pt: Path, rsmf_binary: str, tmp_path: Path
) -> None:
    out_path = tmp_path / "sample.rsmf"
    _run_pack(rsmf_binary, sample_pt, out_path)

    with rsmf.RsmfFile(out_path) as model:
        meta = model.metadata()
        # The torch path currently goes through the safetensors pipeline
        # via a temp file, so the source metadata reflects that. If ADR 0012
        # adds a richer marker later this assertion will tighten.
        assert meta.get("source") == "safetensors"
