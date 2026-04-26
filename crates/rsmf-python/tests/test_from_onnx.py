"""End-to-end test for `rsmf pack --from-onnx`.

Skips cleanly when `onnx` is not installed in the test interpreter.
When it is, it drives the full path:

    onnx.save  →  rsmf CLI subprocess with RSMF_PYTHON_BIN=sys.executable
               →  rsmf.RsmfFile round-trip  →  numpy equality check
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto, numpy_helper
import rsmf  # noqa: E402


@pytest.fixture(scope="module")
def sample_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("rsmf-onnx-roundtrip")
    
    # Create two initializers (tensors)
    w_arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    b_arr = np.zeros(4, dtype=np.float32)
    w_init = numpy_helper.from_array(w_arr, name="encoder.weight")
    b_init = numpy_helper.from_array(b_arr, name="encoder.bias")

    # Minimum valid graph needs a node and I/O matching the initializers (loosely)
    node = helper.make_node('Identity', ['encoder.weight'], ['output1'])
    node2 = helper.make_node('Identity', ['encoder.bias'], ['output2'])
    
    in_w = helper.make_tensor_value_info('encoder.weight', TensorProto.FLOAT, [4, 6])
    in_b = helper.make_tensor_value_info('encoder.bias', TensorProto.FLOAT, [4])
    out1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [4, 6])
    out2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [4])
    
    graph = helper.make_graph(
        [node, node2],
        'test-model',
        [in_w, in_b],
        [out1, out2],
        [w_init, b_init]
    )
    model = helper.make_model(graph, producer_name='rsmf-test')
    
    onnx_path = work / "sample.onnx"
    onnx.save(model, str(onnx_path))
    return onnx_path


def _run_pack(rsmf_binary: str, onnx_path: Path, out_path: Path, *extra: str) -> None:
    env = {**os.environ, "RSMF_PYTHON_BIN": sys.executable}
    cmd = [
        rsmf_binary,
        "pack",
        "--from-onnx",
        str(onnx_path),
        "--out",
        str(out_path),
        *extra,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"rsmf pack --from-onnx exited with {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_from_onnx_roundtrip_tensor_values(
    sample_onnx: Path, rsmf_binary: str, tmp_path: Path
) -> None:
    out_path = tmp_path / "sample.rsmf"
    _run_pack(rsmf_binary, sample_onnx, out_path)

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


def test_from_onnx_composes_with_quantization(
    sample_onnx: Path, rsmf_binary: str, tmp_path: Path
) -> None:
    out_path = tmp_path / "sample_q4.rsmf"
    _run_pack(
        rsmf_binary,
        sample_onnx,
        out_path,
        "--quantize-q4_0",
        "cpu_generic",
        "--compress-tensors",
    )

    with rsmf.RsmfFile(out_path) as model:
        model.verify()
        variants = model.tensor_variants("encoder.weight")
        assert len(variants) == 2
        assert variants[0]["is_canonical"] is True
        assert variants[0]["storage_dtype"] == "f32"
        assert variants[1]["target"] == "cpu_generic"
        assert variants[1]["storage_dtype"] == "q4_0"
