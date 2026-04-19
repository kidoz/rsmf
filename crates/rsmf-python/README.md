# rsmf

Python bindings for the [Rust Split Model Format](https://github.com/kidoz/rsmf)
— a binary container for ML model tensors with:

- `mmap`-backed CPU loading for low peak RSS,
- multiple packed variants per tensor (`cpu_generic`, `cpu_avx2`, `wgpu`, `cuda`, `metal`, …),
- block quantization (Q4_0, Q8_0, NF4, Q3_K, FP8) with transparent dequantization,
- BLAKE3 integrity checks over preamble, sections, variants, graphs, and assets,
- bundled ONNX / ORT graph payload and named assets in a single file.

## Install

```sh
pip install rsmf
```

Wheels are published for CPython 3.11–3.14 on Linux, macOS, and Windows.
Building from source requires a recent Rust toolchain and
[maturin](https://github.com/PyO3/maturin):

```sh
pip install maturin
maturin develop --release
```

## Usage

```python
import rsmf

with rsmf.RsmfFile("model.rsmf") as model:
    model.verify()                                 # full BLAKE3 pass

    info = model.file_info()
    print(info["format_major"], info["tensor_count"])

    # Iterate tensors; `name in model` and `len(model)` also work.
    for name in model:
        tensor_info = model.tensor_info(name)
        print(name, tensor_info["dtype"], tensor_info["shape"])

    # Canonical variant, always decoded to the logical dtype (F32 / I32 / ...).
    weights = model.get_tensor("embedding.weight")

    # Prefer a quantized cpu_generic variant; silently fall back to canonical
    # if the file doesn't have one.
    compact = model.get_tensor("embedding.weight", target="cpu_generic")

    # Raise if the requested variant is absent.
    try:
        gpu_weights = model.get_tensor("embedding.weight", target="cuda", strict=True)
    except rsmf.RsmfNotFound:
        ...

    # Opaque graph payload + named assets bundled in the same file.
    if model.graph_count() > 0:
        onnx_bytes = model.get_graph(0)
    config_bytes = model.get_asset("config.json")
```

## Exceptions

All failures subclass `rsmf.RsmfError`. Route each failure mode distinctly:

| Class | Raised when |
|---|---|
| `RsmfNotFound` | Tensor, asset, graph, or shard not present |
| `RsmfStructuralError` | Malformed preamble, overlapping sections, invalid manifest |
| `RsmfVerificationError` | BLAKE3 mismatch |
| `RsmfIoError` | Underlying I/O failure |
| `RsmfUnsupportedError` | Dtype or operation unsupported by this build |

## Types

The package ships a `py.typed` marker (PEP 561) and type stubs for the native
module. `file_info()`, `tensor_info()`, and `tensor_variants()` return
`TypedDict`s (`rsmf.FileInfo`, `rsmf.TensorInfo`, `rsmf.VariantInfo`) so
editors and `mypy` see the named fields.

## License

MIT.
