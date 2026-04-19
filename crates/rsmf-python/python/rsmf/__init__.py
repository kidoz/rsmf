"""Python bindings for the Rust Split Model Format (rsmf).

Quick start::

    import rsmf

    with rsmf.RsmfFile("model.rsmf") as model:
        model.verify()                                     # full BLAKE3 pass
        info = model.file_info()                           # dict of counters
        weights = model.get_tensor("embedding.weight")     # NumPy ndarray
        q4 = model.get_tensor("embedding.weight", target="cpu_generic")
        for name in model:                                 # iterate tensor names
            print(name, model.tensor_info(name)["shape"])

The wrapper accepts ``str`` or ``os.PathLike`` paths, is a context manager
that releases the mmap on exit, and exposes ``len(model)`` / ``name in model``
/ ``for name in model`` over the tensor name list.

All failures raise a subclass of :class:`RsmfError` so callers can route
verification failures differently from missing-tensor bugs.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ._rsmf import (
    RsmfError,
    RsmfIoError,
    RsmfNotFound,
    RsmfStructuralError,
    RsmfUnsupportedError,
    RsmfVerificationError,
)
from ._rsmf import RsmfFile as _NativeRsmfFile
from ._types import FileInfo, TensorInfo, VariantInfo

if TYPE_CHECKING:
    import numpy as np

__all__ = [
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
]

__version__ = "0.1.0"


class RsmfFile:
    """Ergonomic Python wrapper around the native :class:`_rsmf.RsmfFile`.

    Adds :pep:`343` context-manager support that releases the mmap on exit,
    ``os.PathLike`` input, and iteration / ``len`` / ``in`` over the tensor
    name list. After :meth:`close` (or exiting the ``with`` block) every
    method raises :class:`RsmfError`.
    """

    __slots__ = ("_inner",)

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._inner: _NativeRsmfFile | None = _NativeRsmfFile(os.fspath(path))

    # -- context manager and lifecycle ----------------------------------

    def close(self) -> None:
        """Release the native object and its mmap. Idempotent."""
        # Dropping the only reference runs the Rust `Drop` impl on the
        # underlying `memmap2::Mmap`, which unmaps the file. On Windows
        # this is what lets the caller delete/replace the underlying file
        # afterwards.
        self._inner = None

    def __enter__(self) -> RsmfFile:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        # Defensive: make sure the mmap is released even when callers
        # forget to use `with`. __slots__ may not be fully initialised if
        # __init__ raised, hence the getattr guard.
        if getattr(self, "_inner", None) is not None:
            self.close()

    @property
    def closed(self) -> bool:
        return self._inner is None

    def _require_open(self) -> _NativeRsmfFile:
        inner = self._inner
        if inner is None:
            raise RsmfError("RsmfFile is closed")
        return inner

    # -- collection-style sugar over tensor names -----------------------

    def __len__(self) -> int:
        return len(self._require_open().tensor_names())

    def __iter__(self):
        return iter(self._require_open().tensor_names())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name in self._require_open().tensor_names()

    def __repr__(self) -> str:
        if self._inner is None:
            return "<RsmfFile closed>"
        return repr(self._inner)

    # -- typed-dict façade over the native accessors --------------------

    def file_info(self) -> FileInfo:
        """Summary counters for the open file."""
        return self._require_open().file_info()  # type: ignore[return-value]

    def tensor_info(self, name: str) -> TensorInfo:
        """Lightweight descriptor for ``name`` — no variant bytes are read."""
        return self._require_open().tensor_info(name)  # type: ignore[return-value]

    def tensor_variants(self, name: str) -> list[VariantInfo]:
        """Per-variant metadata (canonical first, then packed)."""
        return self._require_open().tensor_variants(name)  # type: ignore[return-value]

    # -- pass-throughs that don't need wrapping -------------------------

    def metadata(self) -> dict[str, str]:
        return self._require_open().metadata()

    def tensor_names(self) -> list[str]:
        return self._require_open().tensor_names()

    def asset_names(self) -> list[str]:
        return self._require_open().asset_names()

    def graph_count(self) -> int:
        return self._require_open().graph_count()

    def graph_kind(self, idx: int) -> str:
        return self._require_open().graph_kind(idx)

    def get_graph(self, idx: int) -> bytes:
        return self._require_open().get_graph(idx)

    def get_asset(self, name: str) -> bytes | None:
        return self._require_open().get_asset(name)

    def verify(self) -> None:
        """Full BLAKE3 pass. Raises :class:`RsmfVerificationError` on mismatch."""
        self._require_open().verify()

    def get_tensor(
        self,
        name: str,
        *,
        target: str | None = None,
        strict: bool = False,
    ) -> np.ndarray:
        """Load ``name`` as a NumPy array.

        When ``target`` is given (e.g. ``"cpu_generic"``), the first packed
        variant with that backend tag is loaded. Quantized storage types
        (Q4_0, Q8_0, NF4, F16, BF16) are dequantized to ``float32``.

        With ``strict=True``, a missing target raises :class:`RsmfNotFound`
        instead of silently falling back to the canonical variant.
        """
        return self._require_open().get_tensor(name, target, strict)

    def get_tensor_variant(self, name: str, variant_idx: int) -> np.ndarray:
        """Load a specific variant by the global index from
        :meth:`tensor_variants`.

        Raises :class:`RsmfNotFound` when ``variant_idx`` isn't owned by
        the named tensor.
        """
        return self._require_open().get_tensor_variant(name, variant_idx)
