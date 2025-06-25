from __future__ import annotations

"""Zstandard-based implementation of :pyclass:`medmask.compression.MaskCodec`."""

from typing import Any

import numpy as np
import zstandard as zstd

from . import MaskCodec

__all__ = [
    "ZstdMaskCodec",
    "encode_mask_bytes",
    "decode_mask_bytes",
]


# ---------------------------------------------------------------------------
# Helper functions -----------------------------------------------------------
# ---------------------------------------------------------------------------

def encode_mask_bytes(arr: np.ndarray, *, level: int | None = None) -> bytes:
    """Return compressed *bytes* representation of *arr* using zstd.

    The on-wire layout is compatible with the original implementation:

    ``flag (1 byte) + dtype_len (1 byte) + dtype_str + shape_len (1 byte) + shape + compressed``

    *flag* = ``1`` → boolean array (``np.packbits`` trick)  
    *flag* = ``0`` → other dtypes
    """

    compressor = zstd.ZstdCompressor(level=level) if level is not None else zstd.ZstdCompressor()
    is_bool = arr.dtype == bool

    # ---------------- metadata -------------------------------------------
    shape = arr.shape
    shape_bytes = np.array(shape, dtype=np.int64).tobytes()

    dtype_str = str(arr.dtype)
    dtype_bytes = dtype_str.encode("utf-8")
    dtype_len = np.array([len(dtype_bytes)], dtype=np.uint8).tobytes()

    # ---------------- payload --------------------------------------------
    if is_bool:
        payload = compressor.compress(np.packbits(arr).tobytes())
        flag = b"\x01"
    else:
        payload = compressor.compress(arr.tobytes())
        flag = b"\x00"

    shape_len = np.array([len(shape)], dtype=np.uint8).tobytes()

    return b"".join([flag, dtype_len, dtype_bytes, shape_len, shape_bytes, payload])


def decode_mask_bytes(blob: bytes) -> np.ndarray:  # noqa: D401 – simple function
    """Inverse of :func:`encode_mask_bytes`."""

    decompressor = zstd.ZstdDecompressor()

    pos = 0
    flag = blob[pos]
    pos += 1

    dtype_len = blob[pos]
    pos += 1
    dtype_str = blob[pos : pos + dtype_len].decode("utf-8")
    dtype = np.dtype(dtype_str)
    pos += dtype_len

    shape_len = blob[pos]
    pos += 1
    shape = tuple(np.frombuffer(blob[pos : pos + shape_len * 8], dtype=np.int64))
    pos += shape_len * 8

    compressed = blob[pos:]

    decompressed = decompressor.decompress(compressed)

    if flag == 1:
        packbits = np.frombuffer(decompressed, dtype=np.uint8)
        return np.unpackbits(packbits)[: np.prod(shape)].reshape(shape).astype(bool)

    return np.frombuffer(decompressed, dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Codec class ----------------------------------------------------------------
# ---------------------------------------------------------------------------


class ZstdMaskCodec:
    """Zstandard codec wrapper implementing :pyclass:`MaskCodec`."""

    id: int = 0  # first codec ID
    name: str = "zstd"

    # Using a singleton pattern is fine because codec has no mutable state.

    def encode(self, arr: np.ndarray, /, *, level: int | None = None) -> bytes:  # noqa: D401
        """Encode *arr* -> bytes. Optional *level* overrides default compression level."""

        return encode_mask_bytes(arr, level=level)

    def decode(self, blob: bytes) -> np.ndarray:  # noqa: D401 – simple wrapper
        """Decode bytes -> ndarray."""

        return decode_mask_bytes(blob)

    # Make the instance callable for convenience ------------------------------------------------

    def __call__(self, arr: np.ndarray, /, *, level: int | None = None) -> bytes:  # pragma: no cover
        return self.encode(arr, level=level)

    # ---------------------------------------------------------------------
    # Equality / representation -------------------------------------------
    # ---------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} id={self.id} name='{self.name}'>" 