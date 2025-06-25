from __future__ import annotations

"""Compression sub-package.

This package defines the codec registry plus helper to obtain a codec by name.
Currently only *zstd* is implemented but the design allows future extensions
(e.g. lz4, bz2, gpu-based codecs …).
"""

from typing import Dict, Protocol, runtime_checkable

__all__ = [
    "MaskCodec",
    "get_codec",
]


@runtime_checkable
class MaskCodec(Protocol):
    """Minimal interface all codecs must implement."""

    id: int  # unique numeric ID written to header
    name: str  # human readable name (e.g. "zstd")

    def encode(self, arr, /):  # noqa: ANN001 – keep generic for ndarray-like
        """Return *bytes* representing *arr* compressed with this codec."""

    def decode(self, blob: bytes):  # noqa: D401 – short description OK
        """Return ndarray reconstructed from *blob*."""


# ---------------------------------------------------------------------------
# Registry helpers ----------------------------------------------------------
# ---------------------------------------------------------------------------
_codec_registry: Dict[str, MaskCodec] = {}


def _register(codec: MaskCodec) -> None:  # pragma: no cover – called on import
    _codec_registry[codec.name] = codec


# Import concrete implementations (auto-register)
from .zstd_codec import ZstdMaskCodec as _ZstdMaskCodec  # noqa: E402  (import after Protocol)

_register(_ZstdMaskCodec())


def get_codec(name: str | None = None) -> MaskCodec:
    """Return codec by *name* (case-insensitive).  Defaults to *zstd*."""

    if name is None:
        name = "zstd"
    try:
        return _codec_registry[name.lower()]
    except KeyError as err:
        raise ValueError(f"Unknown codec '{name}'. Available: {list(_codec_registry)}") from err 