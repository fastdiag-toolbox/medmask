from __future__ import annotations

"""Single-mask container (.msk).

`MaskFile` stores exactly one :class:`medmask.core.segmask.SegmentationMask` on disk.
Persistence helpers `save_mask()` / `load_mask()` offer functional usage.

File format design for cross-language compatibility:
- Mask data: stored as-is in (z,y,x) format 
- Space description: stored in (x,y,z) format for C++ compatibility
- Python users: see data(z,y,x) + C order + space(z,y,x) → aligned
- C++ users: see data(x,y,z) + fortran order + space(x,y,z) → aligned in their axis convention
- No axis_reversed complications for users, conversion handled transparently
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
import json
import os
import struct

from spacetransformer import Space

from medmask.compression import get_codec
from medmask.core.mapping import LabelMapping

if TYPE_CHECKING:  # pragma: no cover – type hints only
    from medmask.core.segmask import SegmentationMask  # noqa: F401

__all__ = [
    "MaskFile",
    "save_mask",
    "load_mask",
]


class MaskFile:
    """Light-weight file format for a single segmentation mask (.msk)."""

    MAGIC_NUMBER = b"MSK1" 
    VERSION = (1, 0)  # major, minor

    # <4s 2B B 6Q 7x> = 64 B total 
    HEADER_STRUCT = "<4s2B B 6Q 7x"

    def __init__(self, path: str, mode: str = "r", *, codec: str | None = None):
        self.path = path
        self.mode = mode
        self.codec = get_codec(codec)  # default zstd
        self._header_cache: Optional[Dict[str, int]] = None

        if os.path.exists(path):
            if mode == "w":
                os.remove(path)  # overwrite
            elif mode not in {"r", "a"}:
                raise ValueError(f"unsupported mode: {mode}")
        else:
            if mode == "r":
                raise FileNotFoundError(path)
            # for "w" we'll create on write

    # ------------------------------------------------------------------
    # Header helpers
    # ------------------------------------------------------------------
    def _pack_header(self, **kw: int) -> bytes:
        major, minor = self.VERSION
        return struct.pack(
            self.HEADER_STRUCT,
            self.MAGIC_NUMBER,
            major,
            minor,
            self.codec.id,
            kw["space_offset"],
            kw["space_length"],
            kw["mapping_offset"],
            kw["mapping_length"],
            kw["data_offset"],
            kw["data_length"],
        )

    def _write_header(self, fp: Any, **kw: int) -> None:
        fp.seek(0)
        fp.write(self._pack_header(**kw))

    def _read_header(self) -> Dict[str, int]:
        with open(self.path, "rb") as fp:
            raw = fp.read(struct.calcsize(self.HEADER_STRUCT))
            
        (
            magic,
            ver_major,
            ver_minor,
            codec_id,
            space_offset,
            space_length,
            mapping_offset,
            mapping_length,
            data_offset,
            data_length,
        ) = struct.unpack(self.HEADER_STRUCT, raw)
        
        if magic != self.MAGIC_NUMBER:
            raise ValueError(f"Invalid magic number: {magic}. Expected {self.MAGIC_NUMBER}")
        if (ver_major, ver_minor) != self.VERSION:
            raise ValueError(f"Unsupported version {(ver_major, ver_minor)}. Expected {self.VERSION}")
        if codec_id != self.codec.id:
            raise ValueError(
                f"Codec mismatch: file uses id {codec_id}, but current codec id is {self.codec.id}"
            )
        
        return {
            "space_offset": space_offset,
            "space_length": space_length,
            "mapping_offset": mapping_offset,
            "mapping_length": mapping_length,
            "data_offset": data_offset,
            "data_length": data_length,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write(self, segmask: "SegmentationMask") -> None:
        """Write segmentation mask to file.
        
        Stores mask data as-is, but converts space description to (x,y,z) order
        for C++ compatibility.
        """
        if self.mode == "r":
            raise IOError("file opened read-only")

        # Convert space from internal (z,y,x) to file (x,y,z) format
        space_for_file = segmask.space.reverse_axis_order()
        space_b = space_for_file.to_json().encode("utf-8")
        
        # Keep mapping and data as-is
        mapping_b = json.dumps(segmask.mapping._name_to_label).encode("utf-8")
        data_b = self.codec.encode(segmask.data)

        hdr_size = struct.calcsize(self.HEADER_STRUCT)
        space_offset = hdr_size
        space_length = len(space_b)
        mapping_offset = space_offset + space_length
        mapping_length = len(mapping_b)
        data_offset = mapping_offset + mapping_length
        data_length = len(data_b)

        with open(self.path, "wb") as fp:
            # header first
            self._write_header(
                fp,
                space_offset=space_offset,
                space_length=space_length,
                mapping_offset=mapping_offset,
                mapping_length=mapping_length,
                data_offset=data_offset,
                data_length=data_length,
            )
            # sections
            fp.seek(space_offset)
            fp.write(space_b)
            fp.seek(mapping_offset)
            fp.write(mapping_b)
            fp.seek(data_offset)
            fp.write(data_b)

        self._header_cache = {
            "space_offset": space_offset,
            "space_length": space_length,
            "mapping_offset": mapping_offset,
            "mapping_length": mapping_length,
            "data_offset": data_offset,
            "data_length": data_length,
        }

    # ------------------------------------------------------------------
    def read(self) -> "SegmentationMask":
        """Read segmentation mask from file.
        
        Reads mask data as-is, but converts space description from (x,y,z) back 
        to internal (z,y,x) format.
        """
        from medmask.core.segmask import SegmentationMask  # inline import to avoid cycle

        hdr = self.header
        with open(self.path, "rb") as fp:
            fp.seek(hdr["space_offset"])
            space_json = fp.read(hdr["space_length"]).decode("utf-8")
            # Convert space from file (x,y,z) to internal (z,y,x) format
            space_from_file = Space.from_json(space_json)
            space = space_from_file.reverse_axis_order()

            fp.seek(hdr["mapping_offset"])
            mapping_json = fp.read(hdr["mapping_length"]).decode("utf-8")
            mapping = LabelMapping(json.loads(mapping_json))

            fp.seek(hdr["data_offset"])
            data_b = fp.read(hdr["data_length"])
            arr = self.codec.decode(data_b)

        return SegmentationMask(arr, mapping, space=space)

    # ------------------------------------------------------------------
    @property
    def header(self) -> Dict[str, int]:
        if self._header_cache is None:
            self._header_cache = self._read_header()
        return self._header_cache

    # context manager ---------------------------------------------------
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False  # propagate exceptions


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def save_mask(segmask: "SegmentationMask", path: str, *, codec: str | None = None) -> None:
    """Save *segmask* to *path* (.msk)."""
    MaskFile(path, mode="w", codec=codec).write(segmask)


def load_mask(path: str, *, codec: str | None = None):  # -> SegmentationMask
    """Load and return a :class:`SegmentationMask` from *path*."""
    return MaskFile(path, mode="r", codec=codec).read() 