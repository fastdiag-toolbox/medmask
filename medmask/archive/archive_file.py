from __future__ import annotations

"""Binary *MaskArchive* (*.maska*) container.

Design goals compared with the legacy *MaskFile* (DCBM):
 • Magic number changed to ``b"MSKA"``
 • Compression codec is pluggable via :pymod:`medmask.compression`
 • API kept almost identical for smooth migration
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
import os
import struct

import numpy as np
from spacetransformer import Space

from medmask.compression import get_codec
from medmask.core.mapping import LabelMapping

if TYPE_CHECKING:  # pragma: no cover – for type checkers only
    from medmask.core.segmask import SegmentationMask  # noqa: F401
    Mask = SegmentationMask  # noqa: F401 – legacy alias

__all__ = ["MaskArchive"]


class MaskArchive:
    """Light-weight container storing multiple segmentation masks.

    The on-disk layout inherits from *DCBM* but is simplified:

    ``
    Header (100 B)
        magic     "MSKA" (4)
        version   0x0100 (2 bytes)  • major/minor
        codec_id  (B)
        reserved  (1 B)
        space_offset, space_length
        index_offset, index_length
        data_offset, data_length
        mask_count
        reserved (24 B)
    ``

    Fixed-sized header (100 B) eases random access and future expansion.
    """

    MAGIC_NUMBER = b"MSKA"
    VERSION = (1, 0)  # major, minor

    HEADER_STRUCT = "<4s2B B 1x 7Q24x"  # =100 B: magic + ver(2) + codec + pad + 7Q + pad

    MAX_INDEX_LENGTH = 4000  # bytes reserved for JSON index initially

    # ---------------------------------------------------------------------
    def __init__(self, path: str, mode: str = "r", *, space: Optional[Space] = None, codec: str | None = None):
        self.path = path
        self.mode = mode
        self.space = space
        self.codec = get_codec(codec)  # default zstd
        self._header_cache: Optional[Dict[str, int]] = None

        if os.path.exists(path) and mode in {"r", "a"}:
            hdr = self._read_header()
            if hdr.get("space_offset", 0):
                with open(path, "rb") as f:
                    f.seek(hdr["space_offset"])
                    space_json = f.read(hdr["space_length"]).decode("utf-8")
                    self.space = Space.from_json(space_json)
        elif mode == "w":
            if os.path.exists(path):
                os.remove(path)
        else:
            if mode not in {"r", "w", "a"}:
                raise ValueError(f"unsupported mode: {mode}")
            if mode == "r" and not os.path.exists(path):
                raise FileNotFoundError(path)

    # ------------------------------------------------------------------
    # Header helpers ----------------------------------------------------
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
            kw["index_offset"],
            kw["index_length"],
            kw["data_offset"],
            kw["data_length"],
            kw["mask_count"],
        )

    def _write_header(self, fp: Any, **kw: int) -> None:
        fp.seek(0)
        fp.write(self._pack_header(**kw))

    def _read_header(self) -> Dict[str, int]:
        if not os.path.exists(self.path):
            # return zeros when file absent – simplifies first-write logic
            return {k: 0 for k in ("space_offset", "space_length", "index_offset", "index_length", "data_offset", "data_length", "mask_count")}

        with open(self.path, "rb") as fp:
            hdr_raw = fp.read(struct.calcsize(self.HEADER_STRUCT))
        (
            magic,
            ver_major,
            ver_minor,
            codec_id,
            space_offset,
            space_length,
            index_offset,
            index_length,
            data_offset,
            data_length,
            mask_count,
        ) = struct.unpack(self.HEADER_STRUCT, hdr_raw)

        if magic != self.MAGIC_NUMBER:
            raise ValueError("invalid magic")
        if (ver_major, ver_minor) != self.VERSION:
            raise ValueError(f"unsupported version {(ver_major, ver_minor)}")
        if codec_id != self.codec.id:
            raise ValueError(f"codec mismatch: file uses id {codec_id}, but current codec id is {self.codec.id}")

        return {
            "space_offset": space_offset,
            "space_length": space_length,
            "index_offset": index_offset,
            "index_length": index_length,
            "data_offset": data_offset,
            "data_length": data_length,
            "mask_count": mask_count,
        }

    # ------------------------------------------------------------------
    # Index helpers -----------------------------------------------------
    # ------------------------------------------------------------------
    def _read_index(self) -> List[Dict[str, Any]]:
        hdr = self.header
        if hdr["index_length"] == 0:
            return []
        with open(self.path, "rb") as fp:
            fp.seek(hdr["index_offset"])
            idx_json = fp.read(hdr["index_length"]).rstrip(b"\0").decode("utf-8")
        return json.loads(idx_json) if idx_json else []

    # ------------------------------------------------------------------
    @property
    def header(self) -> Dict[str, int]:
        if self._header_cache is None:
            self._header_cache = self._read_header()
        return self._header_cache

    # ------------------------------------------------------------------
    # Data preparation --------------------------------------------------
    # ------------------------------------------------------------------
    def _prepare_data(self, segmask: "SegmentationMask") -> bytes:
        if self.space is None:
            if segmask.space is None:
                raise ValueError("space undefined for both archive and segmask")
            self.space = segmask.space
        else:
            assert segmask.space == self.space, "space mismatch between segmask and archive"
        return self.codec.encode(segmask.data)

    # ------------------------------------------------------------------
    def _ensure_index_capacity(self, fp, hdr: Dict[str, int], req_len: int) -> Dict[str, int]:
        size = hdr["index_length"] or 512
        while size < req_len:
            size *= 2
        if size == hdr["index_length"]:
            return hdr
        # relocate data block
        fp.seek(hdr["data_offset"])
        blob = fp.read(hdr["data_length"])
        new_data_offset = hdr["index_offset"] + size
        fp.seek(new_data_offset)
        fp.write(blob)

        hdr = hdr.copy()
        hdr["index_length"] = size
        hdr["data_offset"] = new_data_offset
        return hdr

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    # ------------------------------------------------------------------
    def add_segmask(self, segmask: "SegmentationMask", name: str) -> None:
        data_blob = self._prepare_data(segmask)
        idx_list = self._read_index() if os.path.exists(self.path) else []
        if any(e["name"] == name for e in idx_list):
            raise ValueError(f"SegmentationMask {name} already exists")

        # compute or read header
        if not idx_list:
            space_offset = struct.calcsize(self.HEADER_STRUCT)
            space_json = self.space.to_json()
            space_length = len(space_json.encode("utf-8"))
            index_offset = space_offset + space_length
            index_length = max(self.MAX_INDEX_LENGTH, 512)
            data_offset = index_offset + index_length
            hdr = {
                "space_offset": space_offset,
                "space_length": space_length,
                "index_offset": index_offset,
                "index_length": index_length,
                "data_offset": data_offset,
                "data_length": 0,
                "mask_count": 0,
            }
        else:
            hdr = self.header.copy()

        # write to disk
        with open(self.path, "r+b" if os.path.exists(self.path) else "wb") as fp:
            # write space JSON if first segmask
            if hdr["mask_count"] == 0:
                fp.seek(hdr["space_offset"])
                fp.write(self.space.to_json().encode("utf-8"))

            # ensure index capacity first (with existing entries only)
            json_bytes = json.dumps(idx_list).encode("utf-8")
            old_data_offset = hdr["data_offset"]
            hdr = self._ensure_index_capacity(fp, hdr, len(json_bytes) + 200)  # extra space for new entry
            
            # if data was relocated, update all existing entry offsets
            if hdr["data_offset"] != old_data_offset:
                offset_delta = hdr["data_offset"] - old_data_offset
                for entry in idx_list:
                    entry["offset"] += offset_delta
            
            # place new blob at end (after potential data relocation)
            offset = hdr["data_offset"] + hdr["data_length"]
            length = len(data_blob)
            idx_list.append(
                {
                    "name": name,
                    "offset": offset,
                    "length": length,
                    "mapping": segmask.mapping._name_to_label,
                }
            )

            # re-serialize with the new entry
            json_bytes = json.dumps(idx_list).encode("utf-8")
            
            # write index (zero-padded)
            fp.seek(hdr["index_offset"])
            fp.write(json_bytes)
            pad = hdr["index_length"] - len(json_bytes)
            if pad:
                fp.write(b"\0" * pad)

            # write data blob
            fp.seek(offset)
            fp.write(data_blob)

            # update header bookkeeping
            hdr["data_length"] += length
            hdr["mask_count"] += 1
            self._write_header(fp, **hdr)
            self._header_cache = hdr  # refresh

    # ------------------------------------------------------------------
    # Reading -----------------------------------------------------------
    # ------------------------------------------------------------------
    def load_segmask(self, name: str) -> "SegmentationMask":
        matches = [e for e in self._read_index() if e["name"] == name]
        if not matches:
            raise ValueError(f"SegmentationMask {name} not found")
        entry = matches[0]
        with open(self.path, "rb") as fp:
            fp.seek(entry["offset"])
            blob = fp.read(entry["length"])

        from medmask.core.segmask import SegmentationMask  # local import to avoid cycle

        arr = self.codec.decode(blob)
        mapping = LabelMapping(entry["mapping"])
        return SegmentationMask(arr, mapping, space=self.space)

    # helpers
    def all_names(self) -> List[str]:
        return [e["name"] for e in self._read_index()]

    def read_all_mapping(self) -> Dict[str, LabelMapping]:
        return {e["name"]: LabelMapping(e["mapping"]) for e in self._read_index()}

    def read_all_masks(self) -> Dict[str, "SegmentationMask"]:
        return {n: self.load_segmask(n) for n in self.all_names()} 