from __future__ import annotations

"""Storage subpackage – file persistence for segmentation masks.

This package provides:
1. `MaskFile` – single‐mask container (.msk)
2. `MaskArchive` – multi‐mask archive (.mska)
3. Convenience helpers `save_mask`, `load_mask` for functional usage.
"""

from .maskfile import MaskFile, save_mask, load_mask  # noqa: F401
from .archivefile import MaskArchive  # noqa: F401

__all__: list[str] = [
    "MaskFile",
    "MaskArchive",
    "save_mask",
    "load_mask",
] 