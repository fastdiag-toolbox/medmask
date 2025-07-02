"""
MedMask - Medical Image Mask Compression and Processing Library

A specialized library for efficient compression, storage, and processing of 
medical image segmentation masks.
"""

from .core.segmask import SegmentationMask
from .core.mapping import LabelMapping
from .storage import MaskArchive, MaskFile, save_mask, load_mask  # noqa: F401
from .utils.utils import match_allowed_values

__version__ = "1.1.0"
__all__ = [
    "SegmentationMask",
    "LabelMapping", 
    "MaskArchive",
    "MaskFile",
    "save_mask",
    "load_mask",
    "match_allowed_values",
] 