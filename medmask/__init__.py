"""
MedMask - Medical Image Mask Compression and Processing Library

A specialized library for efficient compression, storage, and processing of 
medical image segmentation masks.
"""

from .core.segmask import SegmentationMask
from .core.mapping import LabelMapping
from .archive import MaskArchive
from .io.utils import match_allowed_values

__version__ = "1.1.0"
__all__ = [
    "SegmentationMask",
    "LabelMapping", 
    "MaskArchive",
    "match_allowed_values",
] 