"""
MedMask - Medical Image Mask Compression and Processing Library

A specialized library for efficient compression, storage, and processing of 
medical image segmentation masks.
"""

from .core.segmask import SegmentationMask
from .core.mapping import LabelMapping
from .storage import MaskArchive, MaskFile, save_mask, load_mask  # noqa: F401
from .utils.utils import match_allowed_values

# 使用setuptools_scm从git tag获取版本号
try:
    from ._version import version as __version__
except ImportError:
    # 如果_version.py不存在（开发环境），回退到默认版本
    __version__ = "0.1.0"
__all__ = [
    "SegmentationMask",
    "LabelMapping", 
    "MaskArchive",
    "MaskFile",
    "save_mask",
    "load_mask",
    "match_allowed_values",
] 