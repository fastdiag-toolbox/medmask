"""Mask archive container format (binary .maska files).

For details see :pyclass:`medmask.archive.archive_file.MaskArchive`.
"""

from .archive_file import MaskArchive  # noqa: F401 – re-export

__all__ = ["MaskArchive"] 