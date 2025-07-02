from __future__ import annotations

"""I/O helpers.

*The legacy ``MaskFile`` implementation has been superseded by
:class:`medmask.archive.MaskArchive`.  This subpackage now only exposes
utility helpers kept for historical reasons.*
"""

from .utils import match_allowed_values

__all__ = ["match_allowed_values"] 