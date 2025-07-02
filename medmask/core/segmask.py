from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import warnings

from spacetransformer import Space

from ..utils.utils import match_allowed_values
from .mapping import LabelMapping

__all__ = ["SegmentationMask", "Mask"]


class SegmentationMask:
    """Represents a 3-D segmentation mask with semantic labels.

    A *segmentation mask* is a 3-D ndarray whose voxel values represent integer
    *labels* (e.g. 0=background, 1=liver, 2=spleen …).  This class stores the
    mask array itself together with its :class:`~spacetransformer.core.space.Space`
    (geometry) and a bi-directional mapping between *names* ("liver") and
    *labels* (1).

    There are two ways to build a mask instance:

    1. **Complete initialisation** – provide a full ndarray and a mapping.
    2. **Lazy initialisation** – create an empty mask of the desired *bit-depth*
       first via :meth:`lazy_init`, then add label regions incrementally with
       :meth:`add_label`.
    """

    # ------------------------------------------------------------------
    # Construction ------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(
        self,
        mask_array: np.ndarray,
        mapping: Union[Dict[str, int], LabelMapping],
        space: Optional[Space] = None,
    ) -> None:
        # ---------- space ---------------------------------------------
        if space is not None:
            # sanity check – ensure shape matches
            assert np.allclose(
                space.shape_zyx, mask_array.shape
            ), f"space.shape_zyx: {space.shape_zyx}, mask_array.shape: {mask_array.shape}"
            self.space: Space = space
        else:
            # construct a default space assuming isotropic spacing=1mm and
            # xyz order – the external Space expects (x,y,z) order, hence
            # reverse the ndarray shape (which is z,y,x)
            self.space = Space(shape=mask_array.shape[::-1])

        # ---------- semantic mapping -----------------------------------
        if isinstance(mapping, LabelMapping):
            self.mapping: LabelMapping = mapping
        else:
            self.mapping = LabelMapping(mapping)

        # ---------- data ----------------------------------------------
        self._mask_array: np.ndarray = mask_array

        # internal cache of existing labels to speed-up checks
        self._existing_labels: set[int] = set(self.mapping._label_to_name.keys())
        self._sync_labels_with_array()

    # ------------------------------------------------------------------
    # Convenience -------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def bit_depth(self) -> int:
        """Bit-depth of the underlying array (1 / 8 / 16 / 32)."""
        dtype = self._mask_array.dtype
        if dtype == np.bool_:
            return 1
        if dtype == np.uint8:
            return 8
        if dtype == np.uint16:
            return 16
        if dtype == np.uint32:
            return 32
        raise ValueError(f"Unsupported dtype: {dtype}")

    # ------------------------------------------------------------------
    # Lazy constructor --------------------------------------------------
    # ------------------------------------------------------------------
    @classmethod
    def lazy_init(
        cls,
        bit_depth: int,
        *,
        space: Optional[Space] = None,
        shape_zyx: Optional[Tuple[int, int, int]] = None,
    ) -> "SegmentationMask":
        """Create an empty mask with given *bit-depth*.

        Either *space* or *shape_zyx* must be supplied to infer the array
        dimensions.
        """
        if space is None and shape_zyx is None:
            raise ValueError("Either space or shape_zyx must be provided.")

        if space is not None:
            shape_zyx = space.shape[::-1]  # xyz → zyx

        dtype_lookup = {1: np.bool_, 8: np.uint8, 16: np.uint16, 32: np.uint32}
        if bit_depth not in dtype_lookup:
            raise ValueError("bit_depth must be one of 1/8/16/32")

        mask_array = np.zeros(shape_zyx, dtype=dtype_lookup[bit_depth])
        return cls(mask_array, mapping={}, space=space)

    # ------------------------------------------------------------------
    # Editing -----------------------------------------------------------
    # ------------------------------------------------------------------
    def add_label(self, mask: np.ndarray, label: int, name: str) -> None:
        """Paint a *mask* region with *label* and register *name*.

        `mask` must be a boolean ndarray of the same shape as this mask.
        """
        if self.mapping.has_label(label):
            raise ValueError(f"Label {label} already exists in the mask array.")
        if label >= 2 ** self.bit_depth:
            raise ValueError(f"Label {label} exceeds bit-depth limit ({self.bit_depth}).")

        if mask.dtype != np.bool_:
            mask = mask > 0

        self._mask_array = np.where(mask, label, self._mask_array)
        self._existing_labels.add(label)
        self.mapping[name] = label

    # ------------------------------------------------------------------
    # Query -------------------------------------------------------------
    # ------------------------------------------------------------------
    def get_binary_mask_by_names(self, names: Union[str, List[str]]) -> np.ndarray:
        if isinstance(names, str):
            return self._mask_array == self.mapping[names]
        labels = [self.mapping[n] for n in names]
        return self.get_binary_mask_by_labels(labels)

    def get_binary_mask_by_labels(self, labels: Union[int, List[int]]) -> np.ndarray:
        if isinstance(labels, int):
            return self._mask_array == labels
        return match_allowed_values(self._mask_array, labels)

    # ------------------------------------------------------------------
    # Array access ------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """Raw ndarray with integer labels (read-only view)."""
        arr = self._mask_array.view()
        arr.flags.writeable = False  # ensure read-only
        return arr

    def to_binary(self) -> np.ndarray:
        """Return a boolean array where non-zero voxels are *True*."""
        return self._mask_array.astype(bool)


    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------
    def _sync_labels_with_array(self):
        """
        Ensure that every label present in the mask array is defined in the semantic mapping.
        If a label is missing, assign a default semantic name.
        """
        for label in self._existing_labels:
            if not self.mapping.has_label(
                label
            ):  # If the semantic mapping lacks this label, add a default name.
                self.mapping[f"idx_{label}"] = label

    # ------------------------------------------------------------------
    # Representation ----------------------------------------------------
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover (human readable)
        return (
            f"SegmentationMask(shape_zyx={self.space.shape_zyx}, "
            f"labels={sorted(self._existing_labels)}, mapping={self.mapping._name_to_label})"
        )

    # ------------------------------------------------------------------
    # Persistence ------------------------------------------------------
    # ------------------------------------------------------------------
    def save(self, path: str, *, codec: str | None = None) -> None:
        """Save this mask to *path* using the storage layer (`MaskFile`)."""
        from ..storage import save_mask  # local import to avoid heavy deps

        save_mask(self, path, codec=codec)

    @classmethod
    def load(cls, path: str):  # noqa: D401 – simple wrapper
        """Load a mask from *path* (.msk) and return a new `SegmentationMask`."""
        from ..storage import load_mask

        return load_mask(path)

