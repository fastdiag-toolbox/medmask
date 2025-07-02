import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from medmask.storage import MaskFile, save_mask, load_mask
from medmask.core.segmask import SegmentationMask
from medmask.core.mapping import LabelMapping
from spacetransformer import Space


@pytest.fixture
def temp_msk_path(tmp_path):
    """Return a temporary .msk path."""
    return str(tmp_path / "sample.msk")


@pytest.fixture
def space():
    return Space(shape=(8, 16, 24), spacing=(1.0, 1.0, 1.0))


@pytest.fixture
def sample_mask(space):
    arr = np.zeros((24, 16, 8), dtype=np.uint8)
    arr[5:15, 4:12, 2:6] = 1
    mapping = LabelMapping({"lesion": 1})
    return SegmentationMask(arr, mapping, space)


# -----------------------------------------------------------------------------
# MaskFile API
# -----------------------------------------------------------------------------

def test_write_and_read_maskfile(temp_msk_path, sample_mask):
    mf = MaskFile(temp_msk_path, mode="w")
    mf.write(sample_mask)
    assert os.path.exists(temp_msk_path)

    mf2 = MaskFile(temp_msk_path)
    loaded = mf2.read()
    np.testing.assert_array_equal(loaded.data, sample_mask.data)
    assert loaded.space == sample_mask.space
    assert str(loaded.mapping) == str(sample_mask.mapping)


def test_functional_helpers(temp_msk_path, sample_mask):
    save_mask(sample_mask, temp_msk_path)
    loaded = load_mask(temp_msk_path)
    np.testing.assert_array_equal(loaded.data, sample_mask.data)


def test_segmask_methods(temp_msk_path, sample_mask):
    sample_mask.save(temp_msk_path)
    loaded = SegmentationMask.load(temp_msk_path)
    np.testing.assert_array_equal(loaded.data, sample_mask.data)


def test_header_validation(temp_msk_path, sample_mask):
    # write then manually corrupt magic number
    mf = MaskFile(temp_msk_path, "w")
    mf.write(sample_mask)

    with open(temp_msk_path, "r+b") as fp:
        fp.write(b"BAD!")

    with pytest.raises(ValueError):
        MaskFile(temp_msk_path).read() 