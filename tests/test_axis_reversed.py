import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import tempfile
import os

from medmask.core.segmask import SegmentationMask
from medmask.core.mapping import LabelMapping
from medmask.storage import MaskFile
from spacetransformer import Space


def test_unified_axis_behavior():
    """Test unified (z,y,x) axis order behavior."""
    # Space takes shape in (z,y,x) order
    zyx_shape = (8, 16, 24)
    space = Space(shape=zyx_shape)

    # Create array in (z,y,x) format
    arr = np.arange(np.prod(zyx_shape), dtype=np.uint8).reshape(zyx_shape)

    mask = SegmentationMask(arr, mapping=LabelMapping({"bg": 0}), space=space)

    # 1. data always returns (z,y,x) format
    assert mask.data.shape == zyx_shape
    np.testing.assert_array_equal(mask.data, arr)

    # 2. data is read-only
    with pytest.raises(ValueError):
        mask.data[0, 0, 0] = 99

    # 3. space shape should match array shape
    assert mask.space.shape == zyx_shape


def test_space_array_shape_consistency():
    """Test space and array shape consistency check."""
    # Correct case
    shape = (4, 8, 12)
    space = Space(shape=shape)
    arr = np.zeros(shape, dtype=np.uint8)
    
    # Should create successfully
    mask = SegmentationMask(arr, mapping={}, space=space)
    assert mask.data.shape == shape
    assert mask.space.shape == shape
    
    # Error case: shape mismatch
    wrong_shape = (4, 8, 10)  # Last dimension doesn't match
    wrong_space = Space(shape=wrong_shape)
    
    with pytest.raises(AssertionError):
        SegmentationMask(arr, mapping={}, space=wrong_space)


def test_lazy_init_unified_behavior():
    """Test unified behavior of lazy_init."""
    shape = (6, 10, 14)
    space = Space(shape=shape)
    
    # Create using space
    mask1 = SegmentationMask.lazy_init(8, space=space)
    assert mask1.data.shape == shape
    assert mask1.space.shape == shape
    
    # Create using shape
    mask2 = SegmentationMask.lazy_init(16, shape=shape)
    assert mask2.data.shape == shape
    assert mask2.space.shape == shape


def test_maskfile_unified_io():
    """Test unified file I/O behavior."""
    zyx_shape = (6, 10, 14)
    space = Space(shape=zyx_shape)

    arr = np.random.randint(0, 3, size=zyx_shape, dtype=np.uint8)
    mask = SegmentationMask(arr, LabelMapping({"obj": 1}), space)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.msk")
        mf = MaskFile(path, mode="w")
        mf.write(mask)

        loaded = MaskFile(path).read()
        
        # Verify data consistency
        np.testing.assert_array_equal(loaded.data, mask.data)
        assert loaded.data.shape == zyx_shape
        assert loaded.space.shape == zyx_shape
        
        # Verify space information consistency
        assert loaded.space.shape == mask.space.shape
        assert np.allclose(loaded.space.spacing, mask.space.spacing)
        assert np.allclose(loaded.space.origin, mask.space.origin)


def test_cross_language_compatibility():
    """Test cross-language compatibility design."""
    # Create test data
    zyx_shape = (4, 6, 8)
    space = Space(shape=zyx_shape, spacing=(1.0, 2.0, 3.0), origin=(10, 20, 30))
    arr = np.random.randint(0, 4, size=zyx_shape, dtype=np.uint8)
    mask = SegmentationMask(arr, LabelMapping({"organ": 1}), space)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cross_lang.msk")
        
        # Save
        mask.save(path)
        
        # Load
        loaded = SegmentationMask.load(path)
        
        # Python user perspective: data(z,y,x) + space(z,y,x) → aligned
        assert loaded.data.shape == zyx_shape
        assert loaded.space.shape == zyx_shape
        np.testing.assert_array_equal(loaded.data, arr)
        
        # Verify correct space information conversion
        assert loaded.space.spacing == space.spacing
        assert loaded.space.origin == space.origin
