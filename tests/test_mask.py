import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from medmask.core.segmask import SegmentationMask as Mask
from medmask.core.mapping import LabelMapping
from spacetransformer import Space


class TestSemanticMapping:
    """Test basic functionality of SemanticMapping class."""

    @pytest.fixture
    def mapping(self):
        """Create a basic semantic mapping for testing."""
        mapping = LabelMapping()
        mapping["lobe1"] = 1
        mapping["lobe2"] = 2
        return mapping

    def test_init(self):
        """Test initialization of LabelMapping."""
        # Test empty initialization
        mapping = LabelMapping()
        assert len(mapping._name_to_label) == 0
        assert len(mapping._label_to_name) == 0

        # Test initialization with dictionary
        init_dict = {"lobe1": 1, "lobe2": 2}
        mapping = LabelMapping(init_dict)
        assert mapping._name_to_label == init_dict
        assert mapping._label_to_name == {1: "lobe1", 2: "lobe2"}

    def test_setitem(self, mapping):
        """Test setting mapping values."""
        mapping["lobe3"] = 3
        assert mapping._name_to_label["lobe3"] == 3
        assert mapping._label_to_name[3] == "lobe3"

    def test_getitem(self, mapping):
        """Test getting label values by name."""
        assert mapping["lobe1"] == 1
        assert mapping["lobe2"] == 2
        with pytest.raises(KeyError):
            _ = mapping["nonexistent"]

    def test_getattr(self, mapping):
        """Test accessing labels via attribute access."""
        assert mapping.lobe1 == 1
        assert mapping.lobe2 == 2
        with pytest.raises(AttributeError):
            _ = mapping.nonexistent

    def test_json_conversion(self, mapping):
        """Test JSON serialization and deserialization."""
        json_str = mapping.to_json()
        new_mapping = LabelMapping.from_json(json_str)
        assert new_mapping._name_to_label == mapping._name_to_label
        assert new_mapping._label_to_name == mapping._label_to_name

    def test_inverse(self, mapping):
        """Test inverse mapping from label to name."""
        assert mapping.inverse(1) == "lobe1"
        assert mapping.inverse(2) == "lobe2"
        with pytest.raises(KeyError):
            _ = mapping.inverse(999)


class TestMask:
    """Test basic functionality of Mask class."""

    @pytest.fixture
    def shape(self):
        """Create basic shape for testing."""
        return (10, 10, 10)

    @pytest.fixture
    def space(self, shape):
        """Create Space object for testing."""
        return Space(shape=shape[::-1])

    @pytest.fixture
    def mask_array(self, shape):
        """Create basic mask array for testing."""
        arr = np.zeros(shape, dtype=np.uint8)
        arr[0:5, 0:5, 0:5] = 1
        arr[5:10, 5:10, 5:10] = 2
        return arr

    @pytest.fixture
    def mapping(self):
        """Create semantic mapping for testing."""
        return {"region1": 1, "region2": 2}

    def test_init(self, mask_array, mapping, space):
        """Test initialization of Mask."""
        # Test complete initialization
        mask = Mask(mask_array, mapping, space)
        assert mask.space == space
        np.testing.assert_array_equal(mask._mask_array, mask_array)
        assert mask.mapping["region1"] == 1
        assert mask.mapping["region2"] == 2

        # Test initialization without space
        mask = Mask(mask_array, mapping)
        assert mask.space.shape == mask_array.shape

    def test_lazy_init(self, space):
        """Test lazy initialization of Mask."""
        # Test initialization with space
        mask = Mask.lazy_init(8, space=space)
        assert mask.space == space
        assert mask._mask_array.dtype == np.uint8
        assert mask._mask_array.shape == space.shape

        # Test initialization with shape
        shape = (10, 10, 10)
        mask = Mask.lazy_init(8, shape=shape)
        assert mask.space.shape == shape
        assert mask._mask_array.shape == shape

        # Test different bit depths
        bit_depths = {1: np.bool_, 8: np.uint8, 16: np.uint16, 32: np.uint32}
        for bit_depth, dtype in bit_depths.items():
            mask = Mask.lazy_init(bit_depth, space=space)
            assert mask._mask_array.dtype == dtype

    def test_add_segmask(self, space):
        """Test adding single mask."""
        mask = Mask.lazy_init(8, space=space)

        # Add first mask
        submask = np.zeros(space.shape, dtype=bool)
        submask[0:5, 0:5, 0:5] = True
        mask.add_label(submask, 1, "region1")

        assert mask.mapping["region1"] == 1
        np.testing.assert_array_equal(mask._mask_array[0:5, 0:5, 0:5], 1)

        # Test adding duplicate label
        with pytest.raises(ValueError):
            mask.add_label(submask, 1, "region1_new")

    def test_get_binary_mask_by_names(self, mask_array, mapping, space):
        """Test getting mask by names."""
        mask = Mask(mask_array, mapping, space)

        # Test getting single mask
        result = mask.get_binary_mask_by_names("region1")
        expected = mask_array == 1
        np.testing.assert_array_equal(result, expected)

        # Test getting multiple masks
        result = mask.get_binary_mask_by_names(["region1", "region2"])
        expected = (mask_array == 1) | (mask_array == 2)
        np.testing.assert_array_equal(result, expected)

        # Test non-existent names
        with pytest.raises(KeyError):
            mask.get_binary_mask_by_names("nonexistent")

    def test_data(self, mask_array, mapping, space):
        """Test getting all mask data."""
        mask = Mask(mask_array, mapping, space)

        # Test getting original array
        np.testing.assert_array_equal(mask.data, mask_array)

        # Test getting binary array
        np.testing.assert_array_equal(mask.to_binary(), mask_array > 0)

    def test_label_name_conversion(self, mask_array, mapping, space):
        """Test conversion between labels and names."""
        mask = Mask(mask_array, mapping, space)

        assert mask.mapping["region1"] == 1
        assert mask.mapping["region2"] == 2
        with pytest.raises(KeyError):
            mask.mapping["nonexistent"]

        assert mask.mapping.inverse(1) == "region1"
        assert mask.mapping.inverse(2) == "region2"
        with pytest.raises(KeyError):
            mask.mapping.inverse(999)
