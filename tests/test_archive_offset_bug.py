"""Test offset bug in archive files with multiple independent masks.

When adding large numbers of masks to an archive, the index area expands
and causes data area reallocation. All existing entry offsets must be
updated correctly, otherwise wrong data will be read.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import tempfile
import os

from medmask.storage import MaskArchive
from medmask.core.segmask import SegmentationMask
from medmask.core.mapping import LabelMapping
from spacetransformer import Space


@pytest.fixture
def temp_archive_path():
    """Create temporary archive file path."""
    with tempfile.NamedTemporaryFile(suffix='.mska', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def test_space():
    """Create test space object."""
    return Space(shape=(10, 15, 20), spacing=(1.0, 1.0, 1.0))


@pytest.fixture 
def dummy_mask(test_space):
    """Create a simple dummy mask for repeated testing."""
    # Create simple mask data - note now using (z,y,x) format
    mask_array = np.zeros((10, 15, 20), dtype=np.uint8)  # (z,y,x) format
    mask_array[2:8, 3:12, 5:15] = 1  # Create a simple 3D rectangular region
    
    # Create semantic mapping
    mapping = LabelMapping({"test_organ": 1})
    
    # Create mask object
    mask = SegmentationMask(mask_array, mapping, test_space)
    return mask


def test_large_number_of_masks_offset_bug(temp_archive_path, test_space, dummy_mask):
    """Test offset bug when writing large number of masks - using repeated dummy mask."""
    
    num_masks = 100  # Enough to trigger multiple index expansions
    
    # Step 1: Create archive and add large number of masks
    archive_write = MaskArchive(temp_archive_path, mode="w", space=test_space)
    
    mask_names = []
    original_data = dummy_mask.data.copy()
    
    for i in range(num_masks):
        mask_name = f"mask_{i:03d}"
        archive_write.add_segmask(dummy_mask, mask_name)
        mask_names.append(mask_name)
    
    # Verify archive file was created
    assert os.path.exists(temp_archive_path)
    
    # Step 2: Reopen archive file for reading
    archive_read = MaskArchive(temp_archive_path, mode="r")
    
    # Verify mask name list is correct
    loaded_names = archive_read.all_names()
    assert len(loaded_names) == num_masks
    assert set(loaded_names) == set(mask_names)
    
    # Step 3: Test reading first 10 masks (these added before index expansion are most likely to have issues)
    test_names = mask_names[:10]
    
    for mask_name in test_names:
        try:
            loaded_mask = archive_read.load_segmask(mask_name)
            loaded_data = loaded_mask.data
            
            # Verify data integrity
            assert loaded_data.shape == original_data.shape, f"Shape mismatch for {mask_name}"
            assert loaded_data.dtype == original_data.dtype, f"Dtype mismatch for {mask_name}"
            np.testing.assert_array_equal(
                loaded_data, original_data, 
                err_msg=f"Data mismatch for {mask_name}"
            )
            
            # Verify semantic mapping
            assert "test_organ" in loaded_mask.mapping._name_to_label
            assert loaded_mask.mapping["test_organ"] == 1
            
        except Exception as e:
            pytest.fail(f"Failed to load mask '{mask_name}': {e}")


def test_index_expansion_triggers_offset_update(temp_archive_path, test_space, dummy_mask):
    """Test correct offset update when index expansion occurs."""
    
    archive_write = MaskArchive(temp_archive_path, mode="w", space=test_space)
    
    # Add enough masks to trigger index expansion
    # MaskArchive initial index capacity is 4000 bytes, each entry is approximately 80-100 bytes
    mask_names = []
    
    # Add 60 masks, enough to trigger index expansion
    for i in range(60):
        mask_name = f"test_mask_{i:03d}"
        archive_write.add_segmask(dummy_mask, mask_name)
        mask_names.append(mask_name)
    
    # Close write handle
    del archive_write
    
    # Reopen for reading
    archive_read = MaskArchive(temp_archive_path, mode="r")
    
    # Get index information for debugging
    index = archive_read._read_index()
    header = archive_read._read_header()
    
    print(f"Index entries: {len(index)}")
    print(f"Data start position: {header['data_offset']}")
    print(f"Index capacity: {header['index_length']}")
    
    # Check if the first few entries have correct offsets
    for i, entry in enumerate(index[:5]):
        offset = entry["offset"]
        assert offset >= header["data_offset"], f"Entry {i} offset {offset} is less than data start position {header['data_offset']}"
    
    # Test reading first 5 and last 5 masks
    test_indices = list(range(5)) + list(range(len(mask_names)-5, len(mask_names)))
    
    for i in test_indices:
        mask_name = mask_names[i]
        try:
            # Check raw data first
            entry = index[i]
            with open(temp_archive_path, "rb") as fp:
                fp.seek(entry["offset"])
                raw_data = fp.read(min(50, entry["length"]))
            
            # Ensure we're not reading JSON data (symptom of the bug)
            json_indicators = [b'{', b'"', b'[']
            for indicator in json_indicators:
                assert not raw_data.startswith(indicator), f"Mask {mask_name} data looks like JSON, offset may be wrong"
            
            # Actually read the mask
            loaded_mask = archive_read.load_segmask(mask_name)
            loaded_data = loaded_mask.data
            original_data = dummy_mask.data
            
            np.testing.assert_array_equal(loaded_data, original_data, 
                err_msg=f"Mask {mask_name} data mismatch")
                
        except Exception as e:
            pytest.fail(f"Failed to read mask '{mask_name}': {e}")


def test_simple_offset_bug_reproduction(temp_archive_path, test_space, dummy_mask):
    """Simplest offset bug reproduction test."""
    
    archive = MaskArchive(temp_archive_path, mode="w", space=test_space)
    
    # Add large number of masks to ensure index expansion
    for i in range(120):  # Exceed initial index capacity
        archive.add_segmask(dummy_mask, f"mask_{i}")
    
    # Immediately try to read the first mask in the same session
    # This is the most likely case to expose the offset bug
    try:
        first_mask = archive.load_segmask("mask_0")
        data = first_mask.data
        assert data.shape == dummy_mask.data.shape
        print("✅ Offset bug fixed!")
    except Exception as e:
        pytest.fail(f"❌ Offset bug still exists: {e}")


if __name__ == "__main__":
    # Can run this file directly for testing
    pytest.main([__file__, "-v"]) 