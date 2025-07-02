"""测试归档文件中多个独立掩膜的偏移量bug

当向归档中添加大量掩膜时，索引区域会扩容并导致数据区域重新分配，
此时必须更新所有已有条目的偏移量，否则会读取到错误的数据。
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
    """创建临时归档文件路径"""
    with tempfile.NamedTemporaryFile(suffix='.maska', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # 清理临时文件
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def test_space():
    """创建测试用的空间对象"""
    return Space(shape=(10, 15, 20), spacing=(1.0, 1.0, 1.0))


@pytest.fixture 
def dummy_mask(test_space):
    """创建一个简单的dummy掩膜用于重复测试"""
    # 创建简单的掩膜数据
    mask_array = np.zeros((20, 15, 10), dtype=np.uint8)
    mask_array[5:15, 3:12, 2:8] = 1  # 创建一个简单的3D矩形区域
    
    # 创建语义映射
    mapping = LabelMapping({"test_organ": 1})
    
    # 创建掩膜对象
    mask = SegmentationMask(mask_array, mapping, test_space)
    return mask


def test_large_number_of_masks_offset_bug(temp_archive_path, test_space, dummy_mask):
    """测试大量掩膜写入时的偏移量bug - 使用重复的dummy mask"""
    
    num_masks = 100  # 足够触发多次索引扩容
    
    # 步骤1: 创建归档并添加大量掩膜
    archive_write = MaskArchive(temp_archive_path, mode="w", space=test_space)
    
    mask_names = []
    original_data = dummy_mask.data.copy()
    
    for i in range(num_masks):
        mask_name = f"mask_{i:03d}"
        archive_write.add_segmask(dummy_mask, mask_name)
        mask_names.append(mask_name)
    
    # 验证归档文件已创建
    assert os.path.exists(temp_archive_path)
    
    # 步骤2: 重新打开归档文件进行读取
    archive_read = MaskArchive(temp_archive_path, mode="r")
    
    # 验证掩膜名称列表正确
    loaded_names = archive_read.all_names()
    assert len(loaded_names) == num_masks
    assert set(loaded_names) == set(mask_names)
    
    # 步骤3: 测试前10个掩膜的读取（这些在索引扩容前添加的最容易出问题）
    test_names = mask_names[:10]
    
    for mask_name in test_names:
        try:
            loaded_mask = archive_read.load_segmask(mask_name)
            loaded_data = loaded_mask.data
            
            # 验证数据完整性
            assert loaded_data.shape == original_data.shape, f"Shape mismatch for {mask_name}"
            assert loaded_data.dtype == original_data.dtype, f"Dtype mismatch for {mask_name}"
            np.testing.assert_array_equal(
                loaded_data, original_data, 
                err_msg=f"Data mismatch for {mask_name}"
            )
            
            # 验证语义映射
            assert "test_organ" in loaded_mask.mapping._name_to_label
            assert loaded_mask.mapping["test_organ"] == 1
            
        except Exception as e:
            pytest.fail(f"Failed to load mask '{mask_name}': {e}")


def test_index_expansion_triggers_offset_update(temp_archive_path, test_space, dummy_mask):
    """专门测试索引扩容时偏移量更新的正确性"""
    
    archive_write = MaskArchive(temp_archive_path, mode="w", space=test_space)
    
    # 添加足够多的掩膜来触发索引扩容
    # MaskArchive的初始索引容量是4000 bytes，每个条目大约80-100 bytes
    mask_names = []
    
    # 添加60个掩膜，足够触发索引扩容
    for i in range(60):
        mask_name = f"test_mask_{i:03d}"
        archive_write.add_segmask(dummy_mask, mask_name)
        mask_names.append(mask_name)
    
    # 关闭写入句柄
    del archive_write
    
    # 重新打开读取
    archive_read = MaskArchive(temp_archive_path, mode="r")
    
    # 获取索引信息进行调试
    index = archive_read._read_index()
    header = archive_read._read_header()
    
    print(f"索引条目数: {len(index)}")
    print(f"数据开始位置: {header['data_offset']}")
    print(f"索引容量: {header['index_length']}")
    
    # 检查前几个条目的偏移量是否正确
    for i, entry in enumerate(index[:5]):
        offset = entry["offset"]
        assert offset >= header["data_offset"], f"条目 {i} 的偏移量 {offset} 小于数据开始位置 {header['data_offset']}"
    
    # 测试读取前5个和最后5个掩膜
    test_indices = list(range(5)) + list(range(len(mask_names)-5, len(mask_names)))
    
    for i in test_indices:
        mask_name = mask_names[i]
        try:
            # 先检查原始数据
            entry = index[i]
            with open(temp_archive_path, "rb") as fp:
                fp.seek(entry["offset"])
                raw_data = fp.read(min(50, entry["length"]))
            
            # 确保读取的不是JSON数据（bug的症状）
            json_indicators = [b'{', b'"', b'[']
            for indicator in json_indicators:
                assert not raw_data.startswith(indicator), f"掩膜 {mask_name} 的数据看起来像JSON，偏移量可能错误"
            
            # 实际读取掩膜
            loaded_mask = archive_read.load_segmask(mask_name)
            loaded_data = loaded_mask.data
            original_data = dummy_mask.data
            
            np.testing.assert_array_equal(loaded_data, original_data, 
                err_msg=f"掩膜 {mask_name} 数据不匹配")
                
        except Exception as e:
            pytest.fail(f"读取掩膜 '{mask_name}' 失败: {e}")


def test_simple_offset_bug_reproduction(temp_archive_path, test_space, dummy_mask):
    """最简单的偏移量bug复现测试"""
    
    archive = MaskArchive(temp_archive_path, mode="w", space=test_space)
    
    # 添加大量掩膜，确保触发索引扩容
    for i in range(120):  # 超过初始索引容量
        archive.add_segmask(dummy_mask, f"mask_{i}")
    
    # 立即在同一个session中尝试读取第一个掩膜
    # 这是最容易暴露偏移量bug的情况
    try:
        first_mask = archive.load_segmask("mask_0")
        data = first_mask.data
        assert data.shape == dummy_mask.data.shape
        print("✅ 偏移量bug已修复！")
    except Exception as e:
        pytest.fail(f"❌ 偏移量bug仍然存在: {e}")


if __name__ == "__main__":
    # 可以直接运行这个文件进行测试
    pytest.main([__file__, "-v"]) 