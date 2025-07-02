import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from medmask.archive import MaskArchive as MaskFile
from medmask.core.segmask import SegmentationMask as Mask
from medmask.core.mapping import LabelMapping
from spacetransformer import Space


@pytest.fixture
def temp_maskfile_file(tmp_path):
    """创建临时 DCBM 文件路径"""
    return str(tmp_path / "test.maska")


@pytest.fixture
def space():
    """创建测试用的 Space 对象"""
    return Space(shape=(10, 20, 30), spacing=(1.0, 1.0, 1.0))


@pytest.fixture
def sample_mask(space):
    """创建测试用的 Mask 对象"""
    # 创建一个简单的掩码数组
    mask_array = np.zeros((30, 20, 10), dtype=np.uint8)
    mask_array[10:20, 5:15, 2:8] = 1
    mask_array[15:25, 10:20, 3:9] = 2

    # 创建语义映射
    mapping = LabelMapping()
    mapping["region1"] = 1
    mapping["region2"] = 2

    return Mask(mask_array, mapping=mapping, space=space)


def test_create_empty_file(temp_maskfile_file, space):
    """测试创建空的 DCBM 文件"""
    maskfile = MaskFile(temp_maskfile_file, "w", space=space)
    assert not os.path.exists(temp_maskfile_file)  # 文件不应该被创建
    assert maskfile.space == space


def test_add_first_mask(temp_maskfile_file, space, sample_mask):
    """测试添加第一个掩码"""
    maskfile = MaskFile(temp_maskfile_file, "w", space=space)
    maskfile.add_segmask(sample_mask, "test_mask")

    # 验证文件已创建
    assert os.path.exists(temp_maskfile_file)

    # 验证掩码列表
    assert maskfile.all_names() == ["test_mask"]

    # 读取并验证掩码
    loaded_mask = maskfile.load_segmask("test_mask")
    assert np.array_equal(loaded_mask.data, sample_mask.data)
    assert loaded_mask.space == sample_mask.space
    assert str(loaded_mask.mapping) == str(sample_mask.mapping)


def test_add_multiple_masks(temp_maskfile_file, space, sample_mask):
    """测试添加多个掩码"""
    maskfile = MaskFile(temp_maskfile_file, "w", space=space)

    # 添加第一个掩码
    maskfile.add_segmask(sample_mask, "mask1")

    # 创建并添加第二个掩码
    mask2_array = np.zeros((30, 20, 10), dtype=np.uint8)
    mask2_array[5:15, 8:18, 4:9] = 3
    mapping2 = LabelMapping()
    mapping2["region3"] = 3
    mask2 = Mask(mask2_array, mapping=mapping2, space=space)
    maskfile.add_segmask(mask2, "mask2")

    # 验证掩码列表
    assert set(maskfile.all_names()) == {"mask1", "mask2"}

    # 读取并验证两个掩码
    loaded_masks = maskfile.read_all_masks()
    assert len(loaded_masks) == 2
    assert np.array_equal(
        loaded_masks["mask1"].data, sample_mask.data
    )
    assert np.array_equal(loaded_masks["mask2"].data, mask2.data)


def test_add_duplicate_mask(temp_maskfile_file, space, sample_mask):
    """测试添加重复名称的掩码"""
    maskfile = MaskFile(temp_maskfile_file, "w", space=space)
    maskfile.add_segmask(sample_mask, "test_mask")

    # 尝试添加同名掩码
    with pytest.raises(ValueError, match="Mask test_mask already exists"):
        maskfile.add_segmask(sample_mask, "test_mask")


def test_load_nonexistent_mask(temp_maskfile_file, space, sample_mask):
    """测试加载不存在的掩码"""
    maskfile = MaskFile(temp_maskfile_file, "w", space=space)
    maskfile.add_segmask(sample_mask, "test_mask")

    with pytest.raises(ValueError, match="Mask nonexistent not found"):
        maskfile.load_segmask("nonexistent")


def test_space_mismatch(temp_maskfile_file, space, sample_mask):
    """测试空间不匹配的情况"""
    maskfile = MaskFile(temp_maskfile_file, "w", space=space)
    maskfile.add_segmask(sample_mask, "mask1")

    # 创建一个不同的空间
    different_space = Space(shape=(10, 20, 31), spacing=(1.0, 1.0, 1.0))
    mask_array = np.zeros((31, 20, 10), dtype=np.uint8)  # 注意这里的形状
    mapping = LabelMapping()
    mapping["region1"] = 1
    different_mask = Mask(mask_array, mapping=mapping, space=different_space)

    # 尝试添加空间不匹配的掩码
    with pytest.raises(AssertionError):
        maskfile.add_segmask(different_mask, "mask2")


def test_read_all_mapping(temp_maskfile_file, space, sample_mask):
    """测试读取所有语义映射"""
    maskfile = MaskFile(temp_maskfile_file, "w", space=space)
    maskfile.add_segmask(sample_mask, "test_mask")

    mapping_dict = maskfile.read_all_mapping()
    assert len(mapping_dict) == 1
    assert "test_mask" in mapping_dict
    assert mapping_dict["test_mask"]["region1"] == 1
    assert mapping_dict["test_mask"]["region2"] == 2


def test_file_persistence(temp_maskfile_file, space, sample_mask):
    """测试文件持久化"""
    # 创建文件并添加掩码
    maskfile1 = MaskFile(temp_maskfile_file, "w", space=space)
    maskfile1.add_segmask(sample_mask, "test_mask")

    # 创建新的 MaskFile 实例读取同一个文件
    maskfile2 = MaskFile(temp_maskfile_file)

    # 验证空间信息
    assert maskfile2.space == space

    # 验证掩码数据
    loaded_mask = maskfile2.load_segmask("test_mask")
    assert np.array_equal(loaded_mask.data, sample_mask.data)
    assert str(loaded_mask.mapping) == str(sample_mask.mapping)


def test_large_mask(temp_maskfile_file, space):
    """测试大型掩码的处理"""
    # 创建一个较大的掩码
    large_array = np.random.randint(0, 2, size=(30, 20, 10), dtype=np.uint8)
    mapping = LabelMapping()
    mapping["region"] = 1
    large_mask = Mask(large_array, mapping=mapping, space=space)

    # 添加并读取大型掩码
    maskfile = MaskFile(temp_maskfile_file, "w", space=space)
    maskfile.add_segmask(large_mask, "large_mask")

    loaded_mask = maskfile.load_segmask("large_mask")
    assert np.array_equal(loaded_mask.data, large_mask.data)


def test_semantic_mapping():
    """测试 LabelMapping 类的基本功能"""
    mapping = LabelMapping()

    # 测试添加映射
    mapping["region1"] = 1
    assert mapping["region1"] == 1
    assert mapping("region1") == 1
    assert mapping.has_label(1)
    assert not mapping.has_label(2)

    # 测试反向映射
    assert mapping.inverse(1) == "region1"

    # 测试属性访问
    assert mapping.region1 == 1

    # 测试迭代和长度
    mapping["region2"] = 2
    assert len(mapping) == 2
    assert set(mapping) == {"region1", "region2"}
    assert dict(mapping.items()) == {"region1": 1, "region2": 2}

    # 测试不存在的映射
    with pytest.raises(KeyError):
        _ = mapping["nonexistent"]
    with pytest.raises(AttributeError):
        _ = mapping.nonexistent
    with pytest.raises(KeyError):
        _ = mapping.inverse(999)


def test_semantic_mapping_json():
    """测试 LabelMapping 的 JSON 序列化"""
    mapping1 = LabelMapping()
    mapping1["region1"] = 1
    mapping1["region2"] = 2

    # 转换为 JSON
    json_str = mapping1.to_json()

    # 从 JSON 恢复
    mapping2 = LabelMapping.from_json(json_str)

    # 验证恢复的映射
    assert mapping2["region1"] == 1
    assert mapping2["region2"] == 2
    assert mapping2.has_label(1)
    assert mapping2.has_label(2)
    assert mapping2.inverse(1) == "region1"
    assert mapping2.inverse(2) == "region2"

    # 验证 JSON 格式
    assert json_str == '{"region1": 1, "region2": 2}'


def test_semantic_mapping_bidirectional():
    """测试 LabelMapping 的双向映射功能"""
    mapping = LabelMapping()

    # 测试正向映射（str -> int）
    mapping["region1"] = 1
    mapping["region2"] = 2
    assert mapping["region1"] == 1
    assert mapping("region2") == 2

    # 测试反向映射（int -> str）
    assert mapping.inverse(1) == "region1"
    assert mapping.inverse(2) == "region2"

    # 测试映射的唯一性
    with pytest.raises(ValueError, match="Label 1 already exists"):
        mapping["new_region"] = 1  # 尝试添加重复的标签值

    # 测试更新现有映射
    mapping["region1"] = 3  # 更新现有的映射
    assert mapping["region1"] == 3
    assert mapping.inverse(3) == "region1"
    assert not mapping.has_label(1)  # 原来的标签值应该被移除
