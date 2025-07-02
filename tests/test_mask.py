import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from medmask.core.segmask import SegmentationMask as Mask
from medmask.core.mapping import LabelMapping
from spacetransformer import Space


class TestSemanticMapping:
    """测试 SemanticMapping 类的基本功能"""

    @pytest.fixture
    def mapping(self):
        """创建一个基本的语义映射用于测试"""
        mapping = LabelMapping()
        mapping["lobe1"] = 1
        mapping["lobe2"] = 2
        return mapping

    def test_init(self):
        """测试初始化"""
        # 测试空初始化
        mapping = LabelMapping()
        assert len(mapping._name_to_label) == 0
        assert len(mapping._label_to_name) == 0

        # 测试带字典初始化
        init_dict = {"lobe1": 1, "lobe2": 2}
        mapping = LabelMapping(init_dict)
        assert mapping._name_to_label == init_dict
        assert mapping._label_to_name == {1: "lobe1", 2: "lobe2"}

    def test_setitem(self, mapping):
        """测试设置映射"""
        mapping["lobe3"] = 3
        assert mapping._name_to_label["lobe3"] == 3
        assert mapping._label_to_name[3] == "lobe3"

    def test_getitem(self, mapping):
        """测试获取标签值"""
        assert mapping["lobe1"] == 1
        assert mapping["lobe2"] == 2
        with pytest.raises(KeyError):
            _ = mapping["nonexistent"]

    def test_getattr(self, mapping):
        """测试通过属性访问"""
        assert mapping.lobe1 == 1
        assert mapping.lobe2 == 2
        with pytest.raises(AttributeError):
            _ = mapping.nonexistent

    def test_json_conversion(self, mapping):
        """测试 JSON 转换"""
        json_str = mapping.to_json()
        new_mapping = LabelMapping.from_json(json_str)
        assert new_mapping._name_to_label == mapping._name_to_label
        assert new_mapping._label_to_name == mapping._label_to_name

    def test_inverse(self, mapping):
        """测试反向映射"""
        assert mapping.inverse(1) == "lobe1"
        assert mapping.inverse(2) == "lobe2"
        with pytest.raises(KeyError):
            _ = mapping.inverse(999)


class TestMask:
    """测试 Mask 类的基本功能"""

    @pytest.fixture
    def shape(self):
        """创建基本形状"""
        return (10, 10, 10)

    @pytest.fixture
    def space(self, shape):
        """创建 Space 对象"""
        return Space(shape=shape[::-1])

    @pytest.fixture
    def mask_array(self, shape):
        """创建基本的 mask 数组"""
        arr = np.zeros(shape, dtype=np.uint8)
        arr[0:5, 0:5, 0:5] = 1
        arr[5:10, 5:10, 5:10] = 2
        return arr

    @pytest.fixture
    def mapping(self):
        """创建测试用的语义映射"""
        return {"region1": 1, "region2": 2}

    def test_init(self, mask_array, mapping, space):
        """测试初始化"""
        # 测试完整初始化
        mask = Mask(mask_array, mapping, space)
        assert mask.space == space
        np.testing.assert_array_equal(mask._mask_array, mask_array)
        assert mask.mapping["region1"] == 1
        assert mask.mapping["region2"] == 2

        # 测试不带 space 初始化
        mask = Mask(mask_array, mapping)
        assert mask.space.shape == mask_array.shape[::-1]

    def test_lazy_init(self, space):
        """测试延迟初始化"""
        # 使用 space 初始化
        mask = Mask.lazy_init(8, space=space)
        assert mask.space == space
        assert mask._mask_array.dtype == np.uint8
        assert mask._mask_array.shape == space.shape[::-1]

        # 使用 shape 初始化
        shape = (10, 10, 10)
        mask = Mask.lazy_init(8, shape_zyx=shape)
        assert mask.space.shape == shape[::-1]
        assert mask._mask_array.shape == shape

        # 测试不同位深
        bit_depths = {1: np.bool_, 8: np.uint8, 16: np.uint16, 32: np.uint32}
        for bit_depth, dtype in bit_depths.items():
            mask = Mask.lazy_init(bit_depth, space=space)
            assert mask._mask_array.dtype == dtype

    def test_add_segmask(self, space):
        """测试添加单个 mask"""
        mask = Mask.lazy_init(8, space=space)

        # 添加第一个 mask
        submask = np.zeros(space.shape[::-1], dtype=bool)
        submask[0:5, 0:5, 0:5] = True
        mask.add_label(submask, 1, "region1")

        assert mask.mapping["region1"] == 1
        np.testing.assert_array_equal(mask._mask_array[0:5, 0:5, 0:5], 1)

        # 测试添加重复标签
        with pytest.raises(ValueError):
            mask.add_label(submask, 1, "region1_new")

    def test_get_binary_mask_by_names(self, mask_array, mapping, space):
        """测试通过名称获取 mask"""
        mask = Mask(mask_array, mapping, space)

        # 测试获取单个 mask
        result = mask.get_binary_mask_by_names("region1")
        expected = mask_array == 1
        np.testing.assert_array_equal(result, expected)

        # 测试获取多个 mask
        result = mask.get_binary_mask_by_names(["region1", "region2"])
        expected = (mask_array == 1) | (mask_array == 2)
        np.testing.assert_array_equal(result, expected)

        # 测试不存在的名称
        with pytest.raises(KeyError):
            mask.get_binary_mask_by_names("nonexistent")

    def test_data(self, mask_array, mapping, space):
        """测试获取所有 mask"""
        mask = Mask(mask_array, mapping, space)

        # 测试获取原始数组
        np.testing.assert_array_equal(mask.data, mask_array)

        # 测试获取二值化数组
        np.testing.assert_array_equal(mask.to_binary(), mask_array > 0)

    def test_label_name_conversion(self, mask_array, mapping, space):
        """测试标签和名称之间的转换"""
        mask = Mask(mask_array, mapping, space)

        assert mask.mapping["region1"] == 1
        assert mask.mapping["region2"] == 2
        with pytest.raises(KeyError):
            mask.mapping["nonexistent"]

        assert mask.mapping.inverse(1) == "region1"
        assert mask.mapping.inverse(2) == "region2"
        with pytest.raises(KeyError):
            mask.mapping.inverse(999)
