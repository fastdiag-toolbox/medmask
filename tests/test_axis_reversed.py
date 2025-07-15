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
    """测试统一的 (z,y,x) 轴序行为。"""
    # Space 以 (z,y,x) 顺序给 shape
    zyx_shape = (8, 16, 24)
    space = Space(shape=zyx_shape)

    # 创建 (z,y,x) 格式的数组
    arr = np.arange(np.prod(zyx_shape), dtype=np.uint8).reshape(zyx_shape)

    mask = SegmentationMask(arr, mapping=LabelMapping({"bg": 0}), space=space)

    # 1. data 始终返回 (z,y,x) 格式
    assert mask.data.shape == zyx_shape
    np.testing.assert_array_equal(mask.data, arr)

    # 2. data 是只读的
    with pytest.raises(ValueError):
        mask.data[0, 0, 0] = 99

    # 3. space 的形状应该与数组形状一致
    assert mask.space.shape == zyx_shape


def test_space_array_shape_consistency():
    """测试 space 和 array 的形状一致性检查。"""
    # 正确的情况
    shape = (4, 8, 12)
    space = Space(shape=shape)
    arr = np.zeros(shape, dtype=np.uint8)
    
    # 应该成功创建
    mask = SegmentationMask(arr, mapping={}, space=space)
    assert mask.data.shape == shape
    assert mask.space.shape == shape
    
    # 错误的情况：形状不匹配
    wrong_shape = (4, 8, 10)  # 最后一维不匹配
    wrong_space = Space(shape=wrong_shape)
    
    with pytest.raises(AssertionError):
        SegmentationMask(arr, mapping={}, space=wrong_space)


def test_lazy_init_unified_behavior():
    """测试lazy_init的统一行为。"""
    shape = (6, 10, 14)
    space = Space(shape=shape)
    
    # 使用space创建
    mask1 = SegmentationMask.lazy_init(8, space=space)
    assert mask1.data.shape == shape
    assert mask1.space.shape == shape
    
    # 使用shape创建
    mask2 = SegmentationMask.lazy_init(16, shape=shape)
    assert mask2.data.shape == shape
    assert mask2.space.shape == shape


def test_maskfile_unified_io():
    """测试文件I/O的统一行为。"""
    zyx_shape = (6, 10, 14)
    space = Space(shape=zyx_shape)

    arr = np.random.randint(0, 3, size=zyx_shape, dtype=np.uint8)
    mask = SegmentationMask(arr, LabelMapping({"obj": 1}), space)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.msk")
        mf = MaskFile(path, mode="w")
        mf.write(mask)

        loaded = MaskFile(path).read()
        
        # 验证数据完全一致
        np.testing.assert_array_equal(loaded.data, mask.data)
        assert loaded.data.shape == zyx_shape
        assert loaded.space.shape == zyx_shape
        
        # 验证空间信息一致
        assert loaded.space.shape == mask.space.shape
        assert np.allclose(loaded.space.spacing, mask.space.spacing)
        assert np.allclose(loaded.space.origin, mask.space.origin)


def test_cross_language_compatibility():
    """测试跨语言兼容性设计。"""
    # 创建测试数据
    zyx_shape = (4, 6, 8)
    space = Space(shape=zyx_shape, spacing=(1.0, 2.0, 3.0), origin=(10, 20, 30))
    arr = np.random.randint(0, 4, size=zyx_shape, dtype=np.uint8)
    mask = SegmentationMask(arr, LabelMapping({"organ": 1}), space)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cross_lang.msk")
        
        # 保存
        mask.save(path)
        
        # 加载
        loaded = SegmentationMask.load(path)
        
        # Python用户视角：data(z,y,x) + space(z,y,x) → aligned
        assert loaded.data.shape == zyx_shape
        assert loaded.space.shape == zyx_shape
        np.testing.assert_array_equal(loaded.data, arr)
        
        # 验证空间信息正确转换
        assert loaded.space.spacing == space.spacing
        assert loaded.space.origin == space.origin
