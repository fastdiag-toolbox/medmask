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


@pytest.mark.parametrize("axis_reversed", [False, True])
def test_axis_alignment(axis_reversed):
    """验证 data 与 data_aligned 的轴顺序关系以及形状校验。"""
    # Space 以 XYZ 顺序给 shape
    xyz_shape = (8, 16, 24)
    space = Space(shape=xyz_shape)

    # 根据 axis_reversed 构造数组形状
    shape = tuple(reversed(xyz_shape)) if axis_reversed else xyz_shape
    arr = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

    mask = SegmentationMask(arr, mapping=LabelMapping({"bg": 0}), space=space, axis_reversed=axis_reversed)

    # 1. 属性保持
    assert mask.axis_reversed is axis_reversed

    # 2. data_native 与构造数组一致
    np.testing.assert_array_equal(mask.data, arr)

    # 3. data_aligned 的形状应为 XYZ
    assert mask.data_aligned.shape == xyz_shape

    # 4. 若未翻转，aligned 就是 native；否则 aligned == native.transpose(reversed(range(ndim)))
    if axis_reversed:
        expected = arr.transpose(tuple(reversed(range(arr.ndim))))
        np.testing.assert_array_equal(mask.data_aligned, expected)
    else:
        np.testing.assert_array_equal(mask.data_aligned, mask.data)


@pytest.mark.parametrize("axis_reversed", [False, True])
def test_maskfile_axis_flag_persistence(axis_reversed):
    """写入 .msk 后读取，确保 axis_reversed 得到正确恢复。"""
    xyz_shape = (6, 10, 14)
    space = Space(shape=xyz_shape)
    shape = tuple(reversed(xyz_shape)) if axis_reversed else xyz_shape

    arr = np.random.randint(0, 3, size=shape, dtype=np.uint8)
    mask = SegmentationMask(arr, LabelMapping({"obj": 1}), space, axis_reversed=axis_reversed)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.msk")
        mf = MaskFile(path, mode="w")
        mf.write(mask)

        loaded = MaskFile(path).read()
        assert loaded.axis_reversed is axis_reversed
        np.testing.assert_array_equal(loaded.data, mask.data)
        assert loaded.space == space
