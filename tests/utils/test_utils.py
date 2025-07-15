# -----------------------------------------------------------------------------------
# ObjectRL: An Object-Oriented Reinforcement Learning Codebase
# Copyright (C) 2025 ADIN Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------------

import pytest
import torch
import numpy as np
import sys

from objectrl.utils.utils import totorch, tonumpy, toint, dim_check


def test_totorch():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
    t = totorch(arr, dtype=torch.float64, device="cpu")
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float64
    assert t.device.type == "cpu"
    np.testing.assert_array_equal(t.cpu().numpy(), arr)

    lst = [[5, 6], [7, 8]]
    t2 = totorch(lst, dtype=torch.int32, device="cpu")
    assert t2.dtype == torch.int32
    assert t2.device.type == "cpu"
    np.testing.assert_array_equal(t2.cpu().numpy(), np.array(lst))

    if torch.cuda.is_available():
        t3 = totorch([1, 2, 3])
        assert t3.device.type == "cuda"
    else:
        t3 = totorch([1, 2, 3], device="cpu")
        assert t3.device.type == "cpu"


def test_tonumpy():
    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    arr = tonumpy(t)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, t.numpy())

    if torch.cuda.is_available():
        t_gpu = t.to("cuda")
        arr_gpu = tonumpy(t_gpu)
        np.testing.assert_array_equal(arr_gpu, t.numpy())


def test_toint():
    t = torch.tensor(42)
    assert toint(t) == 42

    t1 = torch.tensor([7])
    assert toint(t1) == 7

    t2 = torch.tensor(3.0)
    assert toint(t2) == 3

    t_multi = torch.tensor([1, 2])
    with pytest.raises(Exception):
        toint(t_multi)


def test_dim_check():
    t1 = torch.randn(2, 3)
    t2 = torch.randn(2, 3)
    dim_check(t1, t2)

    t3 = torch.randn(3, 2)
    with pytest.raises(AssertionError) as excinfo:
        dim_check(t1, t3)
    assert "Shapes are" in str(excinfo.value)
