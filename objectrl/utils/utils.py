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

import numpy
import torch


def totorch(
    x: numpy.ndarray,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
) -> "torch.Tensor":
    """
    Converts a NumPy array or other compatible object to a PyTorch tensor.

    Args:
        x (array-like): Input data to convert.
        dtype (torch.dtype, optional): Desired data type of the output tensor. Default is torch.float32.
        device (str or torch.device, optional): Device to store the tensor on. Default is "cuda".
    Returns:
        torch.Tensor: A tensor containing the same data as `x`, on the specified device and with the specified dtype.
    """
    return torch.as_tensor(x, dtype=dtype, device=device)


def tonumpy(x: torch.Tensor) -> numpy.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.
    Automatically moves the tensor to CPU and detaches it from the computation graph.

    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        numpy.ndarray: NumPy array with the same data as the input tensor.
    """
    return x.data.cpu().numpy()


def toint(x: torch.Tensor) -> int:
    """Converts a PyTorch tensor to an integer.

    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        int: Integer value of the input tensor.
    """
    return int(x)


def dim_check(tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
    """
    Asserts that two tensors have the same shape.
    Useful for debugging shape mismatches in model inputs/outputs.

    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
    Raises:
        AssertionError: If the shapes of the two tensors do not match.
    """
    assert (
        tensor1.shape == tensor2.shape
    ), f"Shapes are {tensor1.shape} vs {tensor2.shape}"
