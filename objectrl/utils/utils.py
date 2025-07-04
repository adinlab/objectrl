# -----------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 ADIN Lab
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
