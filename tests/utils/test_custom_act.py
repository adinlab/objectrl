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

import torch
from objectrl.utils.custom_act import CReLU  # Replace with the actual import path


def test_crelu_output_shape_and_values():
    crelu = CReLU()

    # Test input tensor of shape (batch_size, features)
    x = torch.tensor([[1.0, -2.0, 0.0], [-0.5, 3.0, -4.0]])
    output = crelu(x)

    # Expected shape doubles last dimension
    assert output.shape == (2, 6)

    # Output should be relu of concatenation of x and -x
    expected = torch.relu(torch.cat((x, -x), dim=-1))
    assert torch.allclose(output, expected)


def test_crelu_negative_and_positive_inputs():
    crelu = CReLU()

    # Test with negative and positive values
    x = torch.tensor([-1.0, 0.0, 2.0])
    output = crelu(x)

    expected = torch.relu(torch.cat((x, -x), dim=-1))
    assert torch.allclose(output, expected)


def test_crelu_high_dim_input():
    crelu = CReLU()

    # Test input with more dimensions (e.g., 3D tensor)
    x = torch.randn(4, 5, 3)
    output = crelu(x)
    assert output.shape == (4, 5, 6)
    expected = torch.relu(torch.cat((x, -x), dim=-1))
    assert torch.allclose(output, expected)


def test_crelu_gradients():
    crelu = CReLU()
    x = torch.randn(2, 3, requires_grad=True)
    output = crelu(x)
    output.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
