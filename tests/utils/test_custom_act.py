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
