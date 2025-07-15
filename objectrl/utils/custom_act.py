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
from torch import nn as nn
from torch.nn import functional as F


class CReLU(nn.Module):
    """
    Concatenated ReLU (CReLU) activation module.

    This module implements the CReLU activation function, which concatenates
    the ReLU activations of both the input and its negation along the last dimension:

        CReLU(x) = ReLU([x, -x])

    This increases the representational capacity of the model by doubling the
    number of features while preserving non-linearity.
    """

    def __init__(self) -> None:
        """
        Initialize the CReLU module.
        Args:
            None
        Returns:
            None
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CReLU activation.
        Args:
            x (torch.Tensor): Input tensor of shape (..., features)
        Returns:
            torch.Tensor: Output tensor with ReLU applied to both x and -x,
                          concatenated along the last dimension (shape: (..., 2 * features))
        """
        x = torch.cat((x, -x), -1)
        return F.relu(x)
