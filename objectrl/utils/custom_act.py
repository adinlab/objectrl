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
