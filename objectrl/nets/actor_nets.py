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

from typing import Any, Literal

import torch
from torch import nn as nn

from objectrl.nets.layers.heads import SquashedGaussianHead
from objectrl.utils.net_utils import MLP


class ActorNetProbabilistic(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_act: int,
        n_heads: int = 1,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
        upper_clamp: float = -2.0,
    ) -> None:
        """
        Probabilistic Actor Network that outputs a Gaussian distribution
        over actions using a squashed Gaussian head.

        Args:
        dim_state (int): Observation space dimension (assumed 1D tuple).
        dim_act (int): Action space dimension (assumed 1D tuple).
        n_heads (int): Number of policy heads (useful for ensemble methods).
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function to use.
        has_norm (bool): Whether to include normalization layers.
        upper_clamp (float): Upper clamp value for log-variance in Squashed Gaussian.
        """
        super().__init__()
        self.dim_act = dim_act
        self.n_heads = n_heads

        # Create the network architecture
        self.arch = MLP(dim_state, 2 * dim_act * n_heads, depth, width, act, has_norm)

        # Gaussian distribution head for action selection
        self.head = SquashedGaussianHead(self.dim_act, upper_clamp)

    def forward(self, x: torch.Tensor, is_training: bool = True) -> dict[str, Any]:
        """
        Forward pass to generate a squashed Gaussian action distribution.

        Args:
            x (Tensor): Input observation tensor.
            is_training (bool): Whether to sample actions stochastically.
        """
        f = self.arch(x)
        if self.n_heads > 1:
            f = f.view(-1, self.n_heads, 2 * self.dim_act)
        return self.head(f, is_training)


class ActorNet(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_act: int,
        n_heads: int = 1,
        depth: int = 3,
        width: int = 256,
        act: Literal["crelu", "relu"] = "relu",
        has_norm: bool = False,
    ) -> None:
        """
        Deterministic Actor Network that outputs continuous actions.

        Args:
        dim_state (int): Observation space dimension.
        dim_act (int): Action space dimension.
        n_heads (int): Number of parallel output heads.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function name.
        has_norm (bool): Whether to use normalization layers.
        """
        super().__init__()

        self.dim_act = dim_act
        self.n_heads = n_heads

        self.arch = nn.Sequential(
            MLP(dim_state, dim_act * n_heads, depth, width, act, has_norm),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, is_training: bool | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for deterministic action prediction.

        Args:
            x (Tensor): Input observation tensor.
            is_training (Optional[bool]): Unused; included for interface compatibility.
        """
        out = self.arch(x)
        if self.n_heads > 1:
            out = out.view(-1, self.n_heads, self.dim_act)
        return_dict = {
            "action": out,
        }
        return return_dict
