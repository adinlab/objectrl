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

from typing import Literal

import torch
from torch import nn as nn

from objectrl.utils.net_utils import MLP, BayesianMLP


class CriticNet(nn.Module):
    """
    Deterministic Critic Network (Q-network).
    Estimates the expected return (Q-value) for a given state-action pair.

    Args:
        dim_state (int): Dimension of observation space.
        dim_act (int): Dimension of action space.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function to use.
        has_norm (bool): Whether to include normalization layers.
    """

    def __init__(
        self,
        dim_state: int,
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
    ) -> None:
        super().__init__()

        self.arch = MLP(
            dim_state + dim_act, 1, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Args:
            x (Tensor): Concatenated observation and action tensor.
        """
        return self.arch(x)


class ValueNet(CriticNet):
    """
    Value network for estimating V(s) without action input.

    Inherits from CriticNet but ignores action dimensions by setting dim_act to 0.
    Suitable for use in value-based methods like PPO or baseline estimation.

    Args:
        dim_state (int): Dimension of the input state.
        dim_act (int): Unused, kept for compatibility (should be 0).
        depth (int): Number of hidden layers in the network.
        width (int): Width (number of units) in each hidden layer.
        act (str): Activation function to use ("relu" or "crelu").
        has_norm (bool): Whether to apply normalization (e.g., LayerNorm).
    """

    def __init__(
        self,
        dim_state: int,
        dim_act: int,  # kept for interface compatibility
        depth: int = 3,
        width: int = 256,
        act: str = "relu",
        has_norm: bool = False,
    ) -> None:
        super().__init__(dim_state, 0, depth, width, act, has_norm)


class CriticNetProbabilistic(nn.Module):
    """
    Probabilistic Critic Network.

    Args:
        dim_state (int): Observation space dimension.
        dim_act (int): Action space dimension.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function to use.
        has_norm (bool): Whether to use normalization layers.
    """

    def __init__(
        self,
        dim_state: int,
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
    ) -> None:
        super().__init__()

        self.arch = MLP(
            dim_state + dim_act, 2, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the probabilistic critic.

        Args:
           x  (Tensor): Concatenated observation and action tensor.
        """
        return self.arch(x)


class BNNCriticNet(nn.Module):
    """
    A Bayesian Critic Network (Q-network).

    Args:
        dim_state (int): Observation space dimension.
        dim_act (int): Action space dimension.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (Literal["relu", "crelu"]): Activation function to use.
        has_norm (bool): Whether to include normalization layers.
    """

    def __init__(
        self,
        dim_state: int,
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
    ) -> None:
        super().__init__()

        # A BNN with local-reparameterization layers
        self.arch = BayesianMLP(
            dim_in=dim_state + dim_act,
            dim_out=1,
            depth=depth,
            width=width,
            act=act,
            has_norm=has_norm,
            layer_type="lr",
        )

        self._map = False

    def map(self, on: bool = True) -> None:
        "Switch maximum a posteriori mode on/off"
        self._map = on
        for layer in self.arch:
            if hasattr(layer, "_map"):
                layer.map(on)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        "Forward pass of the BNNCriticNet"

        return self.arch(x)


class EMstyle(nn.Module):
    """
    Encoder network for EM-style models.

    Args:
        dim_state (int): Observation space dimension.
        dim_act (int): Action space dimension.
        depth (int): Number of hidden layers.
        width (int): Hidden layer width and output dimensionality.
        act (str): Activation function to use.
        has_norm (bool): Whether to use normalization layers.
    """

    def __init__(
        self,
        dim_state: int,
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
    ) -> None:
        super().__init__()
        self.arch = MLP(
            dim_state + dim_act, width, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce latent feature encoding.

        Args:
            x (Tensor): Concatenated input tensor.
        """
        x = self.arch(x)
        return x


class DQNNet(nn.Module):
    """
    Deterministic Critic Network (Q-network).

    Args:
        dim_state (int): Dimension of observation space.
        dim_act (int): Dimension of action space.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function to use.
        has_norm (bool): Whether to include normalization layers.
    """

    def __init__(
        self,
        dim_state: int,
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: str = "relu",
        has_norm: bool = False,
    ) -> None:
        super().__init__()

        self.arch = MLP(dim_state, dim_act, depth, width, act=act, has_norm=has_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Args:
            x (Tensor): Concatenated observation and action tensor.
        """
        return self.arch(x)
