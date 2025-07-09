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

import importlib
from collections.abc import Callable
from typing import Literal

import torch
import torch.optim as optim
from torch import nn as nn

from objectrl.nets.layers.bayesian_layers import (
    BBBLinear,
    CLTLinear,
    CLTLinearDet,
    LRLinear,
)
from objectrl.utils.custom_act import CReLU


def create_optimizer(config) -> Callable:
    """
    Creates a PyTorch optimizer based on the configuration.

    Args:
        config: Configuration object containing:
            - config.optimizer (str): Name of the optimizer (e.g., 'Adam', 'SGD').
            - config.learning_rate (float): Learning rate for the optimizer.
    Returns:
        Callable: A function that accepts model parameters and returns an optimizer instance.
    Raises:
        NotImplementedError: If the optimizer name is not available in torch.optim.
    """
    optimizer_name = config.optimizer
    if hasattr(optim, optimizer_name):
        optimizer = getattr(optim, optimizer_name)
    else:
        raise NotImplementedError(f"{optimizer_name} is not found in torch.optim")

    return lambda params: optimizer(params, lr=config.learning_rate)


def create_loss(config, reduction: str = "none") -> nn.Module:
    """
    Creates a loss function module from either torch.nn or a custom module.

    Args:
        config: Configuration object containing:
            - config.loss (str): Name of the loss function.
        reduction (str, optional): Reduction method ('none', 'mean', or 'sum'). Defaults to "none".
    Returns:
        nn.Module: A PyTorch loss function module.
    Raises:
        NotImplementedError: If the loss is not found in torch.nn or the custom module.
    """
    loss_name = config.loss
    if hasattr(nn, loss_name):
        return getattr(nn, loss_name)(reduction=reduction)
    else:
        loss_module = importlib.import_module(
            "objectrl.models.basic.loss"
        )  # Import the loss.py module
        if hasattr(loss_module, loss_name):
            loss_class = getattr(loss_module, loss_name)
            return loss_class(config)
        else:
            raise NotImplementedError(
                f"{loss_name} is not found in torch.nn or in loss.py"
            )


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        width: int,
        act: str = "relu",
        has_norm: bool = False,
    ) -> None:
        """
        Constructs a fully connected Multi-Layer Perceptron (MLP).

        Args:
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            depth (int): Total number of layers (>= 1).
            width (int): Width of the hidden layers.
            act (str): Activation function. Options are:
                - "relu": Standard ReLU.
                - "crelu": Concatenated ReLU (doubles width).
            has_norm (bool): If True, applies LayerNorm between layers.
        Raises:
            AssertionError: If depth <= 0.
            NotImplementedError: If unknown activation function is specified.
        """
        super().__init__()
        assert depth > 0, "Need at least one layer"

        if act == "crelu":
            self.activation_fn = CReLU
            width_multiplier = 2
        elif act == "relu":
            self.activation_fn = nn.ReLU
            width_multiplier = 1
        else:
            raise NotImplementedError(
                f"{act} is not implemented. User should add other activation functions if needed."
            )

        effective_width = width * width_multiplier

        layers = []

        if depth == 1:
            layers.append(nn.Linear(dim_in, dim_out))
        else:
            layers.append(nn.Linear(dim_in, width))
            # Hidden layers
            for i in range(depth - 1):
                if has_norm:
                    layers.append(nn.LayerNorm(width, elementwise_affine=False))
                layers.append(self.activation_fn())
                # Last hidden layer connects to output
                if i == depth - 2:
                    layers.append(nn.Linear(effective_width, dim_out))
                else:
                    layers.append(nn.Linear(effective_width, width))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the standard MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim_out).
        """
        return self.model(x)


class BayesianMLP(nn.Module):
    def __init__(  # noqa: C901
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        width: int,
        layer_type: Literal["bbb", "lr", "clt", "cltdet"] = "lr",
        act: Literal["crelu", "relu"] = "relu",
        has_norm: bool = False,
    ) -> None:
        """
        Constructs a Bayesian MLP using probabilistic linear layers.
        Supports various types of Bayesian layers for uncertainty estimation.

        Args:
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            depth (int): Number of layers (>= 1).
            width (int): Width of the hidden layers.
            layer_type (str): Type of Bayesian linear layer. One of:
                - "bbb": Bayes by Backprop.
                - "lr": Local Reparameterization trick.
                - "clt": Central Limit Theorem (probabilistic forward).
                - "cltdet": CLT with deterministic weights.
            act (str): Activation function. One of "relu" or "crelu".
            has_norm (bool): Whether to apply LayerNorm. Not supported for CLT variants.
        Raises:
            AssertionError: If depth <= 0 or incompatible settings.
            NotImplementedError: For unknown layer or activation types.
        """
        super().__init__()
        assert depth > 0, "Need at least one layer"

        # Pick the layer type
        if "clt" in layer_type:
            assert act in [
                "crelu",
                "relu",
            ], "Deterministic uncertainty propagation is only available for 'relu' and 'crelu' activations"
            assert not has_norm, "Not available for CLT-based layers"

        # Identify the chosen layer
        match layer_type:
            case "bbb":
                bnn_layer = BBBLinear
                det_uncertainty = False
            case "lr":
                bnn_layer = LRLinear
                det_uncertainty = False
            case "clt":
                bnn_layer = CLTLinear
                det_uncertainty = True
            case "cltdet":
                bnn_layer = CLTLinearDet
                det_uncertainty = True
            case _:
                raise NotImplementedError(f"{layer_type} is not implemented")

        # Select activation and width multiplier
        if act == "crelu":
            activation_fn = CReLU
            width_multiplier = 2
        elif act == "relu":
            activation_fn = nn.ReLU
            width_multiplier = 1
        else:
            raise NotImplementedError(f"{act} is not implemented")

        effective_width = width * width_multiplier

        layers = []

        # Single-layer case needs to be handled differently
        if depth == 1:
            layers.append(bnn_layer(dim_in, dim_out))
        else:
            layers.append(bnn_layer(dim_in, width))

            for i in range(depth - 1):
                if has_norm and not det_uncertainty:
                    layers.append(nn.LayerNorm(width, elementwise_affine=False))
                if not det_uncertainty:
                    layers.append(activation_fn())
                if i == depth - 2:
                    layers.append(bnn_layer(effective_width, dim_out))
                else:
                    layers.append(bnn_layer(effective_width, width))

        self.model = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the Bayesian MLP.

        Args:
            x (Union[Tensor, Tuple[Tensor, Optional[Tensor]]]):
                - For standard use: input tensor.
                - For CLT-based: tuple of (mean, variance).
        Returns:
            Union[Tensor, Tuple[Tensor, Optional[Tensor]]]: Output in the same format as input.
        """
        # CLT-based layers can have (mean, variance) input
        return self.model(x)

    def get_kl(self) -> tuple[torch.Tensor, int]:
        """
        Get the KL divergence of the Bayesian MLP.
        """
        total_kl = 0.0
        n_params = 0
        for layer in self.model:
            if isinstance(layer, (BBBLinear, LRLinear, CLTLinear)):
                kl, n = layer.KL()
                total_kl += kl
                n_params += n
        return total_kl, n_params
