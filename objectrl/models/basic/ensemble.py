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

import copy
import functools
from abc import ABC
from typing import Literal

import torch
from torch import func as thf
from torch import nn as nn


class Ensemble[T: nn.Module](nn.Module, ABC):
    """
    A generic ensemble of neural networks
    This class allows for parallelizing the forward pass of multiple models
    while maintaining a consistent interface.

    Attributes:
        n_members (int): Number of members in the ensemble.
        prototype (nn.Module): Prototype model used to create new members.
        device (str): Device type for the ensemble (e.g., "cpu", "cuda").
        params (dict[str, torch.Tensor]): Stacked parameters of the ensemble members.
        buffers (dict[str, torch.Tensor]): Stacked buffers of the ensemble members.
        base_model (nn.Module): Base model structure for functional calls.
        forward_model (torch.nn.functional): Vectorized function to call the model.
    """

    def __init__(
        self,
        n_members: int,
        prototype: T,
        models: list[T],
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        """
        Initialize the ensemble

        Args:
            n_members (int): The number of members in the ensemble
            prototype (nn.Module): The prototype of the ensemble
            models (list[nn.Module]): List of models to parallelize
            device (str): The device to use
        Returns:
            None
        """
        super().__init__()

        self.n_members = n_members
        self.prototype = prototype

        self.device = device

        self.params, self.buffers = thf.stack_module_state(models)

        self.base_model = copy.deepcopy(models[0]).to("meta")

        # Register the parameters and buffer in the ensemble's state dict
        # Replace '.' with '_' as dots are not allowed
        for name, param in self.params.items():
            self.register_parameter(
                f"stacked-{name.replace(".", "_")}", nn.Parameter(param)
            )
        for name, buffer in self.buffers.items():
            self.register_buffer(f"stacked-{name.replace(".", "_")}", buffer)

        def _fmodel(
            base_model: nn.Module,
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            x: torch.Tensor,
        ) -> torch.Tensor:
            """
            Function to call a model with given parameters and buffers.

            Args:
                base_model (nn.Module): The base model to call.
                params (dict[str, torch.Tensor]): Parameters of the model.
                buffers (dict[str, torch.Tensor]): Buffers of the model.
                x (torch.Tensor): Input tensor to the model.
            Returns:
                torch.Tensor: Output tensor from the model.
            """
            return thf.functional_call(base_model, (params, buffers), (x,))

        self.forward_model = thf.vmap(
            functools.partial(_fmodel, self.base_model), randomness="different"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_model(self.params, self.buffers, self.expand(input))

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(self.n_members, *x.shape) if len(x.shape) < 3 else x

    def _get_single_member(self, index: int = 0) -> T:
        """
        Extract a single member from the ensemble.

        Args:
            index: Index of the member to extract (default: 0)
        Returns:
            T: A single member of the ensemble with the specified index.
        """
        # Create a new critic with the same configuration
        single_model = copy.deepcopy(self.prototype)

        # Extract parameters for the specified index
        for name, param in single_model.named_parameters():
            # Get the corresponding parameter from the stacked params
            stacked_param = self.params[name]
            # Extract the parameters for the specified member
            param.data.copy_(stacked_param[index])

        # Extract buffers (like batch norm stats) if any
        for name, buffer in single_model.named_buffers():
            stacked_buffer = self.buffers[name]
            buffer.data.copy_(stacked_buffer[index])

        return single_model

    def _get_all_members(self) -> nn.ModuleList:
        """
        Extract all members from the ensemble.
        """
        return nn.ModuleList(
            [self._get_single_member(i) for i in range(self.n_members)]
        )

    def __getitem__(self, index: int) -> T:
        """
        Get a single member of the ensemble by index

        Args:
            index: Index of the member to extract (default: 0)
        Returns:
            T: A single member of the ensemble with the specified index.
        """
        return self._get_single_member(index)
