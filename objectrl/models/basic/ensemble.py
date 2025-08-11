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
from abc import ABC
from typing import Literal, Any

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
        models: list[T],
        device: Literal["cpu", "cuda"] = "cpu",
        sequential: bool = False,
        compile: bool = True,
    ) -> None:
        """
        Initialize the ensemble

        Args:
            n_members (int): The number of members in the ensemble
            models (list[nn.Module]): List of models to parallelize
            device (str): The device to use
            sequential (bool): Whether the ensemble is sequential (necessary for layers with states)
            compile (bool): Whether the ensemble is compiled or not
        Returns:
            None
        """
        super().__init__()

        self.n_members = n_members

        self.sequential = sequential
        self.device = device

        if sequential:
            self.models = nn.ModuleList(models)
            self.forward_model = lambda input: torch.stack(
                [net(input) for net in self.models]
            )
            self.params = self.models.state_dict()
            self.buffers = self.models.buffers()
        else:
            stacked_params, stacked_buffers = thf.stack_module_state(models)

            self.base_model = copy.deepcopy(models[0]).to("meta")
            self.prototype = copy.deepcopy(models[0])

            # Register storages once, but keep a mapping with original 'dotted' names
            # so functional_call sees the exact keys the base model expects.
            # IMPORTANT: We construct self.params/self.buffers so that their values
            # reference the registered tensors, avoiding duplicated state.
            params_map: dict[str, torch.Tensor] = {}
            buffers_map: dict[str, torch.Tensor] = {}

            for name, tensor in stacked_params.items():
                sanitized = name.replace(".", "_")
                p = nn.Parameter(tensor.to(device))
                self.register_parameter(f"stacked__{sanitized}", p)
                params_map[name] = p  # original key -> registered tensor

            for name, tensor in stacked_buffers.items():
                sanitized = name.replace(".", "_")
                # Register buffer and keep a direct reference to the registered storage
                self.register_buffer(f"stacked__{sanitized}", tensor.to(device))
                buffers_map[name] = getattr(self, f"stacked__{sanitized}")

            # These dicts are used by functional_call. They point to registered tensors.
            self.params: dict[str, torch.Tensor] = params_map
            self.buffers: dict[str, torch.Tensor] = buffers_map

            def _fmodel(
                base_model: nn.Module,
                params: dict[str, torch.Tensor],
                buffers: dict[str, torch.Tensor],
                x: torch.Tensor,
            ) -> torch.Tensor:
                return thf.functional_call(base_model, (params, buffers), (x,))

            vmapped = thf.vmap(
                lambda p, b, x: _fmodel(self.base_model, p, b, x),
                randomness="different",
            )

            if compile:
                self.forward_model = torch.compile(
                    vmapped, dynamic=True, mode="max-autotune"
                )
            else:
                self.forward_model = vmapped

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.sequential:
            return self.forward_model(input)
        else:
            return self.forward_model(self.params, self.buffers, self.expand(input))

    def expand(
        self, x: torch.Tensor | tuple | list, force: bool = False
    ) -> torch.Tensor | tuple | list:
        def _expand_single(x: Any) -> Any:
            if not isinstance(x, torch.Tensor):
                return x
            elif not force and x.ndim >= 1 and x.size(0) == self.n_members:
                return x
            else:
                return x.expand(self.n_members, *x.shape)

        if isinstance(x, torch.Tensor):
            return _expand_single(x)
        elif isinstance(x, tuple):
            return tuple(_expand_single(x) for x in x)
        elif isinstance(x, list):
            return [_expand_single(x) for x in x]
        raise TypeError(f"Expanding {type(x)} is not supported")

    @torch.no_grad()
    def _get_single_member(self, index: int = 0) -> T:
        """
        Extract a single member from the ensemble.

        Args:
            index: Index of the member to extract (default: 0)
        Returns:
            T: A single member of the ensemble with the specified index.
        """

        if not (0 <= index < self.n_members):
            raise IndexError(f"Index {index} is out of range ({self.n_members = })")

        if self.sequential:
            return self.models[index]

        # Create a new critic with the same configuration
        single_model = copy.deepcopy(self.prototype)

        # Extract parameters for the specified index
        for name, param in single_model.named_parameters():
            stacked_param = self.params[name]
            param.copy_(stacked_param[index])

        # Extract buffers (like batch norm stats) if any
        for name, buffer in single_model.named_buffers():
            stacked_buffer = self.buffers[name]
            buffer.copy_(stacked_buffer[index])

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
