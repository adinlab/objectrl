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

import typing
from abc import ABC, abstractmethod

import torch
from torch import nn as nn

from objectrl.models.basic.critic import CriticEnsemble
from objectrl.utils.net_utils import create_optimizer

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class Actor(nn.Module, ABC):
    """
    Abstract base class for Actor network in Actor-Critic algorithms.

    Handles policy network, optional target network, and optimization.

    Attributes:
        config (MainConfig): Configuration object.
        device (torch.device): Device for tensor computations.
        verbose (bool): Verbosity flag.
        has_target (bool): Flag for using a target network.
        iter (int): Training iteration counter.
        dim_state (int): Observation space shape.
        dim_act (int): Action space shape.
        _tau (float): Polyak averaging coefficient for target updates.
        _gamma (float): Discount factor for returns.
        _reset (bool): Flag whether to reset model at initialization.
        model (nn.Module): Main actor network.
        target (nn.Module, optional): Target actor network.
        optim (torch.optim.Optimizer): Optimizer for the actor parameters.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        """
        Initializes the Actor.

        Args:
            config (MainConfig): Configuration dataclass instance.
            dim_state (int): Dimension of observation space.
            dim_act (int): Dimension of action space.
        Returns:
            None
        """
        super().__init__()
        self.config = config
        self.device = config.system.device
        self.verbose = config.verbose
        self.has_target = config.model.actor.has_target
        self.iter = 0
        self.dim_state, self.dim_act = dim_state, dim_act
        self._tau = config.model.tau
        self._gamma = config.training.gamma
        self._reset = config.model.actor.reset

        self.reset()

    def reset(self) -> None:
        """
        Initializes or resets the main and target policy networks and optimizer.
        Also sets the model architecture based on the configuration.

        Args:
            None
        Returns:
            None
        """
        self.model = self.config.model.actor.arch(
            self.dim_state,
            self.dim_act,
            depth=self.config.model.actor.depth,
            width=self.config.model.actor.width,
            act=self.config.model.actor.activation,
            has_norm=self.config.model.actor.norm,
            n_heads=self.config.model.actor.n_heads,
        ).to(self.device)

        self.optim = create_optimizer(self.config.training)(self.model.parameters())

        # Initialize target network if required
        if self.has_target:
            self.target = self.config.model.actor.arch(
                self.dim_state,
                self.dim_act,
                depth=self.config.model.actor.depth,
                width=self.config.model.actor.width,
                act=self.config.model.actor.activation,
                has_norm=self.config.model.actor.norm,
                n_heads=self.config.model.actor.n_heads,
            ).to(self.device)
            self.init_target()

    def init_target(self) -> None:
        """
        Copies the main model parameters to the target network.

        Args:
            None
        Returns:
            None
        """
        assert self.has_target, "There is no target network to initialize"
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters(), strict=True
        ):
            target_param.data.copy_(local_param.data)

    def act(self, state: torch.Tensor, is_training: bool = True) -> dict:
        """
        Computes actions given input states.

        Args:
            state (torch.Tensor): Input state tensor.
            is_training (bool): Whether in training mode.
        Returns:
            dict: Dictionary containing action tensor and optionally log probabilities.
        """
        return_dict = self.model(state, is_training=is_training)

        return return_dict

    def act_target(self, state: torch.Tensor) -> dict:
        """
        Computes actions using the target policy network.

        Args:
            state (torch.Tensor): Input state tensor.
        Returns:
            dict: Dictionary containing action tensor and log probabilities.
        """
        assert self.has_target, "There is no target network to evaluate"
        return self.target(state)

    @torch.no_grad()
    def update_target(self) -> None:
        """
        Performs a soft update of the target network using Polyak averaging.

        Args:
            None
        Returns:
            None
        """
        if not self.has_target:
            return None
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters(), strict=True
        ):
            # Combine x = (1 - tau) * x + tau * y into a single inplace operation
            target_param.data.lerp_(local_param.data, self._tau)

    @abstractmethod
    def loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Abstract method to compute the loss for the actor.
        Should be overridden in subclasses.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            torch.Tensor: Computed loss tensor.
        """
        pass

    def update(self, state: torch.Tensor, critics: CriticEnsemble) -> None:
        """
        Performs a gradient update on the actor network.

        Args:
            state (Tensor): Input state batch.
            critics (object): Critic networks for computing Q-values.
        Returns:
            None
        """
        self.optim.zero_grad()
        loss = self.loss(state, critics)
        loss.backward()
        if self.config.model.actor.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.model.actor.max_grad_norm,
            )
        self.optim.step()

        self.iter += 1  # Increment iteration counter
