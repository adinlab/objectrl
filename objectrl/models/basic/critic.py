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
import typing
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn as nn

from objectrl.models.basic.ensemble import Ensemble
from objectrl.utils.net_utils import create_loss, create_optimizer

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class Critic(nn.Module):
    """
    A critic network that estimates Q-values for state-action pairs.

    Attributes:
        device (torch.device): Device for computations.
        has_target (bool): Flag for presence of target network.
        _tau (float): Polyak averaging factor for target update.
        _gamma (float): Discount factor for rewards.
        model (nn.Module): Main critic network.
        target (nn.Module, optional): Target critic network.
    """

    model: nn.Module
    target: nn.Module | None

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        """
        Initialize the critic network.

        Args:
            config: Configuration object containing model parameters
            dim_state: Dimension of the state space
            dim_act: Dimension of the action space
        Returns:
            None
        """
        super().__init__()

        critic = config.model.critic
        self.device = config.system.device
        self.has_target = critic.has_target

        self._tau = config.model.tau
        self._gamma = config.training.gamma

        self.model = critic.arch(
            dim_state,
            dim_act,
            depth=critic.depth,
            width=critic.width,
            act=critic.activation,
            has_norm=critic.norm,
        ).to(self.device)

        if self.has_target:
            self.target = critic.arch(
                dim_state,
                dim_act,
                depth=critic.depth,
                width=critic.width,
                act=critic.activation,
                has_norm=critic.norm,
            ).to(self.device)
            self.init_target()

    def reduce(self, q_val: torch.Tensor) -> torch.Tensor:
        """
        Reduce Q-values if needed.

        Args:
            q_val (torch.Tensor): Q-values tensor
        Returns:
            torch.Tensor: Reduced Q-values.
        """
        return q_val

    def Q(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values for given state-action pairs.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Q-values for the state-action pairs.
        """
        return self.model(self._prepare_input(state, action))

    @staticmethod
    def _prepare_input(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Concatenate state and action tensors for critic input.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Prepared input tensor for the critic.
        """
        if action.shape == ():
            action = action.view(1, 1)
        return torch.cat((state, action), -1)

    def init_target(self) -> None:
        """
        Initialize target network with weights from the main network.

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

    def update_target(self) -> None:
        """
        Update target network parameters using soft update.

        Args:
            None
        Returns:
            None
        """
        assert self.has_target, "There is no target network to update"
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters(), strict=True
        ):
            target_param.data.mul_(1.0 - self._tau)
            target_param.data.add_(self._tau * local_param.data)

    def Q_t(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values using the target network.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Target Q-values.
        """
        assert self.has_target, "There is no target network to evaluate"
        return self.target(self._prepare_input(state, action))

    def __getitem__(self) -> "Critic":
        """
        Return self when indexed.

        Args:
            None
        Returns:
            Critic: Self instance.
        """
        return self


class CriticEnsemble(nn.Module, ABC):
    """
    Ensemble of critic networks for robust Q-value estimation.

    Attributes:
        n_members (int): Number of critics in the ensemble.
        config (MainConfig): Configuration object.
        dim_state (int): Dimension of the state space.
        dim_act (int): Dimension of the action space.
        has_target (bool): Flag for target networks.
        _reset (bool): Reset flag from config.
        device (torch.device): Device for computations.
        loss (callable): Loss function.
        _tau (float): Polyak averaging factor.
        _gamma (float): Discount factor.
        model_ensemble (Ensemble[Critic]): Ensemble of critic models.
        optim (torch.optim.Optimizer): Optimizer for ensemble parameters.
        target_ensemble (Ensemble[Critic], optional): Target ensemble.
        iter (int): Training iteration counter.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        """
        Initialize the critic ensemble.

        Args:
            config (MainConfig): Configuration object with model parameters.
            dim_state (int): Dimension of the state space.
            dim_act (int): Dimension of the action space.
        Returns:
            None
        """

        super().__init__()

        self.n_members = config.model.critic.n_members
        self.config = config
        self.dim_state = dim_state
        self.dim_act = dim_act
        self.has_target = config.model.critic.has_target
        self._reset = config.model.critic.reset
        self.device = config.system.device
        self.loss = create_loss(config.model, reduction="none")
        self.dim_state = dim_state
        self.dim_act = dim_act
        self._tau = config.model.tau
        self._gamma = config.training.gamma

        self.model_ensemble = Ensemble[nn.Module](
            n_members=int(self.n_members),
            models=[
                Critic(config, dim_state, dim_act).model for _ in range(self.n_members)
            ],
            device=self.device,
        )

        self.optim = create_optimizer(self.config.training)(
            self.model_ensemble.parameters()
        )

        if self.has_target:
            self.target_ensemble = Ensemble[nn.Module](
                n_members=int(self.n_members),
                models=[
                    Critic(config, dim_state, dim_act).target
                    for _ in range(self.n_members)
                ],
                device=self.device,
            )
            self.target_ensemble.load_state_dict(self.model_ensemble.state_dict())

        self.iter = 0

    def reset(self) -> None:
        """
        Reset the ensemble models and optimizer.

        Args:
            None
        Returns:
            None
        """
        self.model_ensemble = Ensemble[nn.Module](
            n_members=int(self.n_members),
            models=[
                Critic(self.config, self.dim_state, self.dim_act).model
                for _ in range(self.n_members)
            ],
            device=self.device,
        )

        if self.has_target:
            self.target_ensemble = copy.deepcopy(self.model_ensemble)

        self.optim = create_optimizer(self.config.training)(
            self.model_ensemble.parameters()
        )

    def reduce(self, q_val: torch.Tensor, reduce_type: str) -> torch.Tensor:
        """
        Reduce Q-values from multiple critics according to the configured method.
        Currently supports 'min' or 'mean'. User should add more methods if needed.

        Args:
            q_val (torch.Tensor): Q-values tensor from all critics.
            reduce_type (str): How to reduce the Q-values.
        Returns:
            torch.Tensor: Reduced Q-values.
        """
        if reduce_type == "min":
            return q_val.min(0).values
        elif reduce_type == "mean":
            return q_val.mean(0)
        else:
            raise ValueError(
                f"Unknown reduction method {self.config.model.critic.reduce}. Implement it if needed."
            )

    def _get_single_critic(self, index: int = 0) -> Critic:
        """
        Get a single Critic instance from the ensemble.

        Args:
            index (int): Index of the critic to retrieve.
        Returns:
            Critic: Critic instance.
        """
        single_critic = Critic(self.config, self.dim_state, self.dim_act)

        single_critic.model.load_state_dict(self.model_ensemble[index].state_dict())
        if self.has_target:
            single_critic.target.load_state_dict(  # type: ignore // we know that there is a state_dict
                self.target_ensemble[index].state_dict()
            )
        return single_critic

    def __getitem__(self, index: int) -> Critic:
        """
        Indexing to access a single critic from the ensemble.

        Args:
            index (int): Index of the critic.
        Returns:
            Critic: Critic instance.
        """
        return self._get_single_critic(index)

    def Q(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values for given state-action pairs using all critics.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Q-values from all critics.
        """
        if action is None:
            sa = state
        else:
            sa = torch.cat((state, action), -1)

        return self.model_ensemble(sa)

    @torch.no_grad()
    def Q_t(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values using the target networks.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Target Q-values from all critics.
        """
        assert self.has_target, "There is no target network to evaluate"
        if action is None:
            sa = state
        else:
            sa = torch.cat((state, action), -1)
        return self.target_ensemble(sa)

    def update(
        self, state: torch.Tensor, action: torch.Tensor, y: torch.Tensor
    ) -> None:
        """
        Update critic networks using the provided Bellman targets.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
            y (torch.Tensor): Bellman target values.
        Returns:
            None
        """
        self.optim.zero_grad()
        loss = self.loss(self.Q(state, action), self.model_ensemble.expand(y))
        # Sum over the ensemble members and average over the batches
        loss = loss.sum(0).mean() if self.n_members > 1 else loss.mean()
        loss.backward()
        self.optim.step()
        self.iter += 1

    @torch.no_grad()
    def update_target(self) -> None:
        """
        Update target network parameters using soft update.

        Args:
            None
        Returns:
            None
        """
        assert self.has_target, "There is no target network to update"
        for key in self.model_ensemble.params.keys():
            self.target_ensemble.params[key].data.mul_(1.0 - self._tau)
            self.target_ensemble.params[key].data.add_(
                self._tau * self.model_ensemble.params[key].data
            )

    @abstractmethod
    @torch.no_grad()
    def get_bellman_target(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Calculate the Bellman target for training the critic.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Bellman target tensor.
        """
        pass
