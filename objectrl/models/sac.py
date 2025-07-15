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

import math
import typing

import torch

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.actor import Actor
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.utils.net_utils import create_optimizer

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class SACActor(Actor):
    """
    Soft Actor network with automatic temperature tuning.

    Args:
        config (MainConfig): Configuration object with hyperparameters.
        dim_state (int): Observation space dimensions.
        dim_act (int): Action space dimensions.
    Attributes:
        target_entropy (float): Target entropy for temperature tuning.
        log_alpha (Tensor): Learnable log temperature parameter.
        optim_alpha (Optimizer): Optimizer for temperature parameter.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)
        self.target_entropy = (
            -dim_act
            if config.model.target_entropy is None
            else config.model.target_entropy
        )
        self.log_alpha = torch.tensor(
            math.log(config.model.alpha),
            requires_grad=True,
            device=self.device,
        )
        self.optim_alpha = create_optimizer(config.training)([self.log_alpha])

    def update_alpha(self, act_dict: dict) -> None:
        """
        Updates the temperature parameter alpha based on current policy entropy.

        Args:
            act_dict (dict): Dictionary with keys 'action_logprob' containing log probabilities.
        Returns:
            None
        """
        log_prob = act_dict["action_logprob"]
        loss = -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        self.optim_alpha.zero_grad()
        loss.mean().backward()
        self.optim_alpha.step()

    def loss(
        self, state: torch.Tensor, critics: CriticEnsemble
    ) -> tuple[torch.Tensor, dict]:
        """
        Computes the SAC actor loss.

        Args:
            state (Tensor): Batch of states.
            critics (CriticEnsemble): Critic networks for Q-value estimation.

        Returns:
            tuple: Actor loss and action dictionary containing action and log probability.
        """
        act_dict = self.act(state)
        action, log_prob = act_dict["action"], act_dict["action_logprob"]
        q_values = critics.Q(state, action)
        q = critics.reduce(q_values, reduce_type=self.config.model.critic.reduce)
        loss = (-q + self.log_alpha.exp() * log_prob).mean()
        return loss, act_dict

    def update(self, state: torch.Tensor, critics: CriticEnsemble) -> None:
        """
        Performs a gradient step on the actor network and updates alpha.

        Args:
            state (Tensor): Batch of states.
            critics (CriticEnsemble): Critic ensemble for Q-value estimates.

        Returns:
            None
        """
        self.optim.zero_grad()
        loss, act_dict = self.loss(state, critics)
        loss.backward()
        self.optim.step()
        self.update_alpha(act_dict)

        self.iter += 1  # Increment iteration counter


class SACCritic(CriticEnsemble):
    """
    SAC critic ensemble handling Bellman target computation and updates.

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): State space dimensions.
        dim_act (int): Action space dimensions.
    Attributes:
        _gamma (float): Discount factor for future rewards.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)

    @torch.no_grad()
    def get_bellman_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        actor: SACActor,
    ) -> torch.Tensor:
        """
        Computes target Q-values using entropy-regularized Bellman backup.

        Args:
            reward (Tensor): Reward batch.
            next_state (Tensor): Next state batch.
            done (Tensor): Done flags batch.
            actor (SACActor): Actor network for next action sampling.

        Returns:
            Tensor: Target Q-values for critic training.
        """
        alpha = actor.log_alpha.exp().detach()
        act_dict = actor.act(next_state)
        next_action = act_dict["action"]
        action_logprob = act_dict["action_logprob"]
        target_values = self.Q_t(next_state, next_action)
        target_value = (
            self.reduce(
                target_values, reduce_type=self.config.model.critic.target_reduce
            )
            - alpha * action_logprob
        )
        y = reward.unsqueeze(-1) + (
            self._gamma * target_value * (1 - done.unsqueeze(-1))
        )
        return y


class SoftActorCritic(ActorCritic):
    """
    Soft Actor-Critic agent combining SACActor and SACCritic.

    Args:
        config (MainConfig): Configuration object containing model hyperparameters.
        critic_type (type): Type of critic to use, defaults to SACCritic.
        actor_type (type): Type of actor to use, defaults to SACActor.
    Attributes:
        config (MainConfig): Configuration object.
        critic (SACCritic): Critic network for Q-value estimation.
        actor (SACActor): Actor network for action selection.
        device (torch.device): Device for computation (CPU or GPU).
        iter (int): Iteration counter for training steps.
        optim (torch.optim.Optimizer): Optimizer for the actor network.
    """

    _agent_name = "SAC"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = SACCritic,
        actor_type: type = SACActor,
    ) -> None:
        """
        Initializes SAC agent.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type): Critic class type.
            actor_type (type): Actor class type.
        Returns:
            None
        """
        super().__init__(config, critic_type, actor_type)
