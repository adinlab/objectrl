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

import torch

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.actor import Actor
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.utils.utils import totorch

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class TD3Actor(Actor):
    """
    TD3 actor network with action noise for exploration and target policy smoothing.

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): Observation space dimensions.
        dim_act (int): Action space dimensions.

    Attributes:
        policy_noise (float): Noise std for exploration.
        target_policy_noise (float): Noise std for target policy smoothing.
        target_policy_noise_clip (float): Clipping range for target noise.
        action_limit_low (Tensor): Lower bound for actions.
        action_limit_high (Tensor): Upper bound for actions.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)
        noise = config.model.noise
        self.policy_noise = noise.policy_noise
        self.target_policy_noise = noise.target_policy_noise
        self.target_policy_noise_clip = noise.target_policy_noise_clip

        self.action_limit_low = totorch(
            config.env.env.action_space.low, device=self.device  # type:ignore
        )
        self.action_limit_high = totorch(
            config.env.env.action_space.high, device=self.device  # type:ignore
        )

    def act(self, state: torch.Tensor, is_training: bool = True) -> dict:
        """
        Computes actions with optional noise added for exploration.

        Args:
            state (Tensor): Batch of states.
            is_training (bool): Whether in training mode (adds noise if True).
        Returns:
            dict: Contains 'action' tensor and 'action_wo_noise' tensor.
        """
        action_dict = super().act(state)
        action = action_dict["action"]
        action_dict["action_wo_noise"] = action
        if is_training:
            noise = torch.normal(0, self.policy_noise, action.shape).to(self.device)
            action += noise
            action = torch.clip(action, self.action_limit_low, self.action_limit_high)
            action_dict["action"] = action
        return action_dict

    def act_target(self, state: torch.Tensor) -> dict:
        """
        Computes target policy action with smoothing noise added.

        Args:
            state (Tensor): Batch of next states.
        Returns:
            dict: Contains 'action' tensor with added clipped noise.
        """
        action_dict = super().act(state)
        action = action_dict["action"]
        noise = torch.normal(0, self.target_policy_noise, action.shape).to(self.device)
        noise = torch.clip(
            noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
        )
        action += noise
        action = torch.clip(action, self.action_limit_low, self.action_limit_high)
        action_dict["action"] = action
        return action_dict

    def loss(self, state: torch.Tensor, critics: CriticEnsemble) -> torch.Tensor:
        """
        Computes actor loss as negative Q-value estimate.

        Args:
            state (Tensor): Batch of states.
            critics (CriticEnsemble): Critic networks.
        Returns:
            Tensor: Actor loss to maximize Q-values.
        """
        act_dict = self.act(state, is_training=False)
        action = act_dict["action"]
        q = critics[0].Q(state, action)
        return (-q).mean()


class TD3Critic(CriticEnsemble):
    """
    TD3 critic ensemble handling Bellman target computation and training loss.

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): Observation space dimensions.
        dim_act (int): Action space dimensions.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)

    @torch.no_grad()
    def get_bellman_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        actor: TD3Actor,
    ) -> torch.Tensor:
        """
        Computes target Q-values using Bellman backup.

        Args:
            reward (Tensor): Rewards batch.
            next_state (Tensor): Next state batch.
            done (Tensor): Done flags batch.
            actor (TD3Actor): Target policy actor network.
        Returns:
            Tensor: Bellman target Q-values.
        """
        next_action_dict = actor.act_target(next_state)
        next_action = next_action_dict["action"]
        target_values = self.Q_t(next_state, next_action)
        target_value = self.reduce(
            target_values, reduce_type=self.config.model.critic.target_reduce
        )
        y = reward.unsqueeze(-1) + self._gamma * target_value * (1 - done.unsqueeze(-1))
        return y


class TwinDelayedDeepDeterministicPolicyGradient(ActorCritic):
    """
    TD3 agent combining delayed policy updates and clipped noise target smoothing.
    """

    _agent_name = "TD3"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = TD3Critic,
        actor_type: type = TD3Actor,
    ) -> None:
        """
        Initializes the TD3 agent.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type): Critic network class type.
            actor_type (type): Actor network class type.
        Returns:
            None
        """
        super().__init__(config, critic_type, actor_type)
