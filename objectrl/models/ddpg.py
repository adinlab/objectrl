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
from objectrl.utils.utils import totorch

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class OrnsteinUhlenbeckNoise:
    """
    Implements Ornstein-Uhlenbeck process to generate temporally correlated noise.
    Commonly used in DDPG to add exploration noise to continuous actions.

    Args:
        dim_act (int): Shape of the action space.
        mu (float): Long-running mean.
        theta (float): Speed of mean reversion.
        sigma (float): Volatility (standard deviation).
        dt (float): Time step size.
        x0 (float, optional): Initial value of the process.
    """

    def __init__(
        self,
        dim_act: int,
        mu: float = 0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: float | None = None,
    ) -> None:
        self.dim_act = dim_act
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = mu
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self) -> None:
        """Resets internal state to initial value or mean (mu)."""
        self.state = self.x0 if self.x0 is not None else self.mu

    def evolve_state(self) -> torch.Tensor:
        """Applies the OU process to evolve the noise state."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * math.sqrt(
            self.dt
        ) * torch.randn(self.dim_act)
        self.state = x + dx
        return self.state

    def sample(self) -> torch.Tensor:
        """
        Returns a sample from the current noise state.

        Args:
            None
        Returns:
            torch.Tensor: A sample tensor generated from the noise process.
        """
        return self.evolve_state()

    def __call__(self) -> torch.Tensor:
        """
        Enables the object to be called directly like a function.

        Args:
            None
        Returns:
            torch.Tensor: A sample tensor generated from the noise process.
        """
        return self.sample()


class DDPGActor(Actor):
    """
    DDPG actor module that produces actions and applies OU noise during training.

    Attributes:
        interaction_iter (int): Counter for interaction steps.
        sampling_rate (int): Frequency of head sampling for posterior sampling.
        idx_active_critic (int): Index of the currently active critic head.
        is_episode_end (bool): Flag indicating if the current episode has ended.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)
        noise = config.model.noise
        self.noise = OrnsteinUhlenbeckNoise(
            dim_act,
            mu=noise.mu,
            theta=noise.theta,
            sigma=noise.sigma,
            dt=noise.dt,
            x0=noise.x0,
        )
        self.action_limit_low = totorch(
            config.env.env.action_space.low, device=self.device  # type: ignore
        )
        self.action_limit_high = totorch(
            config.env.env.action_space.high, device=self.device  # type: ignore
        )

    def act(self, state: torch.Tensor, is_training: bool = True) -> dict:
        """
        Produces an action given a state. Adds noise during training.

        Args:
            state (torch.Tensor): The input state.
            is_training (bool): Whether to add exploration noise.
        Returns:
            dict: Dictionary containing 'action' and 'action_wo_noise'.
        """
        action_dict = super().act(state)
        action = action_dict["action"]
        action_dict["action_wo_noise"] = action.clone()
        if is_training:
            noise = self.noise().to(action.device)
            action += noise
            action = torch.clip(action, self.action_limit_low, self.action_limit_high)
            action_dict["action"] = action
        return action_dict

    def loss(self, state: torch.Tensor, critics: CriticEnsemble) -> torch.Tensor:
        """
        Computes the loss for the actor by maximizing the critic's Q-value.

        Args:
            state (torch.Tensor): Batch of input states.
            critics (CriticEnsemble): Critic network(s).
        Returns:
            torch.Tensor: Actor loss (negative Q-value).
        """
        act_dict = self.act(state, is_training=False)
        action = act_dict["action"]
        q_values = critics.Q(state, action)
        q = critics.reduce(q_values, reduce_type=self.config.model.critic.reduce)
        return (-q).mean()


class DDPGCritic(CriticEnsemble):
    """
    DDPG critic module implementing Q-value estimation and Bellman target computation.

    Attributes:
        loss (PACBayesLoss): Loss function for training.
        gamma (float): Discount factor for future rewards.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)

    @torch.no_grad()
    def get_bellman_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        actor: DDPGActor,
    ) -> torch.Tensor:
        """
        Computes the Bellman target for critic training.

        Args:
            reward (torch.Tensor): Reward signal.
            next_state (torch.Tensor): Next state.
            done (torch.Tensor): Done flag (1 if terminal, else 0).
            actor (DDPGActor): Actor network (used for target action).
        Returns:
            torch.Tensor: Bellman target (y).
        """
        next_action_dict = actor.act_target(next_state)
        next_action = next_action_dict["action"]
        target_values = self.Q_t(next_state, next_action)
        target_value = self.reduce(
            target_values, reduce_type=self.config.model.critic.target_reduce
        )
        y = reward.unsqueeze(-1) + self._gamma * target_value * (1 - done.unsqueeze(-1))
        return y


class DeepDeterministicPolicyGradient(ActorCritic):
    """
    Full DDPG agent, combining actor and critic with target networks and experience replay.

    Args:
        config (MainConfig): Main configuration for training and environment.
        critic_type (Type): Class to be used for the critic (default: DDPGCritic).
        actor_type (Type): Class to be used for the actor (default: DDPGActor).
    Attributes:
        config (MainConfig): Configuration object.
        actor (DDPGActor): Actor network for action selection.
        critic (DDPGCritic): Critic network for Q-value estimation.
    """

    _agent_name = "DDPG"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = DDPGCritic,
        actor_type: type = DDPGActor,
    ) -> None:
        super().__init__(config, critic_type, actor_type)

    def store_transition(self, transition: dict) -> None:
        """
        Stores a transition in the replay buffer. Resets noise if episode is done.

        Args:
            transition (dict): A dictionary containing keys like 'state', 'action', 'reward', 'done'.
        """
        done = transition["terminated"] or transition["truncated"]
        if done:
            self.actor.noise.reset()
        return super().store_transition(transition)
