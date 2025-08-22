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
from torch.distributions import TransformedDistribution

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.actor import Actor
from objectrl.models.basic.critic import CriticEnsemble

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class OptimisticNoise:
    """
    Computes optimistic exploration noise as described in the OAC algorithm.

    Attributes:
        beta_ub (float): Coefficient on standard deviation of Q-values.
        delta (float): Exploration confidence parameter.
    """

    def __init__(self, beta_ub: float, delta: float) -> None:
        self.beta_ub = beta_ub
        self.delta = delta

    def compute(
        self,
        state: torch.Tensor,
        critics: CriticEnsemble,
        transformed_dist: TransformedDistribution,
    ) -> dict:
        """
        Computes the optimistic adjustment to the mean of the action distribution.

        Args:
            state (Tensor): Input state tensor.
            critics (CriticEnsemble): Critic ensemble to evaluate Q-values.
            transformed_dist (TransformedDistribution): Tanh-transformed distribution from actor.
        Returns:
            dict: Contains adjusted mean ('mu_e') and scale ('scale') of the new action distribution.
        """
        pre_tanh_mu = transformed_dist.base_dist.loc  # type: ignore
        pre_tanh_mu.requires_grad_()

        tanh_mu = torch.tanh(pre_tanh_mu)
        q_values = critics.Q(state, tanh_mu)

        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0, unbiased=False)
        q_ub = q_mean + self.beta_ub * q_std

        grad = torch.autograd.grad(q_ub.sum(), pre_tanh_mu)[0]
        sigma_t = transformed_dist.base_dist.scale.square()  # type: ignore
        denom = torch.sqrt(torch.sum(grad.square() * sigma_t)) + 1e-6

        mu_c = math.sqrt(2.0 * self.delta) * (sigma_t * grad) / denom
        mu_e = pre_tanh_mu + mu_c

        return {"mu_e": mu_e, "scale": transformed_dist.base_dist.scale}  # type: ignore


class GaussianNoise:
    """
    Adds Gaussian noise to actions, used for target value perturbation in critic updates.

    Attributes:
        sigma_target (float): Standard deviation of the noise.
        noise_clamp (float): Value to clamp the noise between [-noise_clamp, noise_clamp].
    """

    def __init__(self, sigma_target=0, noise_clamp=0.15):
        self.sigma_target = sigma_target
        self.noise_clamp = noise_clamp

    def add_noise(self, next_action_shape: torch.Size) -> torch.Tensor:
        """
        Generates Gaussian noise for a given action shape.

        Args:
            next_action_shape (torch.Size): Shape of the action tensor.
        Returns:
            Tensor: Clamped noise tensor.
        """
        noise = torch.distributions.Normal(0, self.sigma_target).sample(
            sample_shape=next_action_shape
        )
        noise = noise.clamp(-self.noise_clamp, self.noise_clamp)
        return noise


class OACActor(Actor):
    """
    OAC-specific actor class with optimistic noise-based exploration.

    Inherits from a base probabilistic actor, and modifies the loss function
    to incorporate upper-confidence bounds via Q-value ensembles.

    Args:
        config (MainConfig): Global configuration.
        dim_state (int): Dimensionality of observation space.
        dim_act (int): Dimensionality of action space.
    Attributes:
        optimist_noise (OptimisticNoise): Instance to compute optimistic exploration noise.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int):
        super().__init__(config, dim_state, dim_act)
        exploration = config.model.exploration
        self.optimistic_noise = OptimisticNoise(
            beta_ub=exploration.beta_ub, delta=exploration.delta
        )

    def loss(self, state: torch.Tensor, critics: CriticEnsemble) -> torch.Tensor:
        """
        Computes the actor loss using the mean Q-value.

        Args:
            state (Tensor): Input states.
            critics (CriticEnsemble): Critic networks.
        Returns:
            Tensor: Scalar loss value.
        """
        act_dict = self.act(state, is_training=False)
        action = act_dict["action"]
        q_values = critics.Q(state, action)
        q = critics.reduce(q_values, reduce_type=self.config.model.critic.reduce)
        return (-q).mean()


class OACCritic(CriticEnsemble):
    """
    OAC-specific critic ensemble class, adds Gaussian noise to actions
    for more robust target computation.

    Args:
        config (MainConfig): Global configuration.
        dim_state (int): Dimensionality of state space.
        dim_act (int): Dimensionality of action space.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int):
        super().__init__(config, dim_state, dim_act)
        noise = config.model.noise
        self.noise = GaussianNoise(
            sigma_target=noise.sigma_target,
            noise_clamp=noise.noise_clamp,
        )

    @torch.no_grad()
    def get_bellman_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        actor: OACActor,
    ) -> torch.Tensor:
        """
        Computes the Bellman target for TD learning using noisy next actions.

        Args:
            reward (Tensor): Reward signal.
            next_state (Tensor): Next state input.
            done (Tensor): Episode termination flags.
            actor (OACActor): Actor used to compute next action.
        Returns:
            Tensor: Bellman target values.
        """
        act_dict = actor.act(next_state)
        next_action = act_dict["action"]
        noise = self.noise.add_noise(next_action.shape).to(self.device)
        next_action += noise
        target_values = self.Q_t(next_state, next_action)  # Use perturbed action
        target_value = self.reduce(
            target_values, reduce_type=self.config.model.critic.target_reduce
        )
        y = reward.unsqueeze(-1) + (
            self._gamma * target_value * (1 - done.unsqueeze(-1))
        )
        return y


class OptimisticActorCritic(ActorCritic):
    """
    Main OAC agent class that integrates the OAC actor and critic.
    Implements action selection with or without optimistic exploration.

    Args:
        config (MainConfig): Global training and model configuration.
        critic_type (type): Class used for the critic ensemble (default OACCritic).
        actor_type (type): Class used for the actor (default OACActor).
    Attributes:
        config (MainConfig): Configuration object.
        actor (OACActor): Actor network for action selection.
        critic (OACCritic): Critic network for Q-value estimation.
        device (torch.device): Device for computation (CPU or GPU).
    """

    _agent_name = "OAC"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = OACCritic,
        actor_type: type = OACActor,
    ):
        super().__init__(config, critic_type, actor_type)

    def select_action(
        self, state: torch.Tensor, is_training: bool = True
    ) -> torch.Tensor:
        """
        Selects an action given a state, optionally applying optimistic exploration.

        Args:
            state (Tensor): Input state tensor.
            is_training (bool): If True, applies optimistic exploration noise.
        Returns:
            Tensor: Action to execute in the environment.
        """
        state = state.to(self.device)
        act_dict = self.actor.act(state)
        action, transformed_dist = act_dict["action"], act_dict["dist"]
        if is_training:
            result = self.actor.optimistic_noise.compute(
                state, self.critic, transformed_dist
            )
            mu_e, scale = result["mu_e"], result["scale"]
            dist_bt = torch.distributions.Normal(mu_e, scale, validate_args=False)
            dist = torch.distributions.TransformedDistribution(
                dist_bt, torch.distributions.transforms.TanhTransform(cache_size=1)
            )
            action = dist.sample()
            act_dict["dist"] = dist
            act_dict["action_logprob"] = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            action = torch.tanh(transformed_dist.base_dist.loc)  # type: ignore
        act_dict["action"] = action
        return act_dict
