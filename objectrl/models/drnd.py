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
from torch import nn as nn

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.models.basic.ensemble import Ensemble
from objectrl.models.sac import SACActor
from objectrl.utils.net_utils import MLP
from objectrl.utils.utils import dim_check

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class DRNDBonus(nn.Module):
    """
    Distributional Random Network Distillation (DRND) bonus module.
    Provides an exploration bonus based on disagreement between an ensemble of
    target networks and a learned predictor network. Based on Yang et al. (2024).

    Args:
        config (MainConfig): Main experiment/configuration object.
        dim_state (int): Observation space dimension.
        dim_act (int): Action space dimension.
    Attributes:
        target_ensemble (Ensemble): Ensemble of target networks.
        predictor (nn.Module): Predictor network for state-action pairs.
        optim_pred (torch.optim.Optimizer): Optimizer for the predictor.
        n_members (int): Number of ensemble members.
        device (torch.device): Device for computations.
        bonus_conf (BonusConfig): Configuration for the bonus module.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__()
        self.device = config.system.device
        self.bonus_conf = config.model.bonus_conf
        self.n_members = self.bonus_conf.n_members
        self.config = config

        self.gen_net = lambda: MLP(
            dim_state + dim_act,
            self.bonus_conf.dim_out,
            self.bonus_conf.depth,
            self.bonus_conf.width,
            act=self.bonus_conf.activation,
            has_norm=self.bonus_conf.norm,
        ).to(self.device)
        self.reset()

    def reset(self) -> None:
        self.target_ensemble = Ensemble(
            n_members=int(self.n_members),
            models=[self.gen_net() for _ in range(self.n_members)],
            device=self.device,
        )

        self.predictor = self.gen_net()
        self.optim_pred = torch.optim.Adam(
            self.predictor.parameters(), lr=self.bonus_conf.learning_rate
        )

    @torch.no_grad()
    def bonus(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the DRND exploration bonus for a given (state, action) pair.
        Combines two terms:
        - Squared difference between predictor and mean of ensemble
        - Normalized difference in variances (distributional bonus)

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.
        Returns:
            torch.Tensor: Exploration bonus.
        """
        sa = torch.cat((state, action), -1)
        target_pred = self.target_ensemble(sa)
        mu = target_pred.mean(0)
        mu2 = mu.pow(2)
        B2 = target_pred.pow(2).mean(0)
        pred = self.predictor(sa)
        dim_check(pred, mu)
        fst = (pred - mu).pow(2).sum(1, keepdim=True)
        # WARNING: The original implementation by Yang et al. has both mean -> sqrt and sqrt -> mean
        #   The second one is what seems to be used more often...
        # The clipping is an undocumented feature in the code
        # snd = torch.sqrt(((pred.pow(2) - mu2).abs() / (B2 - mu2)).clip(1e-3, 1).mean(1))
        snd = torch.sqrt(((pred.pow(2) - mu2).abs() / (B2 - mu2)).clip(1e-3, 1)).mean(
            1, keepdim=True
        )

        return (
            self.bonus_conf.scale_factor * fst
            + (1 - self.bonus_conf.scale_factor) * snd
        )

    def mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_ensemble(x).mean(0)

    def B2(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_ensemble(x).pow(2).mean(0)

    def update_predictor(self, state: torch.Tensor, action: torch.Tensor) -> None:
        """
        Updates the predictor network using a randomly selected ensemble member
        as the regression target.

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.
        Returns:
            None
        """
        sa = torch.cat((state, action), -1)
        self.optim_pred.zero_grad()
        c = torch.randint(self.n_members, ()).item()
        c_target = self.target_ensemble[c](sa)
        pred = self.predictor(sa)
        loss = (pred - c_target).pow(2).mean()
        loss.backward()
        self.optim_pred.step()


# [start-actor-code]
class DRNDActor(SACActor):
    """
    Actor network for DRND, based on SAC but augmented with an exploration bonus.

    Args:
        config (MainConfig): Experiment configuration.
        dim_state (tuple): Observation space dimension.
        dim_act (tuple): Action space dimension.
    Attributes:
        lambda_actor (float): Regularization coefficient for the actor loss.:
    """

    def __init__(
        self, config: "MainConfig", dim_state: tuple[int, ...], dim_act: tuple[int, ...]
    ) -> None:
        super().__init__(config, dim_state, dim_act)

        self.lambda_actor = config.model.actor.lambda_actor

    def loss(
        self, state: torch.Tensor, critics: "DRNDCritics", bonus_ensemble: DRNDBonus
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute actor loss including the entropy term and the DRND exploration bonus.

        Args:
            state (torch.Tensor): Batch of input states.
            critics (DRNDCritics): Critic network(s).
            bonus_ensemble (DRNDBonus): Bonus ensemble for exploration.
        Returns:
            loss (Tensor): Total actor loss.
            act_dict (dict): Output of actor network.
        """
        loss, act_dict = super().loss(state, critics)
        bonus = bonus_ensemble.bonus(state, act_dict["action"]).mean()
        return loss + bonus, act_dict

    def update(
        self, state: torch.Tensor, critics: "DRNDCritics", bonus_ensemble: DRNDBonus
    ) -> None:
        """
        Perform a gradient step for the actor.

        Args:
            state (torch.Tensor): Batch of input states.
            critics (DRNDCritics): Critic network(s).
            bonus_ensemble (DRNDBonus): Bonus ensemble for exploration.
        Returns:
            None
        """
        self.optim.zero_grad()
        loss, act_dict = self.loss(state, critics, bonus_ensemble)
        loss.backward()
        self.optim.step()
        self.update_alpha(act_dict)

        self.iter += 1  # Increment iteration counter


# [end-actor-code]


# [start-critic-code]
class DRNDCritics(CriticEnsemble):
    """
    Critic module for DRND that incorporates exploration bonus into target computation.

    Args:
        config (MainConfig): Experiment configuration.
        dim_state (tuple): Observation space dimension.
        dim_act (tuple): Action space dimension.
    Attributes:
        lambda_critic (float): Regularization coefficient for the critic loss.
        _gamma (float): Discount factor for future rewards.
        _agent_name (str): Name of the agent.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int):
        super().__init__(config, dim_state, dim_act)
        self.lambda_critic = config.model.critic.lambda_critic

    @torch.no_grad()
    def get_bellman_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        actor: DRNDActor,
        bonus_ensemble: DRNDBonus,
    ) -> torch.Tensor:
        """
        Computes the Bellman target including entropy regularization and exploration penalty.

        Args:
            reward (torch.Tensor): Reward signal.
            next_state (torch.Tensor): Next state.
            done (torch.Tensor): Done flag (1 if terminal, else 0).
            actor (DRNDActor): Actor network (used for target action).
            bonus_ensemble (DRNDBonus): Bonus ensemble for exploration.
        Returns:
            y (Tensor): Bellman target.
        """
        alpha = actor.log_alpha.exp().detach()

        act_dict = actor.act(next_state)
        next_action, log_prob = act_dict["action"], act_dict["action_logprob"]

        target_values = self.Q_t(next_state, next_action)
        target_reduced = self.reduce(
            target_values, reduce_type=self.config.model.critic.target_reduce
        )
        bonus = bonus_ensemble.bonus(next_state, next_action)
        q_target = target_reduced - alpha * log_prob - self.lambda_critic * bonus
        reward = reward.unsqueeze(-1)
        dim_check(q_target, reward)
        y = reward + (self._gamma * q_target * (1 - done.unsqueeze(-1)))
        return y


# [end-critic-code]


# # [start-drnd-code]
class DRND(ActorCritic):
    """
    DRND agent integrating exploration through Distributional Random Network Distillation.
    Yang et al. (2024): Exploration and Anti-Exploration with Distributional Random Network Distillation

    Implements actor-critic logic where:
    - Actor loss is regularized by an exploration bonus
    - Critic targets include bonus penalties
    - Bonus predictor is trained online
    """

    _agent_name = "DRND"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = DRNDCritics,
        actor_type: type = DRNDActor,
        bonus_type: type = DRNDBonus,
    ) -> None:
        super().__init__(config, critic_type, actor_type)

        self.bonus_ensemble = bonus_type(config, self.dim_state, self.dim_act)

    def learn(self, max_iter: int = 1, n_epochs: int = 0) -> None:
        """
        Perform the learning process for the agent.

        Args:
            max_iter (int): Maximum number of iterations for learning.
            n_epochs (int): Number of epochs for training. If 0, random sampling is used.
        Returns:
            None
        """
        # Check if there is enough data in memory to sample a batch
        if self.config_train.batch_size > len(self.experience_memory):
            return None

        # Determine the number of steps and initialize the iterator
        n_steps = self.experience_memory.get_steps_and_iterator(
            n_epochs, max_iter, self.config_train.batch_size
        )

        for _ in range(n_steps):
            # Get batch using the internal iterator
            batch = self.experience_memory.get_next_batch(self.config_train.batch_size)

            bellman_target = self.critic.get_bellman_target(
                batch["reward"],
                batch["next_state"],
                batch["terminated"],
                self.actor,
                self.bonus_ensemble,
            )
            self.critic.update(batch["state"], batch["action"], bellman_target)

            # Update the actor network periodically
            if self.n_iter % self.policy_delay == 0:
                self.actor.update(batch["state"], self.critic, self.bonus_ensemble)
                if self.actor.has_target:
                    self.actor.update_target()

            # Update target networks
            if self.critic.has_target:
                self.critic.update_target()
            self.bonus_ensemble.update_predictor(batch["state"], batch["action"])
            self.n_iter += 1
        return None


# [end-drnd-code]
