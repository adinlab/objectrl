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

from dataclasses import dataclass, field

from objectrl.models.ppo import PPOActor, PPOActorNetProbabilistic, PPOCritic
from objectrl.nets.critic_nets import ValueNet


# [start-config]
@dataclass
class PPOActorConfig:
    """
    Configuration for the PPO actor network.

    Attributes:
        arch (type): The architecture class to be used for the actor network.
        actor_type (type): The class implementing the PPO actor logic.
        has_target (bool): Whether to maintain a target network for the actor.
        width (int): Width of hidden layers in the actor network.
        max_grad_norm (float): Maximum norm for gradient clipping in the actor.
    """

    arch: type = PPOActorNetProbabilistic
    actor_type: type = PPOActor
    has_target: bool = False
    width: int = 64
    max_grad_norm: float = 0.5


@dataclass
class PPOCriticConfig:
    """
    Configuration for the PPO critic network.

    Attributes:
        arch (type): The architecture class to be used for the critic network.
        critic_type (type): The class implementing the PPO critic logic.
        has_target (bool): Whether to maintain a target network for the critic.
        n_members (int): Number of ensemble members in the critic network.
        width (int): Width of hidden layers in the critic network.
        max_grad_norm (float): Maximum norm for gradient clipping in the critic.
    """

    arch: type = ValueNet
    critic_type: type = PPOCritic
    has_target: bool = False
    n_members: int = 1
    width: int = 64
    max_grad_norm: float = 0.5


@dataclass
class PPOConfig:
    """
    Full configuration for a PPO agent, including actor, critic, and optimization hyperparameters.

    Attributes:
        name (str): Identifier name for the PPO configuration.
        loss (str): Name of the loss function to use (e.g., 'MSELoss').
        tau (float): Polyak averaging coefficient for target network updates.
        policy_delay (int): Delay interval between policy (actor) updates.
        max_grad_norm (float): Maximum norm for gradient clipping globally.
        clip_rate (float): Clipping factor for the PPO objective.
        GAE_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        normalize_advantages (bool): Whether to normalize advantages during training.
        entropy_coef (float): Coefficient for entropy regularization.
        actor (PPOActorConfig): Configuration object for the PPO actor.
        critic (PPOCriticConfig): Configuration object for the PPO critic.
    """

    name: str = "ppo"
    loss: str = "MSELoss"

    tau: float = 0.0
    policy_delay: int = 1
    max_grad_norm: float = 0.5
    clip_rate: float = 0.2
    GAE_lambda: float = 0.95
    normalize_advantages: bool = True
    entropy_coef: float = 0.0

    actor: PPOActorConfig = field(default_factory=PPOActorConfig)
    critic: PPOCriticConfig = field(default_factory=PPOCriticConfig)


# [end-config]
