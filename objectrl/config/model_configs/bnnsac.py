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

from objectrl.config.model_configs.sac import SACActorConfig, SACConfig
from objectrl.models.sac import SACCritic
from objectrl.nets.critic_nets import BNNCriticNet

# BNNSACActor is identical to SACActor
BNNSACActorConfig = SACActorConfig


@dataclass
class BNNSACCriticConfig:
    """
    Configuration for the BNN-SAC critic network ensemble.

    Args:
        config (MainConfig): Configuration object containing model hyperparameters.
        dim_state (int): Dimensionality of the state space.
        dim_act (int): Dimensionality of the action space.
    Attributes:
        arch (type): Neural network architecture class for the critic.
        critic_type (type): Critic class type.
    """

    arch: type = BNNCriticNet
    critic_type: type = SACCritic


@dataclass
class BNNSACConfig(SACConfig):
    """
    Main BNN-SAC algorithm configuration class.

    Args:
        config (MainConfig): Configuration object containing model hyperparameters.
    Attributes:
        name (str): Algorithm identifier.
        loss (str): Loss function used for critic training.
        policy_delay (int): Number of critic updates per actor update.
        tau (float): Polyak averaging coefficient for target network updates.
        target_entropy (float | None): Target entropy for automatic temperature tuning.
        alpha (float): Initial temperature parameter.
        actor (SACActorConfig): Actor configuration.
        critic (SACCriticConfig): Critic configuration.
    """

    name: str = "bnnsac"

    actor: BNNSACActorConfig = field(default_factory=BNNSACActorConfig)
    critic: BNNSACCriticConfig = field(default_factory=BNNSACCriticConfig)
