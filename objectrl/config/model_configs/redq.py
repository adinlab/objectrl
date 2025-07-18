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
from objectrl.models.redq import REDQCritic
from objectrl.nets.critic_nets import CriticNet


# [start-config]
@dataclass
class REDQActorConfig(SACActorConfig):
    """Actor configuration for REDQ, inherits SACActorConfig without changes."""

    pass


# [start-critic-config]
@dataclass
class REDQCriticConfig:
    """
    Configuration class for the REDQ critic ensemble.

    Attributes:
        arch (type): Neural network architecture for critics.
        critic_type (type): Critic class type.
        n_members (int): Number of critics in the ensemble.
        reduce (str): Reduction method during training.
        target_reduce (str): Reduction method for target Q-value computation.
    """

    arch: type = CriticNet
    critic_type: type = REDQCritic
    n_members: int = 10
    reduce: str = "mean"
    target_reduce: str = "min"


# [end-critic-config]


@dataclass
class REDQConfig(SACConfig):
    """
    Main configuration class for the REDQ algorithm,
    extending SACConfig with REDQ-specific parameters.

    Attributes:
        name (str): Algorithm name identifier.
        n_in_target (int): Number of critics randomly sampled in target Q-value computation.
        policy_delay (int): Number of critic updates per actor update.
        actor (REDQActorConfig): Actor configuration.
        critic (REDQCriticConfig): Critic configuration.
    """

    name: str = "redq"
    n_in_target: int = 2
    policy_delay: int = 20

    actor: REDQActorConfig = field(default_factory=REDQActorConfig)
    critic: REDQCriticConfig = field(default_factory=REDQCriticConfig)


# [end-config]
