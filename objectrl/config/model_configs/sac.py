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

from objectrl.models.sac import SACActor, SACCritic
from objectrl.nets.actor_nets import ActorNetProbabilistic
from objectrl.nets.critic_nets import CriticNet


# [start-config]
@dataclass
class SACActorConfig:
    """
    Configuration for the SAC actor network.

    Attributes:
        arch (type): Neural network architecture class for the actor.
        actor_type (type): Actor class type.
    """

    arch: type = ActorNetProbabilistic
    actor_type: type = SACActor


@dataclass
class SACCriticConfig:
    """
    Configuration for the SAC critic network ensemble.

    Attributes:
        arch (type): Neural network architecture class for the critic.
        critic_type (type): Critic class type.
    """

    arch: type = CriticNet
    critic_type: type = SACCritic


@dataclass
class SACConfig:
    """
    Main SAC algorithm configuration class.

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

    name: str = "sac"
    loss: str = "MSELoss"
    policy_delay: int = 1
    tau: float = 0.005
    target_entropy: float | None = None
    alpha: float = 1.0

    actor: SACActorConfig = field(default_factory=SACActorConfig)
    critic: SACCriticConfig = field(default_factory=SACCriticConfig)


# [end-config]
