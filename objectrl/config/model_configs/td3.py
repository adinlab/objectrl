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

from objectrl.models.td3 import TD3Actor, TD3Critic
from objectrl.nets.actor_nets import ActorNet
from objectrl.nets.critic_nets import CriticNet


# [start-config]
@dataclass
class ActorNoiseConfig:
    """
    Configuration for noise added to actor actions in TD3.

    Attributes:
        policy_noise (float): Std dev of noise added during training.
        target_policy_noise (float): Std dev of noise added to target policy actions.
        target_policy_noise_clip (float): Clipping range for target policy noise.
    """

    policy_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_policy_noise_clip: float = 0.5


@dataclass
class TD3ActorConfig:
    """
    Configuration for the TD3 actor network.

    Attributes:
        arch (type): Actor network architecture class.
        actor_type (type): Actor class type.
        has_target (bool): Whether the actor has a target network.
    """

    arch: type = ActorNet
    actor_type: type = TD3Actor
    has_target: bool = True


@dataclass
class TD3CriticConfig:
    """
    Configuration for the TD3 critic network ensemble.

    Attributes:
        arch (type): Critic network architecture class.
        critic_type (type): Critic class type.
    """

    arch: type = CriticNet
    critic_type: type = TD3Critic


@dataclass
class TD3Config:
    """
    Main TD3 algorithm configuration.

    Attributes:
        name (str): Algorithm identifier.
        noise (ActorNoiseConfig): Noise parameters for exploration.
        loss (str): Loss function for critic training.
        policy_delay (int): Number of critic updates per actor update.
        tau (float): Polyak averaging coefficient for target network updates.
        actor (TD3ActorConfig): Actor network configuration.
        critic (TD3CriticConfig): Critic network configuration.
    """

    name: str = "td3"
    noise: ActorNoiseConfig = field(default_factory=ActorNoiseConfig)
    loss: str = "MSELoss"
    policy_delay: int = 2
    tau: float = 0.005

    actor: TD3ActorConfig = field(default_factory=TD3ActorConfig)
    critic: TD3CriticConfig = field(default_factory=TD3CriticConfig)

    def __post_init__(self):
        if isinstance(self.noise, dict):
            self.noise = ActorNoiseConfig(**self.noise)


# [end-config]
