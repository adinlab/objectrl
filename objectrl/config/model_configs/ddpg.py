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

from objectrl.models.ddpg import DDPGActor, DDPGCritic
from objectrl.nets.actor_nets import ActorNet
from objectrl.nets.critic_nets import CriticNet


# [start-config]
@dataclass
class ActorNoiseConfig:
    """
    Configuration for the Ornstein-Uhlenbeck process used to add noise to actions during training.

    Attributes:
        mu (float): Long-running mean of the noise process.
        theta (float): Speed of mean reversion.
        sigma (float): Volatility (standard deviation of noise).
        dt (float): Time step size.
        x0 (float): Initial state of the noise process (optional).
    """

    mu: float = 0
    theta: float = 0.15
    sigma: float = 0.2
    dt: float = 1e-2
    x0: float | None = None


@dataclass
class DDPGActorConfig:
    """
    Configuration for the DDPG actor network.

    Attributes:
        arch (type): Class for the actor network architecture.
        actor_type (type): Actor class to be used.
        has_target (bool): Whether the actor maintains a target network.
    """

    arch: type = ActorNet
    actor_type: type = DDPGActor
    has_target: bool = True


@dataclass
class DDPGCriticConfig:
    """
    Configuration for the DDPG critic network.

    Attributes:
        arch (type): Class for the critic network architecture.
        critic_type (type): Critic class to be used.
        n_members (int): Number of critic networks to use in the ensemble.
        loss (str): Name of the loss function to use ( "MSELoss").
        policy_delay (int): Number of critic updates per actor update.
        tau (float): Soft update coefficient for Polyak averaging.
    """

    arch: type = CriticNet
    critic_type: type = DDPGCritic
    n_members: int = 1


@dataclass
class DDPGConfig:
    """
    Top-level configuration for the Deep Deterministic Policy Gradient (DDPG) agent.

    Attributes:
        name (str): Name of the algorithm.
        noise (ActorNoiseConfig): Noise configuration for exploration.
        loss (str): Loss function for critic training.
        policy_delay (int): How often to update the actor policy.
        tau (float): Soft update coefficient for target networks.
        actor (DDPGActorConfig): Configuration for the actor.
        critic (DDPGCriticConfig): Configuration for the critic.
    """

    name: str = "ddpg"
    noise: ActorNoiseConfig = field(default_factory=ActorNoiseConfig)
    loss: str = "MSELoss"
    policy_delay: int = 1
    tau: float = 0.005
    actor: DDPGActorConfig = field(default_factory=DDPGActorConfig)
    critic: DDPGCriticConfig = field(default_factory=DDPGCriticConfig)

    def __post_init__(self) -> None:
        """
        Converts `noise` from a dictionary to an ActorNoiseConfig if needed.
        Useful when loading from a JSON or dict-based config file.

        Args:
            None
        Returns:
            None
        """
        if isinstance(self.noise, dict):
            self.noise = ActorNoiseConfig(**self.noise)


# [end-config]
