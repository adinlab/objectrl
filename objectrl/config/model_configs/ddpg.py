# -----------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 ADIN Lab
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
