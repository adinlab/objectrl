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

import math
from dataclasses import dataclass, field

from objectrl.models.oac import OACActor, OACCritic
from objectrl.nets.actor_nets import ActorNetProbabilistic
from objectrl.nets.critic_nets import CriticNet


# [start-config]
@dataclass
class OACActorConfig:
    """
    Configuration for the OAC Actor network.

    Attributes:
        arch (type): The architecture class used for the actor network.
        actor_type (type): The actor implementation class (OACActor).
    """

    arch: type = ActorNetProbabilistic
    actor_type: type = OACActor


@dataclass
class OACCriticConfig:
    """
    Configuration for the OAC Critic network.

    Attributes:
        arch (type): The architecture class used for the critic network.
        critic_type (type): The critic implementation class ( OACCritic).
    """

    arch: type = CriticNet
    critic_type: type = OACCritic


@dataclass
class CriticNoiseConfig:
    """
    Configuration for Gaussian noise added to critic target actions.

    Attributes:
        sigma_target (float): Standard deviation of Gaussian noise.
        noise_clamp (float): Maximum absolute value to clamp the noise.
    """

    sigma_target: float = math.sqrt(0.2)
    noise_clamp: float = 0.5


@dataclass
class ActorExplorationConfig:
    """
    Configuration for optimistic exploration noise in OAC.

    Attributes:
        delta (float): Uncertainty scaling factor for exploration.
        beta_ub (float): Upper bound multiplier on critic std deviation.
    """

    delta: float = 0.1
    beta_ub: float = 4.66


@dataclass
class OACConfig:
    """
    Top-level configuration for the OAC (Optimistic Actor Critic) agent.

    Attributes:
        name (str): Agent name identifier.
        loss (str): Loss function used for training.
        policy_delay (int): Number of critic updates per actor update.
        tau (float): Polyak averaging coefficient for target networks.
        noise (CriticNoiseConfig): Configuration for Gaussian target noise.
        exploration (ActorExplorationConfig): Config for exploration noise.
        target_entropy (float or None): Target policy entropy.
        alpha (float): Entropy regularization coefficient.
        actor (OACActorConfig): Actor network configuration.
        critic (OACCriticConfig): Critic network configuration.
    """

    name: str = "oac"
    loss: str = "MSELoss"
    policy_delay: int = 1
    tau: float = 0.005
    noise: CriticNoiseConfig = field(default_factory=CriticNoiseConfig)
    exploration: ActorExplorationConfig = field(default_factory=ActorExplorationConfig)
    target_entropy: float | None = None
    alpha: float = 1.0
    actor: OACActorConfig = field(default_factory=OACActorConfig)
    critic: OACCriticConfig = field(default_factory=OACCriticConfig)

    def __post_init__(self):
        if isinstance(self.noise, dict):
            self.noise = CriticNoiseConfig(**self.noise)
        if isinstance(self.exploration, dict):
            self.exploration = ActorExplorationConfig(**self.exploration)


# [end-config]
