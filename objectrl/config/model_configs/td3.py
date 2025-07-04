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
