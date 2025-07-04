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
