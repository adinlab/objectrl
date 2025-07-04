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
from typing import Literal

from objectrl.models.drnd import DRNDActor, DRNDCritics
from objectrl.nets.actor_nets import ActorNetProbabilistic
from objectrl.nets.critic_nets import CriticNet


# [start-config]
@dataclass
class DRNDBonusConfig:
    """
    Configuration for the DRND exploration bonus network.
    Implements the randomized target ensemble used for exploration.
    The default values follow Yang et al., 2024

    Attributes:
        depth (int): Number of hidden layers.
        width (int): Width (number of units) in each hidden layer.
        norm (bool): Whether to apply layer normalization.
        activation (str): Activation function to use ('relu' or 'crelu'). User should add other activation functions if needed.
        dim_out (int): Output dimensionality of the bonus network.
        scale_factor (float): Scaling factor between two bonus terms.
        n_members (int): Size of the target ensemble.
        learning_rate (float): Learning rate for training the predictor network.
    """

    depth: int = 4
    width: int = 256
    norm: bool = True
    activation: Literal["relu", "crelu"] = "relu"
    dim_out: int = 32
    scale_factor: float = 0.9
    n_members: int = 10
    learning_rate: float = 1e-4


@dataclass
class DRNDActorConfig:
    """
    Configuration for the actor network used in DRND.

    Attributes:
        arch (type): The neural network architecture to use.
        actor_type (type): The actor class (typically DRNDActor).
        lambda_actor (float): Scaling coefficient for exploration bonus in the actor loss.
    """

    arch: type = ActorNetProbabilistic
    actor_type: type = DRNDActor
    lambda_actor: float = 1.0


@dataclass
class DRNDCriticConfig:
    """
    Configuration for the critic network used in DRND.

    Attributes:
        arch (type): The neural network architecture to use.
        critic_type (type): The critic class (typically DRNDCritics).
        lambda_critic (float): Scaling coefficient for exploration bonus in the critic target.
    """

    arch: type = CriticNet
    critic_type: type = DRNDCritics
    lambda_critic: float = 1.0


@dataclass
class DRNDConfig:
    """
    Full configuration for the DRND algorithm.

    Attributes:
        name (str): Name of the algorithm.
        bonus_conf (DRNDBonusConfig): Configuration for bonus (RND) component.
        target_entropy (float | None): Target entropy for entropy regularization.
        alpha (float): Entropy regularization coefficient.
        loss (str): Type of loss function ('MSELoss').
        policy_delay (int): Number of critic updates per actor update.
        tau (float): Soft update coefficient for Polyak averaging
        actor (DRNDActorConfig): Configuration for actor.
        critic (DRNDCriticConfig): Configuration for critic.
    """

    name: str = "drnd"
    bonus_conf: DRNDBonusConfig = field(default_factory=DRNDBonusConfig)
    target_entropy: float | None = None
    alpha: float = 1.0
    loss: str = "MSELoss"
    policy_delay: int = 1
    tau: float = 0.005
    actor: DRNDActorConfig = field(default_factory=DRNDActorConfig)
    critic: DRNDCriticConfig = field(default_factory=DRNDCriticConfig)

    def __post_init__(self):
        if isinstance(self.bonus_conf, dict):
            self.bonus_conf = DRNDBonusConfig(**self.bonus_conf)


# [end-config]
