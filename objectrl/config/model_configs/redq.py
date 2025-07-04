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
