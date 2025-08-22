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
from typing import Literal

from objectrl.models.pbac import PBACActor, PBACCritic
from objectrl.nets.actor_nets import ActorNetProbabilistic
from objectrl.nets.critic_nets import CriticNet


# [start-config]
@dataclass
class CriticLossConfig:
    """
    Configuration for the PAC-Bayesian critic loss.

    Attributes:
        complexity_coef (float): Weight of the complexity term in the loss.
        prior_variance (float): Prior variance used in KL divergence computation.
        bootstrap_rate (float): Rate for masking samples during loss computation.
        logvar_lower_clamp (float): Minimum value to clamp log variance to.
        logvar_upper_clamp (float): Maximum value to clamp log variance to.
        reduction (Literal): Type of loss reduction ('mean', 'sum', or 'none').
    """

    complexity_coef: float = 0.01
    prior_variance: float = 1.0
    bootstrap_rate: float = 0.05
    logvar_lower_clamp: float = 0.01
    logvar_upper_clamp: float = 100.0
    reduction: Literal["mean", "sum", "none"] = "mean"


@dataclass
class PBACActorConfig:
    """
    Configuration for the PBAC Actor network.

    Attributes:
        arch (type): Actor architecture class.
        actor_type (type): Actor class to use (PBACActor).
        has_target (bool): Whether to use a target actor network.
        n_heads (int): Number of actor output heads (for ensemble policy).
    """

    arch: type = ActorNetProbabilistic
    actor_type: type = PBACActor
    has_target: bool = False
    n_heads: int = 10


@dataclass
class PBACCriticConfig:
    """
    Configuration for the PBAC Critic ensemble.

    Attributes:
        arch (type): Critic network architecture.
        critic_type (type): Critic implementation class.
        n_members (int): Number of critics in the ensemble.
    """

    arch: type = CriticNet
    critic_type: type = PBACCritic
    n_members: int = 10


@dataclass
class PBACConfig:
    """
    Top-level configuration for the PBAC agent.

    Attributes:
        name (str): Agent name identifier.
        lossparams (CriticLossConfig): Configuration for critic loss function.
        target_entropy (float or None): Target entropy for policy regularization.
        loss (str): Name of the loss function.
        policy_delay (int): Number of critic updates per actor update.
        tau (float): Polyak averaging coefficient for target updates.
        posterior_sampling_rate (int): Frequency to resample actor ensemble index.
        alpha (float): Entropy temperature.
        gamma (float): Discount factor for Bellman target.
        sig2_lowerclamp (float): Minimum variance to avoid division by zero.
        actor (PBACActorConfig): Actor configuration.
        critic (PBACCriticConfig): Critic configuration.
    """

    name: str = "pbac"
    lossparams: CriticLossConfig = field(default_factory=CriticLossConfig)
    target_entropy: float | None = None
    loss: str = "PACBayesLoss"
    policy_delay: int = 1
    tau: float = 0.005
    posterior_sampling_rate: int = 5
    alpha: float = 1.0
    gamma: float = 0.99
    sig2_lowerclamp: float = 1e-6

    actor: PBACActorConfig = field(default_factory=PBACActorConfig)
    critic: PBACCriticConfig = field(default_factory=PBACCriticConfig)

    def __post_init__(self):
        if isinstance(self.lossparams, dict):
            self.lossparams = CriticLossConfig(**self.lossparams)


# [end-config]
