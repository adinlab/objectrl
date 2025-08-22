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

from objectrl.models.bnnsac import BNNSoftActorCritic
from objectrl.models.ddpg import DeepDeterministicPolicyGradient
from objectrl.models.dqn import DQN
from objectrl.models.drnd import DRND
from objectrl.models.oac import OptimisticActorCritic
from objectrl.models.pbac import PACBayesianAC
from objectrl.models.ppo import ProximalPolicyOptimization
from objectrl.models.redq import RandomizedEnsembledDoubleQLearning
from objectrl.models.sac import SoftActorCritic
from objectrl.models.td3 import TwinDelayedDeepDeterministicPolicyGradient


def get_model(config) -> object:  # noqa: C901
    """
    Factory function to instantiate a reinforcement learning agent based on the configuration.
    Args:
        config: Configuration object containing model specifications.
            Expected to have:
                - config.model.name (str): Name of the model to instantiate (e.g., "ddpg", "sac").
                - config.model.actor: Actor-related configuration including `actor_type`.
                - config.model.critic: Critic-related configuration including `critic_type`.
    Returns:
        object: An instance of the specified model class.
    """
    model_name = config.model.name.lower()
    if hasattr(config.model, "actor"):
        actor = config.model.actor
    if hasattr(config.model, "critic"):
        critic = config.model.critic
    match model_name:
        case "bnnsac":
            return BNNSoftActorCritic(config, critic.critic_type, actor.actor_type)
        case "ddpg":
            return DeepDeterministicPolicyGradient(
                config, critic.critic_type, actor.actor_type
            )
        case "dqn":
            return DQN(config, critic.critic_type)
        case "drnd":
            return DRND(config, critic.critic_type, actor.actor_type)
        case "oac":
            return OptimisticActorCritic(config, critic.critic_type, actor.actor_type)
        case "pbac":
            return PACBayesianAC(config, critic.critic_type, actor.actor_type)
        case "ppo":
            return ProximalPolicyOptimization(
                config, critic.critic_type, actor.actor_type
            )
        case "redq":
            return RandomizedEnsembledDoubleQLearning(
                config, critic.critic_type, actor.actor_type
            )
        case "sac":
            return SoftActorCritic(config, critic.critic_type, actor.actor_type)
        case "td3":
            return TwinDelayedDeepDeterministicPolicyGradient(
                config, critic.critic_type, actor.actor_type
            )

        case _:
            raise ValueError(f"Unknown model: {model_name}")
