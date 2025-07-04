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
