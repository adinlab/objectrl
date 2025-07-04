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

from typing import cast

import gymnasium as gym

from objectrl.config.config import MainConfig
from objectrl.models.get_model import get_model
from objectrl.utils.make_env import make_env


# [start-class]
class Experiment:
    """
    A base class for setting up and managing reinforcement learning experiments.

    Attributes:
        config : MainConfig
            The configuration used throughout the experiment.
        n_total_steps : int
            Counter for the total number of steps taken during training.
        env : gym.Env
            The main training environment.
        eval_env : gym.Env
            The evaluation environment used to test agent performance.
        agent : Any
            The reinforcement learning agent initialized based on the provided config.
    """

    def __init__(self, config: "MainConfig") -> None:
        """
        Initializes the Experiment with a given configuration.
        """
        self.config = config
        self.n_total_steps = 0

        env_mappings = {
            "ant": "Ant-v5",
            "cartpole": "CartPole-v1",
            "cheetah": "HalfCheetah-v5",
            "hopper": "Hopper-v5",
            "humanoid": "Humanoid-v5",
            "reacher": "Reacher-v5",
            "swimmer": "Swimmer-v5",
            "walker2d": "Walker2d-v5",
        }

        # Use either the nickname in `env_mappings` or the original gymnasium name
        env_name = env_mappings.get(self.config.env.name, self.config.env.name)

        self.env = make_env(env_name, self.config.system.seed, self.config.env)
        self.eval_env = make_env(
            env_name, self.config.system.seed, self.config.env, eval_env=True
        )

        # Extract environmental hyperparameters into the general config file
        self.config.env.env = cast(gym.Env, self.env)

        self.agent = get_model(self.config)
        self.agent.logger.log(f"Model: \n{self.agent}")

        self._discrete_action_space = isinstance(
            self.env.action_space, gym.spaces.Discrete
        )

        if self._discrete_action_space and not self.agent.requires_discrete_actions():

            raise NotImplementedError(
                "The chosen agent is only available for continuous action spaces."
            )
        elif not self._discrete_action_space and self.agent.requires_discrete_actions():

            raise NotImplementedError(
                "The chosen agent is only available for discrete action spaces."
            )

    def train(self) -> None:
        """Starts the training process of the reinforcement learning agent.

        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError(
            f"train() not implemented for {self.__class__.__name__}!"
        )

    def test(self) -> None:
        """Evaluates the performance of the trained agent.

        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError(
            f"test() not implemented for {self.__class__.__name__}!"
        )


# [end-class]
