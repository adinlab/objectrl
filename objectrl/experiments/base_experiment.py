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

        self.env = make_env(
            self.config.env.name, self.config.system.seed, self.config.env
        )
        self.eval_env = make_env(
            self.config.env.name,
            self.config.system.seed,
            self.config.env,
            eval_env=True,
            num_envs=(
                self.config.training.eval_episodes
                if self.config.training.parallelize_eval
                else 1
            ),
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
