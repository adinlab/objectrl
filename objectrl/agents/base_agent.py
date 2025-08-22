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

import typing
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from tensordict import TensorDict
from torch import nn as nn

from objectrl.loggers.logger import Logger
from objectrl.replay_buffers.experience_memory import ReplayBuffer

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class Agent(nn.Module, ABC):
    """
    Abstract base class for reinforcement learning agents.
    Child classes must implement:
    - reset: For resetting agent-specific state
    - learn: The core learning loop
    - select_action: Policy used to choose actions from states

    Attributes:
        config (MainConfig): Main configuration object with all submodules.
        device (torch.device): Device on which computations are performed.
        config_env: Environment-specific config.
        config_train: Training-specific config.
        dim_state (int): Shape of the observation space.
        dim_act (int): Shape of the action space.
        _gamma (float): Discount factor for future rewards.
        _tau (float): Polyak averaging coefficient for target updates.
        experience_memory : Experience replay buffer.
        logger (Logger): Logger used to track training and evaluation.
    """

    def __init__(self, config: "MainConfig") -> None:
        """
        Initializes the base agent class.

        Args:
            config (MainConfig): The experiment's configuration object.
        Returns:
            None
        """
        super().__init__()
        self.config = config
        self.device = config.system.device
        self.config_env = config.env
        self.config_train = config.training

        # Get the state space dimensionality
        assert isinstance(
            self.config_env.env.observation_space, gym.spaces.Box
        ), "The library requires continuous state spaces"
        assert (
            len(self.config_env.env.observation_space.shape) == 1
        ), f"Observation space must be an integer, got {self.config_env.env.observation_space.shape}"
        self.dim_state = self.config_env.env.observation_space.shape[0]

        # Get the action space dimensionality
        if isinstance(self.config_env.env.action_space, gym.spaces.Box):
            assert (
                len(self.config_env.env.action_space.shape) == 1
            ), f"Observation space must be an integer, got {self.config_env.env.observation_space.shape}"
            self.dim_act = self.config_env.env.action_space.shape[0]
        elif isinstance(self.config_env.env.action_space, gym.spaces.Discrete):
            self.dim_act = self.config_env.env.action_space.n
        else:
            raise NotImplementedError(
                f"{self.config_env.env.action_space} is not supported."
            )

        self._gamma = config.training.gamma
        self._tau = config.model.tau

        self.experience_memory = ReplayBuffer(
            torch.device(config.system.device),
            torch.device(config.system.storing_device),
            config.training.buffer_size,
        )
        self.logger = Logger(
            Path(config.logging.result_path),
            config.env.name,
            config.model.name,
            config.system.seed,
            config=config,
        )

        # Per default all agents assume continuous action spaces
        self._discrete_action_space = False

    def generate_transition(self, **kwargs):
        """
        Constructs a transition dictionary for storing in the experience memory.

        The transition includes the current state, action taken, reward received,
        next state, and episode termination information. All tensors are moved
        to the appropriate storage device, and scalar values are converted to floats.

        Expected kwargs:
            state (torch.Tensor): The current state.
            action (torch.Tensor): The action taken.
            reward (float or scalar Tensor): The reward received after taking the action.
            next_state (torch.Tensor): The next state after the transition.
            terminated (bool or int): Indicator whether the episode terminated.
            truncated (bool or int): Indicator whether the episode was truncated.
            step (int): Environment step count or index.

        Returns:
            TensorDict: A dictionary representing a single transition, ready to be stored.
        """
        device = self.experience_memory.storing_device
        transition = TensorDict(
            {
                "state": kwargs["state"].to(device),
                "action": kwargs["action"].to(device),
                "reward": float(kwargs["reward"]),
                "next_state": kwargs["next_state"].to(device),
                "terminated": float(kwargs["terminated"]),
                "truncated": float(kwargs["truncated"]),
                "step": kwargs["step"],
            }
        )
        return transition

    def store_transition(self, transition: tuple[Any, ...]) -> None:
        """
        Stores a transition tuple into the experience replay buffer.

        Args:
            transition (tuple): A transition (s, a, r, s', done) to be stored.
        Returns:
            None
        """
        self.experience_memory.add(transition)

    def save(self) -> None:
        """
        Saves the model weights to disk at the logger's checkpoint path.
        This method saves the current state of the agent, including model parameters.

        Args:
            None
        Returns:
            None
        """
        state_dict = self.state_dict()
        torch.save(state_dict, self.logger.path / "checkpoint.pt")

    def load(self, path: str | Path) -> None:
        """
        Loads model weights from a given checkpoint path.

        Args:
            path (str or Path): Path to the saved model checkpoint.
        Returns:
            None
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def requires_discrete_actions(self) -> bool:
        return self._discrete_action_space

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """
        Reset any internal agent state. Must be implemented in subclass.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs) -> None:
        """
        Perform learning updates from replay buffer. Must be implemented in subclass.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None
        """
        pass

    @abstractmethod
    def select_action(self, *args, **kwargs) -> torch.Tensor:
        """
        Select an action given the current state. Must be implemented in subclass.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            torch.Tensor: The selected action tensor.
        """
        pass
