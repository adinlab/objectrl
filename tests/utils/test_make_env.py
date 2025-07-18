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

import pytest
import gymnasium as gym
import numpy as np
from unittest.mock import MagicMock, patch

from objectrl.utils.make_env import make_env


class DummyEnv(gym.Env):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.reset_called_with_seed = None

    def reset(self, seed=None, **kwargs):
        self.reset_called_with_seed = seed
        return np.zeros(self.observation_space.shape), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def action_space_seed(self, seed):
        pass

    def observation_space_seed(self, seed):
        pass


@pytest.fixture
def env_config():
    class NoisyConfig:
        noisy_act = 0.0
        noisy_obs = 0.0

    class Config:
        noisy = NoisyConfig()
        position_delay = 0
        control_cost_weight = 0.0

    return Config()


@patch("gymnasium.envs.registry")
@patch("gymnasium.make")
def test_make_env_basic(mock_make, mock_registry, env_config):
    mock_registry.keys.return_value = ["Dummy-v0"]

    dummy_env = DummyEnv(
        gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    )
    mock_make.return_value = dummy_env

    env = make_env("Dummy-v0", seed=123, env_config=env_config)

    assert env is not None
    assert dummy_env.reset_called_with_seed == 123
    assert isinstance(env, gym.Env)
