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
