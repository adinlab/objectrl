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
import numpy as np
import gymnasium as gym
from objectrl.utils.environment.noisy_wrappers import (
    NoisyActionWrapper,
    NoisyObservationWrapper,
)  # Adjust import


class DummyDiscreteEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.last_action = None

    def step(self, action):
        self.last_action = action
        obs = np.array([0.5, -0.5])
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self):
        return np.array([0.0, 0.0])


class DummyContinuousEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        self.last_action = None

    def step(self, action):
        self.last_action = action
        obs = np.array([0.1, 0.2, 0.3])
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self):
        return np.array([0.0, 0.0, 0.0])


def test_noisy_action_wrapper_discrete(monkeypatch):
    env = DummyDiscreteEnv()
    wrapper = NoisyActionWrapper(env, noise_act=1.0)

    monkeypatch.setattr(np.random, "random", lambda: 0.0)

    action = 1
    obs, reward, terminated, truncated, info = wrapper.step(action)
    assert env.last_action != action
    assert 0 <= env.last_action < env.action_space.n


def test_no_noise_action_wrapper_discrete(monkeypatch):
    env = DummyDiscreteEnv()
    wrapper = NoisyActionWrapper(env, noise_act=0.0)

    monkeypatch.setattr(np.random, "random", lambda: 1.0)

    action = 2
    obs, reward, terminated, truncated, info = wrapper.step(action)
    assert env.last_action == action


def test_noisy_action_wrapper_continuous(monkeypatch):
    env = DummyContinuousEnv()
    noise_level = 0.5
    wrapper = NoisyActionWrapper(env, noise_act=noise_level)

    monkeypatch.setattr(np.random, "randn", lambda *shape: np.ones(shape))

    action = np.array([0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = wrapper.step(action)
    expected_action = np.clip(
        action + noise_level * np.ones_like(action),
        env.action_space.low,
        env.action_space.high,
    )
    np.testing.assert_array_almost_equal(env.last_action, expected_action)


def test_noisy_observation_wrapper_with_ndarray(monkeypatch):
    env = DummyContinuousEnv()
    wrapper = NoisyObservationWrapper(env, noise_obs=0.3)

    obs = np.array([1.0, 2.0, 3.0])
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.zeros(shape))

    noisy_obs = wrapper.observation(obs)
    np.testing.assert_array_equal(noisy_obs, obs)


def test_noisy_observation_wrapper_with_dict(monkeypatch):
    env = DummyDiscreteEnv()
    wrapper = NoisyObservationWrapper(env, noise_obs=0.2)

    obs = {
        "position": np.array([1.0, 1.0]),
        "velocity": np.array([0.0, 0.0]),
        "other": 5,
    }
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.ones(shape))

    noisy_obs = wrapper.observation(obs)

    np.testing.assert_array_almost_equal(
        noisy_obs["position"], obs["position"] + 0.2 * np.ones_like(obs["position"])
    )
    np.testing.assert_array_almost_equal(
        noisy_obs["velocity"], obs["velocity"] + 0.2 * np.ones_like(obs["velocity"])
    )
    assert noisy_obs["other"] == obs["other"]


def test_noisy_observation_wrapper_invalid_type():
    env = DummyDiscreteEnv()
    wrapper = NoisyObservationWrapper(env, noise_obs=0.1)

    with pytest.raises(ValueError, match="Unsupported observation type"):
        wrapper.observation("invalid_obs_type")
