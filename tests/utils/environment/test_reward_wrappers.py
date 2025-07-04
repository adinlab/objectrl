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

import numpy as np
import pytest
from gymnasium import Env, spaces
from objectrl.utils.environment.reward_wrappers import PositionDelayWrapper


class DummyData:
    def __init__(self, qpos0=0.0, qvel0=0.0):
        self.qpos = np.array([qpos0])
        self.qvel = np.array([qvel0])


class DummyEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.data = DummyData()

    def step(self, action):
        obs = np.array([0.1, 0.2, 0.3])
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def test_initialization_and_attributes():
    env = DummyEnv()
    wrapper = PositionDelayWrapper(env, position_delay=5.0, ctrl_w=0.01)
    assert wrapper.position_delay == 5.0
    assert wrapper.ctrl_w == 0.01
    assert wrapper.env is env


def test_reward_below_position_delay():
    env = DummyEnv()
    wrapper = PositionDelayWrapper(env, position_delay=2.0, ctrl_w=0.1)
    wrapper.data = wrapper.env.data
    env.data.qpos[0] = 1.0
    env.data.qvel[0] = 3.0

    action = np.array([0.5, -0.5])
    obs, reward, terminated, truncated, info = wrapper.step(action)

    expected_ctrl_cost = wrapper.ctrl_w * np.sum(action**2)
    assert np.isclose(reward, -expected_ctrl_cost)
    assert info["x_pos"] == env.data.qpos[0]
    assert np.isclose(info["action_norm"], np.sum(action**2))


def test_reward_above_position_delay():
    env = DummyEnv()
    wrapper = PositionDelayWrapper(env, position_delay=1.5, ctrl_w=0.05)
    wrapper.data = wrapper.env.data
    env.data.qpos[0] = 2.0
    env.data.qvel[0] = 4.0

    action = np.array([1.0, 1.0])
    obs, reward, terminated, truncated, info = wrapper.step(action)

    expected_ctrl_cost = wrapper.ctrl_w * np.sum(action**2)
    expected_forward_reward = env.data.qvel[0]
    expected_reward = expected_forward_reward - expected_ctrl_cost

    assert np.isclose(reward, expected_reward)
    assert info["x_pos"] == env.data.qpos[0]
    assert np.isclose(info["action_norm"], np.sum(action**2))


def test_reward_method_direct_call():
    env = DummyEnv()
    wrapper = PositionDelayWrapper(env, position_delay=3.0, ctrl_w=0.01)
    wrapper.data = wrapper.env.data
    env.data.qpos[0] = 4.0
    env.data.qvel[0] = 5.0

    action = np.array([0.3, -0.4])
    reward = wrapper.reward(None, action)
    expected_ctrl_cost = wrapper.ctrl_w * np.sum(action**2)
    expected_forward_reward = env.data.qvel[0]
    expected_reward = expected_forward_reward - expected_ctrl_cost
    assert np.isclose(reward, expected_reward)
