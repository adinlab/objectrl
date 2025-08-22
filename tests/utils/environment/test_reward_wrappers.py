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
