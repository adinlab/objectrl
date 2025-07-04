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
import torch
from unittest.mock import MagicMock, patch

from objectrl.agents.base_agent import Agent  # Adjust the import path if needed
from tensordict import TensorDict

import gymnasium as gym


# Dummy Concrete Agent for Testing
class DummyAgent(Agent):
    def reset(self, *args, **kwargs):
        pass

    def learn(self, *args, **kwargs):
        pass

    def select_action(self, *args, **kwargs):
        return torch.tensor([0.0])


@pytest.fixture
def mock_config(tmp_path):
    env = MagicMock()
    env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
    env.name = "MockEnv"

    config = MagicMock()
    config.env = MagicMock(env=env, name="MockEnv")
    config.model.name = "MockModel"
    config.model.tau = 0.005
    config.system.device = torch.device("cpu")
    config.system.storing_device = torch.device("cpu")
    config.system.seed = 42
    config.training.gamma = 0.99
    config.training.buffer_size = 1000
    config.logging.result_path = tmp_path
    return config


@pytest.fixture
def agent(mock_config):
    return DummyAgent(mock_config)


# Tests
def test_generate_transition(agent):
    transition = agent.generate_transition(
        state=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        action=torch.tensor([0.1, -0.1]),
        reward=1.0,
        next_state=torch.tensor([1.1, 2.1, 3.1, 4.1]),
        terminated=True,
        truncated=False,
        step=5,
    )
    assert isinstance(transition, TensorDict)
    assert "state" in transition
    assert transition["reward"] == 1.0
    assert transition["terminated"] == 1.0
    assert transition["truncated"] == 0.0


def test_store_transition(agent):
    transition = ("s", "a", "r", "s'", False)
    with patch.object(agent.experience_memory, "add") as mock_add:
        agent.store_transition(transition)
        mock_add.assert_called_once_with(transition)


def test_save_and_load(agent, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    agent.logger.path = tmp_path
    agent.linear = torch.nn.Linear(4, 2)  # add a dummy parameter

    agent.save()
    assert checkpoint_path.exists()

    # Load into a new agent
    new_agent = DummyAgent(agent.config)
    new_agent.linear = torch.nn.Linear(4, 2)  # match architecture
    new_agent.load(checkpoint_path)
    for p1, p2 in zip(agent.parameters(), new_agent.parameters()):
        assert torch.allclose(p1, p2)


def test_requires_discrete_actions(agent):
    assert agent.requires_discrete_actions() is False
