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
import gymnasium as gym
from unittest.mock import MagicMock, patch

from objectrl.experiments.base_experiment import Experiment


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.env.name = "cheetah"
    config.system.device = torch.device("cpu")
    config.system.seed = 123
    config.model.name = "MockModel"
    return config


def make_dummy_env():
    env = MagicMock(spec=gym.Env)
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
    env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
    return env


class DummyAgent:
    def __init__(self, config):
        self.config = config
        self.logger = MagicMock()
        self.logger.log = MagicMock()

    def requires_discrete_actions(self):
        return False

    def __str__(self):
        return "DummyAgent()"


@patch(
    "objectrl.experiments.base_experiment.make_env",
    side_effect=lambda *args, **kwargs: make_dummy_env(),
)
@patch(
    "objectrl.experiments.base_experiment.get_model",
    side_effect=lambda config: DummyAgent(config),
)
def test_experiment_initialization(mock_get_model, mock_make_env, mock_config):
    experiment = Experiment(mock_config)

    # Check that environments and agent were initialized
    assert experiment.env is not None
    assert experiment.eval_env is not None
    assert isinstance(experiment.agent, DummyAgent)

    # Check compatibility flag
    assert experiment._discrete_action_space is False

    # Check agent was logged
    experiment.agent.logger.log.assert_called()
