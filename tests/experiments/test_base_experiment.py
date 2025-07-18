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
