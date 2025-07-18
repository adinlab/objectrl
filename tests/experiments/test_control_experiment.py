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
import numpy as np
from unittest.mock import MagicMock, patch
import gymnasium as gym

from objectrl.experiments.control_experiment import ControlExperiment


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.env.name = "cheetah"
    config.system.device = torch.device("cpu")
    config.system.seed = 42
    config.training.max_steps = 10
    config.training.warmup_steps = 2
    config.training.learn_frequency = 1
    config.training.eval_frequency = 5
    config.training.eval_episodes = 2
    config.training.max_iter = 1
    config.training.n_epochs = 1
    config.training.reset_frequency = 4
    config.logging.save_frequency = 5
    config.progress = False
    return config


def make_dummy_env():
    env = MagicMock()
    env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), {})

    # Create a counter inside the env mock to count steps
    env.step_call_count = 0

    def step_fn(action):
        env.step_call_count += 1
        done = env.step_call_count >= 3  # done after 3 steps
        return (
            np.array([0.1, 0.1, 0.1, 0.1]),
            1.0,
            done,
            False,
            {},
        )

    env.step.side_effect = step_fn
    env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return env


class DummyAgent:
    def __init__(self, config):
        self.config = config
        self.logger = MagicMock()
        self.logger.log = MagicMock()
        self.logger.save = MagicMock()
        self.logger.save_eval_results = MagicMock()
        self.logger.episode_summary = MagicMock()

        # Mock methods to track call_count
        self.select_action = MagicMock(side_effect=self._select_action)
        self.learn = MagicMock(side_effect=self._learn)

    def _select_action(self, state, is_training=True):
        return {"action": torch.tensor([0.0, 0.0])}

    def _learn(self, max_iter, n_epochs):
        pass

    def requires_discrete_actions(self):
        return False

    def generate_transition(self, **kwargs):
        return {"dummy": "transition"}

    def store_transition(self, transition):
        pass

    def reset(self):
        pass

    def save(self):
        pass

    def eval(self):
        pass

    def train(self):
        pass


@patch(
    "objectrl.experiments.base_experiment.get_model",
    return_value=DummyAgent(MagicMock()),
)
@patch(
    "objectrl.experiments.base_experiment.make_env",
    side_effect=lambda env_name, seed, env_config, eval_env=False: make_dummy_env(),
)
@patch(
    "objectrl.experiments.control_experiment.totorch",
    side_effect=lambda x, device=None: torch.tensor(x, dtype=torch.float32),
)
@patch(
    "objectrl.experiments.control_experiment.tonumpy", side_effect=lambda x: x.numpy()
)
def test_control_experiment_train(
    mock_tonumpy,
    mock_totorch,
    mock_make_env,
    mock_get_model,
    mock_config,
):
    exp = ControlExperiment(mock_config)

    # Check init works
    assert exp.agent is not None
    assert exp.env is not None
    assert exp.eval_env is not None

    # Run training
    exp.train()

    # Check agent methods were used
    assert exp.agent.logger.save.call_count >= 1
    assert (
        exp.agent.select_action.call_count
        >= mock_config.training.max_steps - mock_config.training.warmup_steps
    )
    assert exp.agent.learn.call_count >= 1
    assert exp.agent.logger.save_eval_results.call_count >= 1
    assert exp.agent.logger.log.call_count >= 1
