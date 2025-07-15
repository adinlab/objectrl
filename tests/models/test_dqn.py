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
from types import SimpleNamespace
from pathlib import Path
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from unittest import mock

from objectrl.models.dqn import DQN, DQNCritic


class DummyDQNNet(nn.Module):
    def __init__(
        self, dim_state, dim_act, depth=1, width=32, act=nn.ReLU, has_norm=False
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim_state, width))
        layers.append(act())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act())
        layers.append(nn.Linear(width, dim_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def dummy_config():
    system = SimpleNamespace(
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        seed=42,
    )

    training = SimpleNamespace(
        batch_size=4,
        gamma=0.99,
        buffer_size=1000,
        optimizer="Adam",
        learning_rate=1e-3,
    )

    critic = SimpleNamespace(
        arch=DummyDQNNet,
        critic_type=DQNCritic,
        n_members=1,
        exploration_rate=0.0,
        has_target=True,
        reset=False,
        loss="MSELoss",
        depth=1,
        width=32,
        activation=nn.ReLU,
        norm=False,
    )

    model = SimpleNamespace(
        name="dqn",
        loss="MSELoss",
        tau=0.005,
        critic=critic,
    )

    env_inner = SimpleNamespace(
        observation_space=Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
        action_space=Discrete(3),
    )
    env = SimpleNamespace(env=env_inner, name="DummyEnv")

    logging = SimpleNamespace(result_path=Path("./_logs"))

    config = SimpleNamespace(
        system=system,
        training=training,
        model=model,
        env=env,
        logging=logging,
        verbose=True,
    )
    return config


def test_dqcritic_update_and_act(dummy_config):
    dim_state = dummy_config.env.env.observation_space.shape[0]
    dim_act = dummy_config.env.env.action_space.n

    critic = DQNCritic(dummy_config, dim_state, dim_act)

    batch_size = 2
    state = torch.randn(batch_size, dim_state)
    action = torch.randint(0, dim_act, (batch_size,))
    y = torch.randn(batch_size, 1)

    critic.optim = mock.Mock()
    critic.loss = nn.MSELoss(reduction="none")

    critic.update(state, action, y)
    critic.optim.zero_grad.assert_called()
    critic.optim.step.assert_called()

    critic._explore_rate = 0.0
    acts = critic.act(state, is_training=True)
    assert acts.shape == (batch_size, 1)
    assert acts.max() < dim_act and acts.min() >= 0

    critic._explore_rate = 1.0
    random_act = critic.act(state, is_training=True)
    assert random_act.shape == (1,)
    assert (random_act >= 0).all() and (random_act < dim_act).all()

    q_vals = critic.Q(state, None)
    q_t_vals = critic.Q_t(state, None)
    assert q_vals.shape == (batch_size, dim_act)
    assert q_t_vals.shape == (batch_size, dim_act)

    reward = torch.randn(batch_size)
    next_state = torch.randn(batch_size, dim_state)
    done = torch.zeros(batch_size)

    y_bellman = critic.get_bellman_target(reward, next_state, done)
    assert y_bellman.shape == (batch_size, 1)


def test_dqn_learn_select_reset(dummy_config):
    dim_state = dummy_config.env.env.observation_space.shape[0]
    dim_act = dummy_config.env.env.action_space.n

    agent = DQN(dummy_config, DQNCritic)

    class DummyMemory:
        def __len__(self):
            return 10

        def get_steps_and_iterator(self, n_epochs, max_iter, batch_size):
            return 1

        def get_next_batch(self, batch_size):
            return {
                "state": torch.randn(batch_size, dim_state),
                "action": torch.randint(0, dim_act, (batch_size,)),
                "reward": torch.randn(batch_size),
                "next_state": torch.randn(batch_size, dim_state),
                "terminated": torch.zeros(batch_size),
            }

    agent.experience_memory = DummyMemory()

    agent.critic.get_bellman_target = mock.Mock(return_value=torch.randn(4, 1))
    agent.critic.update = mock.Mock()
    agent.critic.update_target = mock.Mock()
    agent.critic.has_target = True

    agent.learn(max_iter=1, n_epochs=0)
    agent.critic.get_bellman_target.assert_called()
    agent.critic.update.assert_called()
    agent.critic.update_target.assert_called()

    state = torch.randn(1, dim_state)
    action = agent.select_action(state, is_training=True)
    assert isinstance(action, torch.Tensor)

    agent.critic._reset = True
    agent.critic.reset = mock.Mock()
    agent.reset()
    agent.critic.reset.assert_called()
