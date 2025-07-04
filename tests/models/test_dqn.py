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
