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

import torch
import torch.nn as nn
from types import SimpleNamespace
from pathlib import Path
from gymnasium.spaces import Box
import pytest
from objectrl.models.sac import SACActor, SACCritic, SoftActorCritic


def simple_actor_arch(
    dim_state, dim_act, depth=2, width=64, act=nn.ReLU, has_norm=False, n_heads=1
):
    layers = []
    in_dim = dim_state
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, width))
        if has_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(act())
        in_dim = width
    layers.append(nn.Linear(width, dim_act * n_heads))
    return nn.Sequential(*layers)


def simple_critic_arch(dim_state, dim_act, **kwargs):
    return nn.Sequential(
        nn.Linear(dim_state + dim_act, 64), nn.ReLU(), nn.Linear(64, 1)
    )


class DummySACActorWrapper(nn.Module):
    def __init__(self, dim_state, dim_act, arch):
        super().__init__()
        self.model = arch(dim_state, dim_act)

    def forward(self, x, is_training=False):
        action = self.model(x)
        action_logprob = torch.zeros_like(action[..., :1])
        return {
            "action": action,
            "action_logprob": action_logprob,
        }


class DummyCritic:
    def Q(self, state, action):
        return torch.ones(state.shape[0], 1)

    def Q_t(self, state, action):
        return torch.ones(state.shape[0], 1)

    def reduce(self, q_vals, reduce_type="min"):
        return q_vals

    def __getitem__(self, idx):
        return self


@pytest.fixture
def dummy_config():
    model = SimpleNamespace(
        tau=0.005,
        name="sac",
        policy_delay=1,
        loss="MSELoss",
        target_entropy=None,
        alpha=0.2,
        critic=SimpleNamespace(
            target_reduce="min",
            reduce="min",
            n_members=2,
            has_target=True,
            reset=True,
            arch=simple_critic_arch,
            depth=2,
            width=64,
            activation=nn.ReLU,
            norm=False,
        ),
        actor=SimpleNamespace(
            arch=lambda dim_state, dim_act, **kwargs: DummySACActorWrapper(
                dim_state, dim_act, simple_actor_arch
            ),
            has_target=False,
            reset=True,
            depth=2,
            width=64,
            activation=nn.ReLU,
            norm=False,
            n_heads=1,
        ),
    )

    env = SimpleNamespace(
        env=SimpleNamespace(
            action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=float),
            observation_space=Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
        ),
        name="DummyEnv",
    )

    training = SimpleNamespace(
        buffer_size=10000,
        gamma=0.99,
        optimizer="Adam",
        learning_rate=1e-3,
    )

    system = SimpleNamespace(
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        seed=42,
    )

    logging = SimpleNamespace(result_path=Path("./_logs"))

    return SimpleNamespace(
        model=model,
        env=env,
        training=training,
        system=system,
        logging=logging,
        verbose=True,
    )


def test_sac_actor_loss_and_update(dummy_config):
    actor = SACActor(dummy_config, dim_state=4, dim_act=2)
    critics = DummyCritic()
    state = torch.randn(8, 4)

    loss, act_dict = actor.loss(state, critics)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert "action" in act_dict and "action_logprob" in act_dict

    actor.update(state, critics)


def test_sac_critic_bellman_target(dummy_config):
    critic = SACCritic(dummy_config, dim_state=4, dim_act=2)
    actor = SACActor(dummy_config, dim_state=4, dim_act=2)
    batch_size = 4
    reward = torch.ones(batch_size)
    next_state = torch.randn(batch_size, 4)
    done = torch.zeros(batch_size)

    y = critic.get_bellman_target(reward, next_state, done, actor)
    assert y.shape == (batch_size, 1)
    assert torch.all(
        (0.0 <= y) & (y <= 2.0)
    ), f"Bellman target out of expected range: {y}"


def test_sac_agent_init(dummy_config):
    agent = SoftActorCritic(dummy_config)
    assert agent._agent_name == "SAC"
    assert hasattr(agent, "critic")
    assert hasattr(agent, "actor")
    assert callable(getattr(agent, "select_action", None))
