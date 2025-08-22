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
from unittest.mock import patch

from objectrl.models.redq import REDQCritic, RandomizedEnsembledDoubleQLearning
import torch
import torch.nn as nn
from types import SimpleNamespace
from gymnasium.spaces import Box
from pathlib import Path


def simple_critic_arch(dim_state, dim_act, **kwargs):
    return nn.Sequential(
        nn.Linear(dim_state + dim_act, 64), nn.ReLU(), nn.Linear(64, 1)
    )


def simple_actor_arch(dim_state, dim_act, **kwargs):
    return nn.Sequential(
        nn.Linear(dim_state, 64), nn.ReLU(), nn.Linear(64, dim_act), nn.Tanh()
    )


class DummyREDQActorWrapper(nn.Module):
    def __init__(self, dim_state, dim_act, arch_fn):
        super().__init__()
        self.model = arch_fn(dim_state, dim_act)

    def forward(self, x, **kwargs):
        action = self.model(x)
        batch_size = action.shape[0]
        dummy_logprob = torch.zeros(batch_size)
        return {"action": action, "action_logprob": dummy_logprob}


@pytest.fixture
def dummy_config():
    model = SimpleNamespace(
        tau=0.005,
        name="redq",
        policy_delay=1,
        loss="MSELoss",
        target_entropy=None,
        alpha=0.2,
        n_in_target=2,
        critic=SimpleNamespace(
            target_reduce="min",
            reduce="mean",
            n_members=5,
            has_target=True,
            reset=True,
            arch=simple_critic_arch,
            depth=2,
            width=64,
            activation=nn.ReLU,
            norm=False,
        ),
        actor=SimpleNamespace(
            arch=lambda dim_state, dim_act, **kwargs: DummyREDQActorWrapper(
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


def test_redq_critic_reduce(dummy_config):
    critic = REDQCritic(dummy_config, dim_state=4, dim_act=2)
    n_members = critic.n_members
    n_in_target = dummy_config.model.n_in_target

    batch_size = 5
    q_vals = [torch.randn(batch_size, 1) for _ in range(n_members)]

    min_reduced = critic.reduce(q_vals, reduce_type="min")
    assert min_reduced.shape == (batch_size, 1)
    assert torch.isfinite(min_reduced).all()

    sampled_indices = torch.randperm(n_members)[:n_in_target]
    stacked_sampled = torch.stack([q_vals[i] for i in sampled_indices], dim=-1).squeeze(
        -2
    )
    expected_min = stacked_sampled.min(dim=-1)[0]

    q_vals_tensor = torch.stack(q_vals, dim=0)
    mean_reduced = critic.reduce(q_vals_tensor, reduce_type="mean")
    assert mean_reduced.shape == (batch_size, 1)
    assert torch.isfinite(mean_reduced).all()

    with pytest.raises(ValueError):
        critic.reduce(q_vals, reduce_type="invalid_type")


def test_redq_agent_integration(dummy_config):
    agent = RandomizedEnsembledDoubleQLearning(dummy_config)

    batch_size = 4
    dim_state = 4
    dim_act = 2

    states = torch.randn(batch_size, dim_state)
    actions = torch.randn(batch_size, dim_act)

    action_dict = agent.actor.act(states)
    assert "action" in action_dict and "action_logprob" in action_dict
    assert action_dict["action"].shape == (batch_size, dim_act)
    assert action_dict["action_logprob"].shape == (batch_size,)

    q_vals = torch.stack(
        [
            member(torch.cat([states, actions], dim=-1))
            for member in agent.critic.model_ensemble
        ]
    )

    assert q_vals.shape[0] == agent.critic.n_members
    assert q_vals.shape[1] == batch_size
    assert q_vals.shape[2] == 1

    reduced_q = agent.critic.reduce(
        [q_vals[i] for i in range(agent.critic.n_members)], reduce_type="min"
    )
    assert reduced_q.squeeze(-1).shape == (batch_size,)
