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
from gymnasium.spaces import Box
import torch
from pathlib import Path

from objectrl.models.bnnsac import BNNSoftActorCritic

import torch.nn as nn


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


def simple_critic_arch(
    dim_state, dim_act, depth=2, width=64, act=nn.ReLU, has_norm=False
):
    layers = []
    input_dim = dim_state + dim_act
    for _ in range(depth):
        layers.append(nn.Linear(input_dim, width))
        if has_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(act())
        input_dim = width
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


@pytest.fixture
def dummy_config():
    system = SimpleNamespace(
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        seed=42,
    )

    training = SimpleNamespace(
        buffer_size=10000,
        gamma=0.99,
        optimizer="Adam",
        learning_rate=1e-3,
    )

    model = SimpleNamespace(
        tau=0.005,
        name="TestModel",
        policy_delay=2,
        loss="MSELoss",
        target_entropy=None,
        alpha=1.0,
        critic=SimpleNamespace(
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
            arch=simple_actor_arch,
            has_target=False,
            reset=True,
            depth=2,
            width=64,
            activation=nn.ReLU,
            norm=False,
            n_heads=1,
        ),
    )

    env_inner = SimpleNamespace(
        observation_space=Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
        action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=float),
    )
    env = SimpleNamespace(env=env_inner, name="DummyEnv")

    logging = SimpleNamespace(
        result_path=Path("./_logs"),
    )

    config = SimpleNamespace(
        system=system,
        training=training,
        model=model,
        env=env,
        logging=logging,
        verbose=True,
    )
    return config


def test_bnn_sac_init(dummy_config):
    agent = BNNSoftActorCritic(dummy_config)
    assert agent._agent_name == "BNN-SAC"
    assert hasattr(agent, "critic")
    assert hasattr(agent, "actor")


def test_bnn_sac_inheritance(dummy_config):
    agent = BNNSoftActorCritic(dummy_config)
    from objectrl.models.sac import SoftActorCritic

    assert isinstance(agent, SoftActorCritic)
