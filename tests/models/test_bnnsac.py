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
