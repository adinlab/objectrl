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
import torch.nn as nn
from pathlib import Path
from unittest import mock
from tensordict import TensorDict

from objectrl.models.ddpg import DeepDeterministicPolicyGradient


class DummyActorModel(nn.Module):
    def __init__(self, dim_state, dim_act):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_state, 64),
            nn.ReLU(),
            nn.Linear(64, dim_act),
        )

    def forward(self, x, is_training=False):
        return self.model(x)


class DummyActorWrapper(nn.Module):
    def __init__(self, dim_state, dim_act):
        super().__init__()
        self.model = DummyActorModel(dim_state, dim_act)
        self.noise = None

    def forward(self, x, is_training=False):
        action_wo_noise = self.model(x)
        return {
            "action": action_wo_noise,
            "action_wo_noise": action_wo_noise,
        }


def dummy_actor_arch(dim_state, dim_act, **kwargs):
    return DummyActorWrapper(dim_state, dim_act)


def simple_critic_arch(dim_state, dim_act, **kwargs):
    return nn.Sequential(
        nn.Linear(dim_state + dim_act, 64), nn.ReLU(), nn.Linear(64, 1)
    )


@pytest.fixture
def dummy_config():
    system = SimpleNamespace(
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        seed=0,
    )

    training = SimpleNamespace(
        buffer_size=10000,
        gamma=0.99,
        optimizer="Adam",
        learning_rate=1e-3,
    )

    model = SimpleNamespace(
        tau=0.005,
        name="TestDDPG",
        policy_delay=1,
        loss="MSELoss",
        noise=SimpleNamespace(mu=0.0, theta=0.15, sigma=0.2, dt=0.01, x0=None),
        critic=SimpleNamespace(
            n_members=1,
            has_target=True,
            reset=True,
            arch=simple_critic_arch,
            reduce="min",
            target_reduce="min",
            depth=2,
            width=64,
            activation=nn.ReLU,
            norm=False,
        ),
        actor=SimpleNamespace(
            has_target=True,
            reset=True,
            arch=dummy_actor_arch,
            depth=2,
            width=64,
            activation=nn.ReLU,
            norm=False,
            n_heads=1,
        ),
    )

    env_inner = SimpleNamespace(
        observation_space=Box(low=-1.0, high=1.0, shape=(3,), dtype=float),
        action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=float),
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


def test_ddpg_initialization(dummy_config):
    agent = DeepDeterministicPolicyGradient(dummy_config)
    assert agent._agent_name == "DDPG"
    assert hasattr(agent, "actor")
    assert hasattr(agent, "critic")


def test_ddpg_act_and_noise_reset(dummy_config):
    agent = DeepDeterministicPolicyGradient(dummy_config)

    state = torch.rand(1, 3)

    noise_tensor = torch.tensor([0.1, -0.1], device=state.device, dtype=state.dtype)
    agent.actor.noise = mock.Mock(return_value=noise_tensor)

    action_dict = agent.actor.act(state, is_training=True)

    assert "action" in action_dict
    assert "action_wo_noise" in action_dict

    diff = torch.norm(action_dict["action"] - action_dict["action_wo_noise"])
    assert diff == 0.0, "In dummy, no noise is added, so actions should match"

    if hasattr(agent.actor.noise, "state"):
        agent.actor.noise.state = torch.ones(2) * 100.0

        transition = TensorDict(
            {
                "terminated": torch.tensor(True),
                "truncated": torch.tensor(False),
            },
            batch_size=[],
        )
        agent.store_transition(transition)
