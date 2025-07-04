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
import torch.nn as nn
from types import SimpleNamespace
from gymnasium.spaces import Box
from pathlib import Path
from objectrl.models.td3 import (
    TD3Actor,
    TD3Critic,
    TwinDelayedDeepDeterministicPolicyGradient,
)


class DummyActorModel(nn.Module):
    def __init__(self, dim_state, dim_act):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_state, 64),
            nn.ReLU(),
            nn.Linear(64, dim_act),
            nn.Tanh(),
        )

    def forward(self, x, is_training=False):
        return self.model(x)


class DummyActorWrapper(nn.Module):
    def __init__(self, dim_state, dim_act):
        super().__init__()
        self.model = DummyActorModel(dim_state, dim_act)

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


class DummyCritic:
    def __init__(self):
        pass

    def Q(self, state, action):
        return torch.ones(state.shape[0], 1)

    def Q_t(self, state, action):
        return torch.ones(state.shape[0], 1)

    def __getitem__(self, idx):
        return self


@pytest.fixture
def dummy_config():
    noise = SimpleNamespace(
        policy_noise=0.1,
        target_policy_noise=0.2,
        target_policy_noise_clip=0.3,
    )

    critic = SimpleNamespace(
        target_reduce="min",
        n_members=2,
        has_target=True,
        reset=True,
        arch=simple_critic_arch,
        depth=2,
        width=64,
        activation=nn.ReLU,
        norm=False,
    )

    actor = SimpleNamespace(
        has_target=True,
        reset=False,
        arch=dummy_actor_arch,
        depth=2,
        width=64,
        activation=nn.ReLU,
        norm=False,
        n_heads=1,
    )

    model = SimpleNamespace(
        noise=noise,
        critic=critic,
        actor=actor,
        tau=0.005,
        policy_delay=2,
        name="td3",
        loss="MSELoss",
    )

    env = SimpleNamespace(
        env=SimpleNamespace(
            action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=float),
            observation_space=Box(low=-1.0, high=1.0, shape=(3,), dtype=float),
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

    logging = logging = SimpleNamespace(result_path=Path("./_logs"))

    return SimpleNamespace(
        model=model,
        env=env,
        training=training,
        system=system,
        logging=logging,
        verbose=True,
    )


def test_td3_actor_act_and_act_target(dummy_config):
    actor = TD3Actor(dummy_config, dim_state=3, dim_act=2)

    state = torch.zeros(5, 3)

    result = actor.act(state, is_training=True)
    assert "action" in result and "action_wo_noise" in result
    assert result["action"].shape == (5, 2)
    torch.testing.assert_close(
        result["action_wo_noise"], result["action"], rtol=1e-2, atol=0.5
    )

    result_eval = actor.act(state, is_training=False)
    torch.testing.assert_close(result_eval["action"], result_eval["action_wo_noise"])

    target_result = actor.act_target(state)
    assert "action" in target_result
    assert torch.all(target_result["action"] <= actor.action_limit_high)
    assert torch.all(target_result["action"] >= actor.action_limit_low)


def test_td3_actor_loss(dummy_config):
    actor = TD3Actor(dummy_config, dim_state=3, dim_act=2)
    dummy_critic = DummyCritic()
    state = torch.zeros(4, 3)

    loss = actor.loss(state, critics=[dummy_critic])
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert torch.isclose(loss, torch.tensor(-1.0))


def test_td3_critic_get_bellman_target(dummy_config):
    critic = TD3Critic(dummy_config, dim_state=3, dim_act=2)
    batch_size = 4
    reward = torch.ones(batch_size)
    next_state = torch.zeros(batch_size, 3)
    done = torch.zeros(batch_size)
    actor = TD3Actor(dummy_config, dim_state=3, dim_act=2)

    y = critic.get_bellman_target(reward, next_state, done, actor)
    assert y.shape == (batch_size, 1)
    assert torch.all(
        (0.0 <= y) & (y <= 2.0)
    ), f"Bellman target out of expected range: {y}"


def test_td3_agent_init(dummy_config):
    agent = TwinDelayedDeepDeterministicPolicyGradient(dummy_config)
    assert agent._agent_name == "TD3"
    assert hasattr(agent, "critic")
    assert hasattr(agent, "actor")
    assert callable(getattr(agent, "select_action", None))
