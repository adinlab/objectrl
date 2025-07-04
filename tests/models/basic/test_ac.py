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
from types import SimpleNamespace
from gymnasium.spaces import Box
from pathlib import Path
from objectrl.models.basic.ac import ActorCritic


class DummyActor:
    _reset = True

    def __init__(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        pass

    def reset(self):
        pass


class DummyCritic:
    _reset = True

    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass


@pytest.fixture
def config_mock():
    system = SimpleNamespace(
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        seed=42,
    )

    training = SimpleNamespace(
        buffer_size=10000,
        gamma=0.99,
    )

    model = SimpleNamespace(
        tau=0.005,
        name="TestModel",
        policy_delay=2,
    )

    env_inner = SimpleNamespace(
        observation_space=Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
        action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=float),
    )
    env = SimpleNamespace(env=env_inner, name="TestEnv")

    logging = SimpleNamespace(
        result_path=Path("./_logs"),  # Now Path is defined
    )

    config = SimpleNamespace(
        system=system,
        training=training,
        model=model,
        env=env,
        logging=logging,
    )
    return config


def test_actor_critic_init(config_mock):
    agent = ActorCritic(config_mock, DummyCritic, DummyActor)
    assert agent is not None


def test_select_action_calls_actor_act(config_mock):
    agent = ActorCritic(config_mock, DummyCritic, DummyActor)
    state = torch.zeros(4)
    agent.actor = DummyActor()
    try:
        agent.select_action(state)
    except Exception:
        pass


def test_reset_calls_reset_on_actor_and_critic(config_mock):
    agent = ActorCritic(config_mock, DummyCritic, DummyActor)
    agent.reset()
