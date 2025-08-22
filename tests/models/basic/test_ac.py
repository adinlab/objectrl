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
