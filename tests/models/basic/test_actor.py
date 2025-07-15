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
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from objectrl.models.basic.actor import Actor


class DummyActor(Actor):
    def loss(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)

    def compute_loss(self, critics, states):
        # Return a differentiable scalar connected to model parameters
        return self.model.linear.weight.norm()

    def update_target(self):
        # Proper soft update of target parameters from model parameters
        tau = self._tau
        for target_param, param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Real module with parameters and correct forward signature
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x, is_training=True):
        return {"action": self.linear(x)}


# Fixture for reusable config
@pytest.fixture
def dummy_config():
    return SimpleNamespace(
        verbose=False,
        system=SimpleNamespace(device=torch.device("cpu")),
        training=SimpleNamespace(gamma=0.99, optimizer="Adam", learning_rate=1e-3),
        model=SimpleNamespace(
            tau=0.005,
            actor=SimpleNamespace(
                has_target=True,
                arch=MagicMock(),
                depth=2,
                width=64,
                activation="relu",
                norm=False,
                n_heads=1,
                max_grad_norm=1.0,
                reset=False,  # added reset flag since your Actor expects it
            ),
        ),
    )


def test_actor_initialization(dummy_config):
    dummy_config.model.actor.arch.side_effect = lambda *args, **kwargs: DummyModel()
    actor = DummyActor(dummy_config, dim_state=4, dim_act=2)
    assert isinstance(actor.model, nn.Module)


def test_act_calls_model(dummy_config):
    dummy_config.model.actor.arch.side_effect = lambda *args, **kwargs: DummyModel()
    actor = DummyActor(dummy_config, dim_state=4, dim_act=2)
    state = torch.rand(1, 4)
    out = actor.act(state)
    assert isinstance(out, dict)
    assert "action" in out


def test_act_target(dummy_config):
    dummy_config.model.actor.arch.side_effect = lambda *args, **kwargs: DummyModel()
    actor = DummyActor(dummy_config, dim_state=4, dim_act=2)
    state = torch.rand(1, 4)
    out = actor.act_target(state)
    assert isinstance(out, dict)
    assert "action" in out


def test_update_target(dummy_config):
    dummy_config.model.actor.arch.side_effect = lambda *args, **kwargs: DummyModel()
    actor = DummyActor(dummy_config, dim_state=4, dim_act=2)

    # Manually perturb target params so they're different from model params
    with torch.no_grad():
        for target_param in actor.target.parameters():
            target_param.add_(torch.randn_like(target_param) * 0.1)

    before = [p.clone() for p in actor.target.parameters()]
    actor.update_target()
    after = list(actor.target.parameters())

    # After update, target params should have changed compared to before
    changed = [not torch.equal(b, a) for b, a in zip(before, after)]
    assert any(changed), "Target parameters did not change after update_target"


def test_update_calls_loss_and_optimizer_step(dummy_config):
    dummy_config.model.actor.arch.side_effect = lambda *args, **kwargs: DummyModel()
    actor = DummyActor(dummy_config, dim_state=4, dim_act=2)
    critics = MagicMock()
    state = torch.rand(10, 4)

    before = [p.clone() for p in actor.model.parameters()]
    actor.update(state, critics)
    after = list(actor.model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_reset_reinitializes_model_and_target(dummy_config):
    dummy_config.model.actor.arch.side_effect = lambda *args, **kwargs: DummyModel()
    actor = DummyActor(dummy_config, dim_state=4, dim_act=2)

    old_model = actor.model
    old_target = actor.target
    actor.reset()
    assert actor.model is not old_model
    assert actor.target is not old_target
