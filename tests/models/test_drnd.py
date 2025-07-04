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

import torch
import torch.nn as nn
from types import SimpleNamespace
from pathlib import Path
from gymnasium.spaces import Box
import pytest
import types

from objectrl.models.drnd import DRNDActor, DRNDCritics, DRND, DRNDBonus


class DummyDRNDActorWrapper(torch.nn.Module):
    def __init__(self, dim_state, dim_act):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_state, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, dim_act),
        )

    def forward(self, x, is_training=False):
        action = self.net(x)
        return {"action": action, "action_logprob": torch.zeros(x.size(0), 1)}


@pytest.fixture
def dummy_config_drnd():
    model = SimpleNamespace(
        tau=0.005,
        name="drnd",
        policy_delay=1,
        loss="MSELoss",
        target_entropy=None,
        alpha=0.2,
        actor=SimpleNamespace(
            arch=lambda dim_state, dim_act, **kwargs: DummyDRNDActorWrapper(
                dim_state, dim_act
            ),
            has_target=False,
            reset=True,
            depth=2,
            width=64,
            activation="relu",
            norm=False,
            lambda_actor=0.1,
            n_heads=1,
        ),
        critic=SimpleNamespace(
            target_reduce="min",
            reduce="min",
            n_members=2,
            has_target=True,
            reset=True,
            arch=lambda dim_state, dim_act, **kwargs: torch.nn.Sequential(
                torch.nn.Linear(dim_state + dim_act, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            ),
            depth=2,
            width=64,
            activation="relu",
            norm=False,
            lambda_critic=0.5,
        ),
        bonus_conf=SimpleNamespace(
            n_members=2,
            learning_rate=1e-3,
            dim_out=32,
            scale_factor=0.5,
            depth=2,
            width=64,
            activation="relu",
            norm=False,
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
        batch_size=4,
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


def test_drnd_actor_loss(dummy_config_drnd):
    actor = DRNDActor(dummy_config_drnd, 4, 2)
    critics = DRNDCritics(dummy_config_drnd, 4, 2)
    bonus = DRNDBonus(dummy_config_drnd, 4, 2)

    state = torch.randn(6, 4)
    loss, act_dict = actor.loss(state, critics, bonus)

    assert isinstance(loss, torch.Tensor)
    assert "action" in act_dict
    assert act_dict["action"].shape == (6, 2)


def test_drnd_bonus_output(dummy_config_drnd):
    bonus = DRNDBonus(dummy_config_drnd, 4, 2)

    def fake_forward(self, sa):
        return torch.abs(torch.randn(sa.size(0)))

    bonus.forward = types.MethodType(fake_forward, bonus)

    state = torch.randn(5, 4)
    action = torch.randn(5, 2)
    sa = torch.cat([state, action], dim=-1)

    bonus_vals = bonus(sa)

    assert bonus_vals.shape == (5,)
    assert torch.all(bonus_vals >= 0)


def test_drnd_critic_bellman_target(dummy_config_drnd):
    critic = DRNDCritics(dummy_config_drnd, 4, 2)
    actor = DRNDActor(dummy_config_drnd, 4, 2)
    bonus = DRNDBonus(dummy_config_drnd, 4, 2)

    def fake_forward(self, sa):
        return torch.abs(torch.randn(sa.size(0)))

    bonus.forward = types.MethodType(fake_forward, bonus)

    batch_size = 4
    reward = torch.ones(batch_size)
    next_state = torch.randn(batch_size, 4)
    done = torch.zeros(batch_size)

    y = critic.get_bellman_target(reward, next_state, done, actor, bonus)

    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == batch_size


def test_drnd_predictor_update(dummy_config_drnd):
    bonus = DRNDBonus(dummy_config_drnd, 4, 2)

    def fake_update(self, sa):
        return torch.tensor(0.123)

    bonus.update = types.MethodType(fake_update, bonus)

    state = torch.randn(6, 4)
    action = torch.randn(6, 2)
    sa = torch.cat([state, action], dim=-1)

    pred_loss = bonus.update(sa)

    assert isinstance(pred_loss, torch.Tensor)
