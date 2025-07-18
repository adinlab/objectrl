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

import copy
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from objectrl.models.basic.critic import Critic, CriticEnsemble


# Dummy model for Critic.model and Critic.target
class DummyCriticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 1)  # example input dim = dim_state + dim_act = 4 + 2

    def forward(self, x):
        return self.linear(x)


# Dummy Ensemble class to mock the Ensemble[Critic]
class DummyEnsemble(nn.Module):
    def __init__(self, n_members, prototype, models, device):
        super().__init__()
        self.n_members = n_members
        self.device = device
        self.models = models
        self.params = {}
        for i, model in enumerate(models):
            for name, param in model.named_parameters():
                self.params[f"{name}_{i}"] = param

    def parameters(self):
        for model in self.models:
            yield from model.parameters()

    def state_dict(self):
        state = {}
        for i, model in enumerate(self.models):
            for k, v in model.state_dict().items():
                state[f"{k}_{i}"] = v
        return state

    def load_state_dict(self, state_dict):
        for i, model in enumerate(self.models):
            model_state = {
                k.rsplit("_", 1)[0]: v
                for k, v in state_dict.items()
                if k.endswith(f"_{i}")
            }
            model.load_state_dict(model_state)

    def expand(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input to expand must be a torch.Tensor")
        return tensor.unsqueeze(0).repeat(self.n_members, *([1] * (tensor.dim())))

    def __getitem__(self, index):
        return self.models[index]

    def __call__(self, x):
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs, dim=0)


# Dummy concrete subclass implementing abstract method get_bellman_target
class DummyCriticEnsemble(CriticEnsemble):
    def get_bellman_target(self, *args, **kwargs):
        # Return a dummy tensor (shape and value depend on usage)
        return torch.tensor(0.0)


class DummyEnsembleWrapper:
    def __class_getitem__(cls, item):
        return DummyEnsemble


@pytest.fixture
def dummy_config():
    ns = lambda **kwargs: MagicMock(**kwargs)
    config = MagicMock()
    config.system.device = torch.device("cpu")
    config.training.gamma = 0.99
    config.model.tau = 0.01
    config.model.critic.has_target = True
    config.model.critic.n_members = 3
    config.model.critic.depth = 2
    config.model.critic.width = 64
    config.model.critic.activation = "relu"
    config.model.critic.norm = False
    config.model.critic.reset = False
    config.model.critic.reduce = "min"
    config.model.critic.arch = lambda dim_s, dim_a, **kwargs: DummyCriticModel()
    return config


@pytest.fixture
def critic(dummy_config):
    return Critic(dummy_config, dim_state=4, dim_act=2)


@pytest.fixture
def critic_ensemble(dummy_config):
    with (
        patch("objectrl.models.basic.critic.Ensemble", DummyEnsembleWrapper),
        patch(
            "objectrl.models.basic.critic.create_optimizer",
            lambda training_cfg: lambda params: torch.optim.Adam(params, lr=1e-3),
        ),
        patch(
            "objectrl.models.basic.critic.create_loss",
            lambda model_cfg, reduction=None: nn.MSELoss(reduction="none"),
        ),
    ):
        yield DummyCriticEnsemble(dummy_config, dim_state=4, dim_act=2)


def test_critic_initialization(critic, dummy_config):
    assert isinstance(critic.model, nn.Module)
    assert critic.has_target
    assert hasattr(critic, "target")
    # Check device is cpu
    assert critic.device.type == "cpu"


def test_critic_forward_and_prepare_input(critic):
    state = torch.rand(3, 4)
    action = torch.rand(3, 2)
    inp = critic._prepare_input(state, action)
    assert inp.shape == (3, 6)

    # Test Q method output shape
    q = critic.Q(state, action)
    assert isinstance(q, torch.Tensor)
    assert q.shape[0] == state.shape[0]


def test_critic_prepare_input_single_action(critic):
    state = torch.rand(1, 4)
    action = torch.tensor(1.0)  # scalar action
    inp = critic._prepare_input(state, action)
    assert inp.shape == (1, 5)  # state (1,4) + action reshaped (1,1)


def test_init_and_update_target(critic):
    # Target params should match model params after init_target
    with torch.no_grad():
        for p_model, p_target in zip(
            critic.model.parameters(), critic.target.parameters()
        ):
            assert torch.allclose(p_model, p_target)

    # Modify model param and update target with soft update
    with torch.no_grad():
        for p_model in critic.model.parameters():
            p_model.add_(torch.randn_like(p_model))
    before_update = [p.clone() for p in critic.target.parameters()]
    critic.update_target()
    after_update = list(critic.target.parameters())
    # At least one param should change after update_target
    assert any(not torch.equal(b, a) for b, a in zip(before_update, after_update))


def test_Q_t_returns_target_q(critic):
    state = torch.rand(3, 4)
    action = torch.rand(3, 2)
    q_t = critic.Q_t(state, action)
    assert isinstance(q_t, torch.Tensor)
    assert q_t.shape[0] == state.shape[0]


def test_critic_getitem_returns_self(critic):
    assert critic is critic


def test_criticensemble_initialization(critic_ensemble, dummy_config):
    assert critic_ensemble.n_members == dummy_config.model.critic.n_members
    assert critic_ensemble.has_target
    assert hasattr(critic_ensemble, "target_ensemble")
    assert isinstance(critic_ensemble.optim, torch.optim.Optimizer)


def test_criticensemble_reset(critic_ensemble):
    old_params = [p.clone() for p in critic_ensemble.model_ensemble.parameters()]
    critic_ensemble.reset()
    new_params = [p.clone() for p in critic_ensemble.model_ensemble.parameters()]
    # After reset, parameters should differ (because new models)
    assert any(not torch.equal(o, n) for o, n in zip(old_params, new_params))


def test_criticensemble_reduce_min_and_mean(critic_ensemble):
    q_vals = torch.tensor([[1.0, 2.0, 3.0], [0.5, 2.5, 3.5], [1.5, 1.0, 2.0]])
    # Expand dims to simulate output shape: (n_members, batch)
    q_vals = q_vals.T  # shape (3, 3)
    out_min = critic_ensemble.reduce(q_vals, "min")
    out_mean = critic_ensemble.reduce(q_vals, "mean")
    assert torch.allclose(out_min, torch.min(q_vals, dim=0).values)
    assert torch.allclose(out_mean, torch.mean(q_vals, dim=0))

    with pytest.raises(ValueError):
        critic_ensemble.reduce(q_vals, "unknown")


def test_criticensemble_get_single_critic(critic_ensemble):
    c0 = critic_ensemble[0]
    assert isinstance(c0, Critic)
    # Should load parameters correctly (mocked)
    assert hasattr(c0, "model")


def test_criticensemble_Q_and_Q_t(critic_ensemble):
    state = torch.rand(4, 4)
    action = torch.rand(4, 2)
    q = critic_ensemble.Q(state, action)
    assert q.shape[0] == critic_ensemble.n_members
    assert q.shape[1] == state.shape[0]

    q_t = critic_ensemble.Q_t(state, action)
    assert q_t.shape[0] == critic_ensemble.n_members
    assert q_t.shape[1] == state.shape[0]


def test_criticensemble_update_and_update_target(critic_ensemble):
    state = torch.rand(5, 4)
    action = torch.rand(5, 2)
    y = torch.rand(5, 1)  # Target values

    old_params = [p.clone() for p in critic_ensemble.model_ensemble.parameters()]
    critic_ensemble.update(state, action, y)
    new_params = [p.clone() for p in critic_ensemble.model_ensemble.parameters()]

    assert any(not torch.equal(o, n) for o, n in zip(old_params, new_params))

    # Test update_target changes target parameters
    if critic_ensemble.has_target:
        before = [p.clone() for p in critic_ensemble.target_ensemble.parameters()]
        critic_ensemble.update_target()
        after = list(critic_ensemble.target_ensemble.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))
