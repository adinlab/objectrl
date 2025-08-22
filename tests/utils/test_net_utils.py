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

import importlib
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

import objectrl.utils.net_utils as net_utils
from objectrl.nets.layers.bayesian_layers import (
    BBBLinear,
    LRLinear,
    CLTLinear,
    CLTLinearDet,
)
from objectrl.utils.custom_act import CReLU


class DummyConfig:
    def __init__(self, optimizer="Adam", learning_rate=1e-3, loss="MSELoss"):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss


def test_create_optimizer():
    config = DummyConfig(optimizer="Adam", learning_rate=0.01)
    opt_fn = net_utils.create_optimizer(config)
    model = nn.Linear(2, 2)
    opt = opt_fn(model.parameters())
    assert isinstance(opt, optim.Adam)
    assert opt.param_groups[0]["lr"] == 0.01

    config.optimizer = "SGD"
    opt_fn = net_utils.create_optimizer(config)
    opt = opt_fn(model.parameters())
    assert isinstance(opt, optim.SGD)

    config.optimizer = "NonExistentOpt"
    with pytest.raises(NotImplementedError):
        net_utils.create_optimizer(config)


def test_create_loss_torch_and_custom(monkeypatch):
    config = DummyConfig(loss="MSELoss")
    loss_fn = net_utils.create_loss(config)
    assert isinstance(loss_fn, nn.MSELoss)

    class CustomLoss(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def forward(self, x):
            return x

    dummy_loss_module = type("dummy_loss_module", (), {"CustomLoss": CustomLoss})

    monkeypatch.setattr(
        net_utils.importlib,
        "import_module",
        lambda name: (
            dummy_loss_module
            if name == "objectrl.models.basic.loss"
            else importlib.import_module(name)
        ),
    )

    config.loss = "CustomLoss"
    loss_fn = net_utils.create_loss(config)
    assert isinstance(loss_fn, CustomLoss)

    config.loss = "UnknownLoss"
    with pytest.raises(NotImplementedError):
        net_utils.create_loss(config)


@pytest.mark.parametrize(
    "act, width_multiplier, act_class",
    [
        ("relu", 1, nn.ReLU),
        ("crelu", 2, CReLU),
    ],
)
def test_mlp_structure_and_forward(act, width_multiplier, act_class):
    dim_in, dim_out, depth, width = 4, 3, 2, 5
    mlp = net_utils.MLP(dim_in, dim_out, depth, width, act=act, has_norm=True)

    assert any(isinstance(l, act_class) for l in mlp.model)

    x = torch.randn(2, dim_in)
    y = mlp(x)
    assert y.shape == (2, dim_out)


@pytest.mark.parametrize(
    "layer_type, expected_class",
    [
        ("bbb", BBBLinear),
        ("lr", LRLinear),
        ("cltdet", CLTLinearDet),
    ],
)
@pytest.mark.parametrize(
    "act, width_multiplier, act_class",
    [
        ("relu", 1, nn.ReLU),
        ("crelu", 2, CReLU),
    ],
)
def test_bayesian_mlp_forward_and_structure(
    layer_type, expected_class, act, width_multiplier, act_class
):
    dim_in, dim_out, width = 4, 3, 5
    has_norm = False

    depth = 1 if layer_type == "cltdet" else 3

    mlp = net_utils.BayesianMLP(
        dim_in, dim_out, depth, width, layer_type=layer_type, act=act, has_norm=has_norm
    )

    found_layers = [l for l in mlp.model if isinstance(l, expected_class)]
    assert len(found_layers) >= 1

    batch_size = 2
    x = torch.randn(batch_size, dim_in)

    if layer_type == "cltdet":
        output = mlp(x)
        if isinstance(output, tuple):
            output = output[0]
        assert isinstance(output, torch.Tensor), "Output should be a Tensor for cltdet"
        assert output.shape[0] == batch_size
    else:
        output = mlp(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size


def test_mlp_invalid_depth_and_activation():
    with pytest.raises(AssertionError):
        net_utils.MLP(4, 2, 0, 5)

    with pytest.raises(NotImplementedError):
        net_utils.MLP(4, 2, 1, 5, act="unknown_act")


def test_bayesian_mlp_invalid_depth_and_activation():
    with pytest.raises(AssertionError):
        net_utils.BayesianMLP(4, 2, 0, 5)

    with pytest.raises(NotImplementedError):
        net_utils.BayesianMLP(4, 2, 1, 5, layer_type="bbb", act="unknown_act")
