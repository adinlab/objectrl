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
import pytest

from objectrl.nets.critic_nets import (
    CriticNet,
    ValueNet,
    CriticNetProbabilistic,
    BNNCriticNet,
    EMstyle,
    DQNNet,
)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_critic_net_forward(batch_size):
    dim_state = 5
    dim_act = 3
    model = CriticNet(dim_state, dim_act)
    x = torch.randn(batch_size, dim_state + dim_act)
    out = model(x)
    assert out.shape == (batch_size, 1)
    assert torch.is_tensor(out)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_value_net_forward(batch_size):
    dim_state = 7
    dim_act = 0  # ignored
    model = ValueNet(dim_state, dim_act)
    x = torch.randn(batch_size, dim_state)
    out = model(x)
    assert out.shape == (batch_size, 1)
    assert torch.is_tensor(out)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_critic_net_probabilistic_forward(batch_size):
    dim_state = 5
    dim_act = 3
    model = CriticNetProbabilistic(dim_state, dim_act)
    x = torch.randn(batch_size, dim_state + dim_act)
    out = model(x)
    assert out.shape == (batch_size, 2)  # probabilistic outputs mean+var or similar
    assert torch.is_tensor(out)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_bnn_critic_net_forward_and_map(batch_size):
    dim_state = 5
    dim_act = 3
    model = BNNCriticNet(dim_state, dim_act)
    x = torch.randn(batch_size, dim_state + dim_act)

    out = model(x)
    assert torch.is_tensor(out) or (
        isinstance(out, tuple)
        and torch.is_tensor(out[0])
        and (out[1] is None or torch.is_tensor(out[1]))
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_emstyle_forward(batch_size):
    dim_state = 5
    dim_act = 3
    width = 10
    model = EMstyle(dim_state, dim_act, width=width)
    x = torch.randn(batch_size, dim_state + dim_act)
    out = model(x)
    assert out.shape == (batch_size, width)
    assert torch.is_tensor(out)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_dqn_net_forward(batch_size):
    dim_state = 5
    dim_act = 3
    model = DQNNet(dim_state, dim_act)
    x = torch.randn(batch_size, dim_state)
    out = model(x)
    assert out.shape == (batch_size, dim_act)
    assert torch.is_tensor(out)
