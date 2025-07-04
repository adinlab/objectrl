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
