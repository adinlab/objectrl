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

from objectrl.nets.actor_nets import ActorNetProbabilistic, ActorNet


@pytest.mark.parametrize("n_heads", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_actor_net_probabilistic_forward_shapes(n_heads, batch_size):
    dim_state = 8
    dim_act = 4
    model = ActorNetProbabilistic(dim_state, dim_act, n_heads=n_heads)

    x = torch.randn(batch_size, dim_state)

    out_train = model(x, is_training=True)
    assert "action" in out_train
    assert "action_logprob" in out_train
    assert "dist" in out_train

    action = out_train["action"]
    expected_shape = (batch_size, dim_act)
    if n_heads > 1:
        expected_shape = (batch_size, n_heads, dim_act)
    assert action.shape == expected_shape

    out_eval = model(x, is_training=False)
    assert "action" in out_eval
    assert "dist" in out_eval
    action_eval = out_eval["action"]
    assert action_eval.shape == expected_shape


@pytest.mark.parametrize("n_heads", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_actor_net_deterministic_forward(n_heads, batch_size):
    dim_state = 6
    dim_act = 2
    model = ActorNet(dim_state, dim_act, n_heads=n_heads)

    x = torch.randn(batch_size, dim_state)
    out = model(x, is_training=True)

    assert "action" in out
    action = out["action"]

    expected_shape = (batch_size, dim_act)
    if n_heads > 1:
        expected_shape = (batch_size, n_heads, dim_act)
    assert action.shape == expected_shape


def test_actor_net_probabilistic_custom_params():
    model = ActorNetProbabilistic(
        dim_state=3,
        dim_act=2,
        n_heads=2,
        depth=2,
        width=16,
        act="relu",
        has_norm=True,
        upper_clamp=-1.5,
    )
    x = torch.randn(4, 3)
    out = model(x)
    assert "action" in out
    assert out["action"].shape == (4, 2, 2)


def test_actor_net_custom_params():
    model = ActorNet(
        dim_state=3,
        dim_act=2,
        n_heads=2,
        depth=2,
        width=16,
        act="relu",
        has_norm=True,
    )
    x = torch.randn(4, 3)
    out = model(x)
    assert "action" in out
    assert out["action"].shape == (4, 2, 2)
