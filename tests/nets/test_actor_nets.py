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
