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
from objectrl.nets.layers.heads import (
    GaussianHead,
    SquashedGaussianHead,
    CategoricalHead,
    DeterministicHead,
)


def test_gaussian_head():
    head = GaussianHead(n=3)
    x = torch.randn(5, 6)
    out = head(x)
    assert out["action"].shape == (5, 3)
    assert out["action_logprob"].shape == (5, 1)
    assert out["mean"].shape == (5, 3)
    assert hasattr(out["dist"], "rsample")


def test_squashed_gaussian_head_training():
    head = SquashedGaussianHead(n=2)
    x = torch.randn(4, 4)
    out = head(x, is_training=True)
    assert out["action"].shape == (4, 2)
    assert out["action_logprob"].shape == (4, 1)
    assert hasattr(out["dist"], "rsample")


def test_squashed_gaussian_head_eval():
    head = SquashedGaussianHead(n=2)
    x = torch.randn(4, 4)
    out = head(x, is_training=False)
    assert out["action"].shape == (4, 2)
    assert "action_logprob" not in out


def test_categorical_head():
    head = CategoricalHead(n=4)
    x = torch.randn(6, 4)
    out = head(x)
    assert out["action"].shape == (6,)
    assert out["action_logprob"].shape == (6, 1)
    assert hasattr(out["dist"], "sample")


def test_deterministic_head():
    head = DeterministicHead(n=3)
    x = torch.randn(10, 3)
    out = head(x)
    assert torch.allclose(out["action"], x)
