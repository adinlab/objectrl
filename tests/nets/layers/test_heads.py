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
