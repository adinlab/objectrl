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
from torch.nn.modules.loss import _Loss

from objectrl.models.basic.loss import ProbabilisticLoss, PACBayesLoss


class DummyConfig:
    class lossparams:
        reduction = "mean"
        logvar_lower_clamp = 1e-6
        logvar_upper_clamp = 1.0
        complexity_coef = 0.5


def test_probabilistic_loss_apply_reduction():
    class TestLoss(ProbabilisticLoss):
        def forward(self, mu, logvar, y):
            return torch.tensor([1.0, 2.0, 3.0])

    # Test mean reduction
    loss_fn = TestLoss(reduction="mean")
    loss = loss_fn._apply_reduction(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(loss, torch.tensor(2.0))

    # Test sum reduction
    loss_fn = TestLoss(reduction="sum")
    loss = loss_fn._apply_reduction(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(loss, torch.tensor(6.0))

    # Test none reduction
    loss_fn = TestLoss(reduction="none")
    loss = loss_fn._apply_reduction(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(loss, torch.tensor([1.0, 2.0, 3.0]))

    # Test invalid reduction
    loss_fn = TestLoss(reduction="invalid")
    with pytest.raises(ValueError):
        loss_fn._apply_reduction(torch.tensor([1.0, 2.0, 3.0]))


def test_pacbayes_loss_forward():
    config = DummyConfig()
    loss_fn = PACBayesLoss(config)

    batch_size = 2
    seq_len = 3

    # mu and logvar predicted by model
    mu = torch.randn(batch_size, seq_len)
    logvar = torch.randn(batch_size, seq_len).clamp(-3, 3)

    # y contains true mean and true variance
    mu_t = torch.randn(batch_size, seq_len)
    sig2_t = torch.rand(batch_size, seq_len).clamp(0.1, 2.0)  # positive variance

    y = torch.stack([mu_t, sig2_t], dim=-1)  # shape: (batch_size, seq_len, 2)

    mu_lvar_dict = {"mu": mu, "lvar": logvar}

    loss = loss_fn(mu_lvar_dict, y)

    # Check loss is scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    # Check loss is positive
    assert loss.item() >= 0


def test_pacbayes_loss_clamping_behavior():
    config = DummyConfig()
    loss_fn = PACBayesLoss(config)

    batch_size, seq_len = 1, 2

    mu = torch.zeros(batch_size, seq_len)
    # logvar values below and above clamps
    logvar = torch.tensor([[-10.0, 10.0]])

    mu_t = torch.zeros(batch_size, seq_len)
    sig2_t = torch.ones(batch_size, seq_len)  # true variance = 1

    y = torch.stack([mu_t, sig2_t], dim=-1)

    mu_lvar_dict = {"mu": mu, "lvar": logvar}

    loss = loss_fn(mu_lvar_dict, y)

    # Manually compute clamped variances
    clamped_sig2 = logvar.exp().clamp(
        loss_fn.logvar_lower_clamp, loss_fn.logvar_upper_clamp
    )
    assert torch.all(clamped_sig2 >= loss_fn.logvar_lower_clamp)
    assert torch.all(clamped_sig2 <= loss_fn.logvar_upper_clamp)

    assert loss.item() >= 0
