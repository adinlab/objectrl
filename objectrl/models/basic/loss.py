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

import typing
from abc import ABC, abstractmethod

import torch
from torch.nn.modules.loss import _Loss

if typing.TYPE_CHECKING:
    from objectrl.config.model_configs.pbac import PBACConfig


class ProbabilisticLoss(_Loss, ABC):
    """
    Base class for probabilistic loss functions.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. Default: 'mean'.
    Attributes:
        reduction (str): Reduction method for the loss.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.reduction = reduction

    @abstractmethod
    def forward(self, mu_lvar_dict: dict, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute loss (to be implemented in subclasses).

        Args:
            mu_lvar_dict (dict): Predicted mean and log_variance tensors
            y (Tensor): Target tensor.
        Returns:
            Tensor: Computed loss.
        """
        pass

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified reduction to the loss tensor.

        Args:
            loss: Tensor of loss values.
        Returns:
            Tensor: Reduced loss tensor based on the specified reduction method.
        Raises:
            ValueError: If an unknown reduction method is specified.
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class PACBayesLoss(ProbabilisticLoss):
    """
    PAC-Bayesian loss combining empirical risk and complexity term.

    Args:
        config: Configuration object with loss parameters:
        - lossparams.reduction (str): Reduction method.
        - lossparams.logvar_lower_clamp (float): Lower clamp for log variance.
        - lossparams.logvar_upper_clamp (float): Upper clamp for log variance.
        - lossparams.complexity_coef (float): Coefficient for complexity term.
    Attributes:
        logvar_lower_clamp (float): Lower clamp for log variance.
        logvar_upper_clamp (float): Upper clamp for log variance.
        complexity_coef (float): Coefficient for complexity term.
    """

    def __init__(self, config: "PBACConfig"):
        """
        Initialize PACBayesLoss with configuration parameters.
        """
        super().__init__(reduction=config.lossparams.reduction)
        self.logvar_lower_clamp = config.lossparams.logvar_lower_clamp
        self.logvar_upper_clamp = config.lossparams.logvar_upper_clamp
        self.complexity_coef = config.lossparams.complexity_coef

    def forward(self, mu_lvar_dict: dict, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the PAC-Bayes loss.

        Args:
            mu_lvar_dict (dict): Dictionary with keys "mu" (mean) and "lvar" (log variance) tensors.
            y (Tensor): Target tensor with shape [..., 2], where last dimension holds
            true mean and true variance (mu_t, sig2_t).
        Returns:
            Tensor: Computed PAC-Bayes loss.
        """
        mu_t = y[:, :, 0]
        sig2_t = y[:, :, 1]
        mu, logvar = mu_lvar_dict["mu"], mu_lvar_dict["lvar"]
        sig2 = logvar.exp().clamp(self.logvar_lower_clamp, self.logvar_upper_clamp)

        # KL divergence term between predicted and true distributions
        sig_ratio = sig2 / sig2_t
        kl_vals = 0.5 * (sig_ratio - sig_ratio.log() + (mu - mu_t) ** 2 / sig2_t - 1)

        # Empirical risk (expected squared error plus predicted variance)
        empirical_risk = ((mu - mu_t) ** 2 + sig2).mean(-1)

        # Complexity regularization scaled by coefficient
        complexity = kl_vals.mean(-1) * self.complexity_coef

        q_loss = empirical_risk + complexity
        return q_loss.sum()
