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
