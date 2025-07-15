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

from typing import Any

import torch
from torch import nn as nn
from torch.distributions import (
    Categorical,
    Normal,
    TransformedDistribution,
)
from torch.distributions.transforms import TanhTransform


class GaussianHead(nn.Module):
    """
    Outputs a Gaussian distribution from a given input tensor.
    The input is split into mean and log-variance components which define
    a Normal distribution. The class returns the sampled action, its log-probability, and the distribution.

    Args:
        n (int): The dimensionality of the output action space.
    Attributes:
        _n (int): Dimensionality of the action space.
    """

    def __init__(self, n: int) -> None:
        """
        Initializes the Gaussian head with the specified action dimensionality.

        Args:
            n (int): Dimensionality of the action space.
        Returns:
            None
        """
        super().__init__()
        self._n = n

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Forward pass to compute the Gaussian distribution and sample an action.

        Args:
            x (Tensor): Input tensor containing concatenated mean and log-variance.
        Returns:
            dict: Dictionary containing the sampled action, its log-probability, mean, and distribution
        """
        mean = x[..., : self._n]
        logvar = x[..., self._n :].clamp(-18, 14)  # range approx [1e8,1e6]
        std = logvar.exp()
        dist = Normal(mean, std, validate_args=False)
        y = dist.rsample()
        y_logprob = dist.log_prob(y).sum(dim=-1, keepdim=True)
        return_dict = {
            "action": y,
            "action_logprob": y_logprob,
            "mean": mean,
            "dist": dist,
        }
        return return_dict


class SquashedGaussianHead(nn.Module):
    """
    Outputs a Tanh-squashed Gaussian distribution, commonly used for bounded actions in reinforcement learning.

    Args:
        n (int): Dimensionality of the action space.
        upper_clamp (float): Upper clamp on log-variance values.
        n_samples (int): Number of samples used in evaluation mode.
        Attributes:
            _n (int): Dimensionality of the action space.
            _upper_clamp (float): Upper clamp for log-variance.
            _n_samples (int): Number of samples for evaluation.
    """

    def __init__(self, n: int, upper_clamp: float = -2.0, n_samples: int = 100) -> None:
        super().__init__()
        self._n = n
        self._upper_clamp = upper_clamp
        self._n_samples = n_samples

    def forward(self, x: torch.Tensor, is_training: float = True) -> dict[str, Any]:
        """
        Forward pass producing a squashed Gaussian distribution.

        Args:
            x (Tensor): Input tensor with mean and log-variance concatenated.
            is_training (bool): Whether in training mode (stochastic sampling) or not.
        Returns:
            dict: Dictionary containing the sampled action, its log-probability, and the distribution.
        """
        mean_bt = x[..., : self._n]
        log_std_bt = x[..., self._n :].clamp(-18, 4.6)  # range [1e-8, 100]
        std_bt = log_std_bt.exp()
        dist_bt = Normal(mean_bt, std_bt)
        transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist_bt, transform)

        return_dict = {
            "dist": dist,
        }
        if is_training:
            y = dist.rsample()
            y_logprob = dist.log_prob(y).sum(dim=-1, keepdim=True)
            return_dict["action_logprob"] = y_logprob
        else:
            y_samples = dist.rsample((self._n_samples,))
            y = y_samples.mean(dim=0)

        return_dict["action"] = y
        return return_dict


class CategoricalHead(nn.Module):
    """
    Outputs a Categorical distribution for discrete action spaces.

    Args:
        n (int): Number of categories (discrete actions).
    Returns:
            dict: Dictionary containing the sampled action, its log-probability, and the distribution.
    """

    def __init__(self, n: int) -> None:
        """
        Initializes the Categorical head with the specified number of categories.

        Args:
            n (int): Number of categories (discrete actions).
        Returns:
            None
        """
        super().__init__()
        self._n = n

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Forward pass to compute categorical distribution and sample an action.

        Args:
            x (Tensor): Input logits for categorical distribution.
        Returns:
            dict: Dictionary containing the sampled action, its log-probability, and the distribution.
        """
        logit = x
        probs = nn.functional.softmax(logit, dim=-1)
        dist = Categorical(probs, validate_args=False)
        y = dist.sample()
        y_logprob = dist.log_prob(y).unsqueeze(-1)
        return_dict = {
            "action": y,
            "action_logprob": y_logprob,
            "dist": dist,
        }
        return return_dict


class DeterministicHead(nn.Module):
    """
    Pass-through head for deterministic outputs.

    Args:
        n (int): Dimensionality of the output.
    Returns:
            dict: Dictionary containing the action.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self._n = n

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Returns the input as the action.

        Args:
            x (Tensor): Input tensor.
        Returns:
            dict: Dictionary containing the action.
        """
        y = x
        return_dict = {
            "action": y,
        }
        return return_dict
