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

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn as nn

from objectrl.utils.custom_act import CReLU


class BayesianLinear(ABC, nn.Module):
    """
    Abstract base class for Bayesian neural network layers.

    Attributes:
        use_softplus (bool): Whether to apply softplus to std dev parameters.
        _manual_reset(bool): If True, keep the random state
        weight_mu (nn.Parameter): Mean of the weight distribution.
        weight_rho (nn.Parameter): Rho (transformed std) of the weight distribution.
        bias_mu (nn.Parameter | None): Mean of the bias distribution (if bias=True).
        bias_rho (nn.Parameter | None): Rho of the bias distribution (if bias=True).
        prior_mean (torch.Tensor | None): Mean of the prior distribution.
        prior_std (torch.Tensor | None): Standard deviation of the prior distribution.
    """

    in_features: int
    out_features: int
    _map: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_mean: float | torch.Tensor | None = None,
        prior_std: float | torch.Tensor | None = None,
        use_softplus: bool = False,
        manual_reset: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            in_features (int): Size of input features.
            out_features (int): Size of output features.
            bias (bool): Whether to include a bias term.
            prior_mean (float or torch.Tensor, optional): Prior mean.
            prior_std (float or torch.Tensor, optional): Prior std deviation.
            use_softplus (bool): If True, apply softplus to std parameters.
            manual_reset(bool): If True, keep the random state
            device (torch.device, optional): Device to use.
            dtype (torch.dtype, optional): Data type to use.
        Returns:
            None
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_softplus = use_softplus
        self._manual_reset = manual_reset
        self._random_weight = None
        self._random_bias = None

        self.weight_mu = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight_rho = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias_mu = nn.Parameter(torch.empty((out_features,), **factory_kwargs))
            self.bias_rho = nn.Parameter(torch.empty((out_features,), **factory_kwargs))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        # Register prior mean and std as buffers to track their device usage
        if prior_mean is None:
            prior_mean = torch.zeros(1, **factory_kwargs)
        elif isinstance(prior_mean, float):
            prior_mean = torch.tensor(prior_mean, **factory_kwargs)
        elif isinstance(prior_mean, torch.Tensor):
            assert prior_mean.shape == (1,) or prior_mean.shape == (
                out_features,
                in_features,
            ), "Prior mean needs to be either a scalar or the same shape of the weights"
            prior_mean = prior_mean.to(**factory_kwargs)
        else:
            raise ValueError("Prior mean needs to be a float or a torch.Tensor")
        self.register_buffer("prior_mean", prior_mean)

        if prior_std is None:
            prior_std = torch.ones(1, **factory_kwargs)
        elif isinstance(prior_std, float):
            prior_std = torch.tensor(prior_std, **factory_kwargs)
        elif isinstance(prior_std, torch.Tensor):
            assert prior_std.shape == (1,) or prior_std.shape == (
                out_features,
                in_features,
            )
            prior_std = prior_std.to(**factory_kwargs)
        else:
            raise ValueError("Prior std needs to be a float or a torch.Tensor")
        self.register_buffer("prior_std", prior_std)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -4.6)
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -4.6)

    def reset_randomness(self) -> None:
        assert self._manual_reset, "Manuel reset needs to be true"
        self._random_weight = None
        self._random_bias = None

    def set_manual_reset(self, manual_reset: bool = True) -> None:
        self._manual_reset = manual_reset

    def get_manual_reset(self) -> bool:
        return self._manual_reset

    def map(self, on: bool = True):
        """Switch maximum a posteriori (MAP) on or off

        Args:
            on (bool): If True, sets MAP mode on.
        Returns:
            None
        """
        self._map = on

    def update_prior(self, prior_mean: torch.Tensor, prior_std: torch.Tensor) -> None:
        self.prior_mean = prior_mean.to(self.weight_mu.device)
        self.prior_std = prior_std.to(self.weight_mu.device)
        self.prior_mean.requires_grad_(False)
        self.prior_std.requires_grad_(False)

    @staticmethod
    def inv_softplus(x: torch.Tensor) -> torch.Tensor:
        """Inverse of the softplus function.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Inverse softplus tensor."""
        return x + torch.log(-torch.expm1(-x))

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        """Softplus activation function.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Softplus tensor.
        """
        return torch.log(1 + torch.exp(x))

    def mean(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            tuple: Mean of the weight distribution and optionally bias distribution.
        """
        return self.weight_mu, self.bias_mu

    def std(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            tuple: Standard deviation of the weight distribution and optionally bias distribution.
        """
        if self.use_softplus:
            return self.softplus(self.weight_rho), (
                self.softplus(self.bias_rho) if self.bias_rho is not None else None
            )
        else:
            return torch.exp(self.weight_rho), (
                torch.exp(self.bias_rho) if self.bias_rho is not None else None
            )

    def var(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            tuple: Variance of the weight distribution and optionally bias distribution.
        """
        weight, bias = self.std()
        return weight.pow(2), bias.pow(2) if bias is not None else None

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the layer.
        """
        pass

    def KL(self) -> tuple[torch.Tensor, int]:
        """
        Computes the KL divergence between posterior and prior.

        Args:
            None
        Returns:
            Tuple[torch.Tensor, int]: KL divergence and number of parameters.
        """
        weight_mu, bias_mu = self.mean()
        weight_std, bias_std = self.std()

        kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(weight_mu, weight_std),
            torch.distributions.Normal(self.prior_mean, self.prior_std),
        ).sum()
        if bias_mu is not None:
            kl += torch.distributions.kl_divergence(
                torch.distributions.Normal(bias_mu, bias_std),
                torch.distributions.Normal(self.prior_mean, self.prior_std),
            ).sum()
        return (
            kl,
            sum(weight_mu.shape) + (sum(bias_mu.shape) if bias_mu is not None else 0),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features})"


class BBBLinear(BayesianLinear):
    """
    Implements a Bayesian Layer following Bayes by Backprop (Blundell et al., 2015)
    Samples weights and biases during the forward pass from the learned posterior
    distribution. In MAP mode, only the means are used.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        prior_mean (float | torch.Tensor | None): Prior mean for weights.
        prior_std (float | torch.Tensor | None): Prior standard deviation for weights.
        use_softplus (bool): Whether to apply softplus activation to std parameters.
        manual_reset(bool): If True, keep the random state
        device (torch.device, optional): Device to use for the layer.
        dtype (torch.dtype, optional): Data type for the layer parameters.
    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        use_softplus (bool): Whether to apply softplus to std dev parameters.
        _manual_reset(bool): If True, keep the random state
        weight_mu (nn.Parameter): Mean of the weight distribution.
        weight_rho (nn.Parameter): Rho (transformed std) of the weight distribution.
        bias_mu (nn.Parameter | None): Mean of the bias distribution (if bias=True).
        bias_rho (nn.Parameter | None): Rho of the bias distribution (if bias=True).
        prior_mean (torch.Tensor | None): Mean of the prior distribution.
        prior_std (torch.Tensor | None): Standard deviation of the prior distribution.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            input (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the layer.
        """
        weight_mu, bias_mu = self.mean()
        weight_std, bias_std = self.std()

        if self._map:
            weight = weight_mu
        else:
            if not self._manual_reset or self._random_weight is None:
                self._random_weight = torch.randn_like(weight_mu)

            weight = weight_mu + weight_std * self._random_weight
        if bias_mu is not None:
            if self._map:
                bias = bias_mu
            else:
                if not self._manual_reset or self._random_bias is None:
                    self._random_bias = torch.randn_like(bias_mu)
                bias = bias_mu + bias_std * self._random_bias
        else:
            bias = None

        return F.linear(input, weight, bias)


class LRLinear(BayesianLinear):
    """
    Implements a Bayesian layer using a local reparameterization trick (Kingma et al., 2015).
    Instead of sampling weights, it samples output activations using propagated mean and variance.
    More efficient and less noisy than direct weight sampling.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        prior_mean (float | torch.Tensor | None): Prior mean for weights.
        prior_std (float | torch.Tensor | None): Prior standard deviation for weights.
        use_softplus (bool): Whether to apply softplus activation to std parameters.
        device (torch.device, optional): Device to use for the layer.
        dtype (torch.dtype, optional): Data type for the layer parameters.
    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        use_softplus (bool): Whether to apply softplus to std dev parameters.
        weight_mu (nn.Parameter): Mean of the weight distribution.
        weight_rho (nn.Parameter): Rho (transformed std) of the weight distribution.
        bias_mu (nn.Parameter | None): Mean of the bias distribution (if bias=True).
        bias_rho (nn.Parameter | None): Rho of the bias distribution (if bias=True).
        prior_mean (torch.Tensor | None): Mean of the prior distribution.
        prior_std (torch.Tensor | None): Standard deviation of the prior distribution.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer using local reparameterization trick.

        Args:
            input (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the layer.
        """
        weight_mu, bias_mu = self.mean()
        weight_std, bias_std = self.std()

        out_mu = F.linear(input, weight_mu, bias_mu)

        if self._map:
            return out_mu

        out_std = torch.sqrt(
            F.linear(
                input.pow(2),
                weight_std**2,
                (bias_std**2 if bias_std is not None else None),
            )
        )

        if not self._manual_reset or self._random_weight is None:
            self._random_weight = torch.randn_like(out_mu)

        return out_mu + out_std * self._random_weight


class CLTLinear(BayesianLinear):
    """
    Implements a Bayesian layer using a central limit theorem (Wu et al., 2019; Haussmann, 2021).
    Supports ReLU and CReLU activations. During forward pass, propagates mean
    and variance analytically instead of sampling.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        prior_mean (float | torch.Tensor | None): Prior mean for weights.
        prior_std (float | torch.Tensor | None): Prior standard deviation for weights.
        use_softplus (bool): Whether to apply softplus activation to std parameters.
        device (torch.device, optional): Device to use for the layer.
        dtype (torch.dtype, optional): Data type for the layer parameters.
    Attributes:
        act (str): Activation type ('relu' or 'crelu').
        is_input (bool): Whether this is the input layer.
        is_output (bool): Whether this is the output layer.
    """

    def __init__(
        self,
        *args,
        act: Literal["relu", "crelu"] = "relu",
        is_input: bool = False,
        is_output: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the CLTLinear layer.

        Args:
            act (Literal["relu", "crelu"]): Activation function to use ('relu' or 'crelu').
            is_input (bool): Whether this is the input layer.
            is_output (bool): Whether this is the output layer.
        Returns:
            None
        """
        super().__init__(*args, **kwargs)
        if act not in ["relu", "crelu"]:
            raise NotImplementedError(
                f"{act} is not implemented. Needs to be 'relu' or 'crelu'."
            )
        self.act = act
        self.is_input = is_input
        self.is_output = is_output

    def reset_randomness(self) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement reset_randomness as it is analytical."
        )

    @staticmethod
    def normal_cdf(
        x, mu: float | torch.Tensor = 0.0, sigma: float | torch.Tensor = 1.0
    ) -> torch.Tensor:
        """
        Computes the cumulative distribution function (CDF) of a normal distribution.

        Args:
            x (torch.Tensor): Input tensor.
            mu (float or torch.Tensor): Mean of the normal distribution.
            sigma (float or torch.Tensor): Standard deviation of the normal distribution.
        Returns:
            torch.Tensor: CDF values for the input tensor.
        """
        return 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))

    @staticmethod
    def normal_pdf(
        x, mu: float | torch.Tensor = 0.0, sigma: float | torch.Tensor = 1.0
    ) -> torch.Tensor:
        """
        Computes the probability density function (PDF) of a normal distribution.

        Args:
            x (torch.Tensor): Input tensor.
            mu (float or torch.Tensor): Mean of the normal distribution.
            sigma (float or torch.Tensor): Standard deviation of the normal distribution.
        Returns:
            torch.Tensor: PDF values for the input tensor.
        """

        return (1 / (math.sqrt(2 * math.pi) * sigma)) * torch.exp(
            -0.5 * ((x - mu) / sigma).pow(2)
        )

    @staticmethod
    def relu_moments(
        mu: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the mean and variance of the ReLU activation function.

        Args:
            mu (torch.Tensor): Mean of the input tensor.
            sigma (torch.Tensor): Standard deviation of the input tensor.
        Returns:
            tuple: Mean and variance of the ReLU activation.
        """
        alpha = mu / sigma
        cdf = CLTLinear.normal_cdf(alpha)
        pdf = CLTLinear.normal_pdf(alpha)
        relu_mean = mu * cdf + sigma * pdf
        relu_var = (
            (sigma.pow(2) + mu.pow(2)) * cdf + mu * sigma * pdf - relu_mean.pow(2)
        )
        relu_mean[sigma.eq(0)] = mu[sigma.eq(0)] * (mu[sigma.eq(0)] > 0)
        relu_var[sigma.eq(0)] = 0.0
        return relu_mean, relu_var

    @staticmethod
    def neg_relu_moments(
        mu: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the mean and variance of the negative ReLU activation function.

        Args:
            mu (torch.Tensor): Mean of the input tensor.
            sigma (torch.Tensor): Standard deviation of the input tensor.
        Returns:
            tuple: Mean and variance of the negative ReLU activation.
        """
        # compute mean and variance of relu(-x)
        alpha = mu / sigma
        cdf = CLTLinear.normal_cdf(alpha)
        pdf = CLTLinear.normal_pdf(alpha)
        neg_relu_mean = mu * cdf + sigma * pdf - mu
        neg_relu_var = (
            (mu.pow(2) + sigma.pow(2)) * (1 - cdf)
            - mu * sigma * pdf
            - neg_relu_mean.pow(2)
        )
        neg_relu_mean[sigma.eq(0)] = -mu[sigma.eq(0)] * (mu[sigma.eq(0)] < 0)
        neg_relu_var[sigma.eq(0)] = 0.0
        return neg_relu_mean, neg_relu_var

    @staticmethod
    def crelu_moments(
        mu: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the mean and variance of the CReLU activation function.

        Args:
            mu (torch.Tensor): Mean of the input tensor.
            sigma (torch.Tensor): Standard deviation of the input tensor.
        Returns:
            tuple: Mean and variance of the CReLU activation.
        """
        # [relu(x), relu(-x)]
        relu_mean, relu_var = CLTLinear.relu_moments(mu, sigma)
        neg_relu_mean, neg_relu_var = CLTLinear.neg_relu_moments(mu, sigma)
        return torch.cat((relu_mean, neg_relu_mean), -1), torch.cat(
            (relu_var, neg_relu_var), -1
        )

    def forward(
        self, mu_h: torch.Tensor, var_h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            mu_h (torch.Tensor): Input mean tensor.
            var_h (torch.Tensor, optional): Input variance tensor.
        Returns:
            tuple: Output mean tensor and optionally output variance tensor.
        """
        if self._map:
            assert (
                var_h is None
            ), "Input variance has to be None for maximum a posteriori forward pass."

        if var_h is None and not self._map:
            assert (
                self.is_input
            ), "Input variance is required for non-input layers if not in MAP mode."

        weight_mu, bias_mu = self.mean()
        weight_std, bias_std = self.std()

        mu_f = F.linear(mu_h, weight_mu, bias_mu)

        if self._map:
            if self.is_output:
                return mu_f, None
            else:
                if self.act == "relu":
                    return nn.ReLU()(mu_f), None
                else:
                    return CReLU()(mu_f), None

        if self.is_input:
            var_f = F.linear(
                mu_h.pow(2),
                weight_std**2,
                bias_std**2 if bias_std is not None else None,
            )
        else:
            assert var_h is not None
            var_f = F.linear(
                var_h + mu_h.pow(2),
                weight_std**2,
                bias_std**2 if bias_std is not None else None,
            ) + F.linear(var_h, weight_mu**2)

        if self.is_output:
            return mu_f, var_f
        else:
            if self.act == "relu":
                return self.relu_moments(mu_f, torch.sqrt(var_f))
            else:
                return self.crelu_moments(mu_f, torch.sqrt(var_f))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features} ({self.is_input = }, {self.is_output = }, act: {self.act})"


class CLTLinearDet(CLTLinear):
    """
    Deterministic version of CLTLinear.
    Disables uncertainty modeling by removing the learned standard deviation.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        prior_mean (float | torch.Tensor | None): Prior mean for weights.
        prior_std (float | torch.Tensor | None): Prior standard deviation for weights.
        use_softplus (bool): Whether to apply softplus activation to std parameters.
        device (torch.device, optional): Device to use for the layer.
        dtype (torch.dtype, optional): Data type for the layer parameters.
    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        use_softplus (bool): Whether to apply softplus to std dev parameters.
        weight_mu (nn.Parameter): Mean of the weight distribution.
        weight_rho (nn.Parameter): Rho (transformed std) of the weight distribution.
        bias_mu (nn.Parameter | None): Mean of the bias distribution (if bias=True).
        bias_rho (nn.Parameter | None): Rho of the bias distribution (if bias=True).
        prior_mean (torch.Tensor | None): Mean of the prior distribution.
        prior_std (torch.Tensor | None): Standard deviation of the prior distribution.
    """

    def __init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_rho = None
        self.bias_rho = None

    def std(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            tuple: Standard deviation of the weight distribution and None for bias.
        """
        raise ValueError("This layer has no standard deviation")

    def KL(self) -> torch.Tensor:
        """
        Computes the KL divergence for this layer.

        Args:
            None
        Returns:
            torch.Tensor: KL divergence of the layer.
        """
        raise ValueError("This layer has no KL divergence")

    def forward(
        self, mu_h: torch.Tensor, var_h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            mu_h (torch.Tensor): Input mean tensor.
            var_h (torch.Tensor, optional): Input variance tensor.
        Returns:
            tuple: Output mean tensor and None for output variance.
        """

        if self._map:
            assert (
                var_h is None
            ), "Input variance has to be None for maximum a posteriori forward pass."

        weight_mu, bias_mu = self.mean()
        mu_f = F.linear(mu_h, weight_mu, bias_mu)

        if self._map:
            if self.is_output:
                return mu_f, None
            else:
                if self.act == "relu":
                    return nn.ReLU()(mu_f), None
                else:
                    return CReLU()(mu_f), None

        if var_h is not None:
            var_f = F.linear(var_h, weight_mu**2)
        else:
            var_f = None

        if self.is_output:
            return mu_f, var_f
        else:
            if self.act == "relu":
                if var_f is None:
                    return nn.ReLU()(mu_f), None
                else:
                    return self.relu_moments(mu_f, torch.sqrt(var_f))
            else:
                if var_f is None:
                    return CReLU()(mu_f), None
                else:
                    return self.crelu_moments(mu_f, torch.sqrt(var_f))
