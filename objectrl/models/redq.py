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

import torch

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.sac import SACActor, SACCritic

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class REDQCritic(SACCritic):
    """
    REDQ critic ensemble implementing Randomized Ensembled Double Q-learning.

    Args:
        config (MainConfig): Configuration object containing model hyperparameters.
        dim_state (int): Dimensionality of the state space.
        dim_act (int): Dimensionality of the action space.

    This class extends the SAC critic ensemble by implementing a
    randomized target Q-value estimation with sub-ensemble sampling.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)

    # [start-reduce-code]
    def reduce(self, q_val_list: torch.Tensor, reduce_type="min") -> torch.Tensor:
        """
        Randomly samples a subset of critics from the ensemble and reduces their Q-values.

        Args:
            q_val_list (torch.Tensor): List of Q-value tensors from each critic in the ensemble.
            reduce_type (str): Reduction method.

        Returns:
            torch.Tensor: Reduced Q-values obtained by taking the minimum over sampled critics.
        """
        if reduce_type == "min":
            if len(q_val_list) < self.config.model.n_in_target:
                raise ValueError(
                    f"Expected at least {self.config.model.n_in_target} critics, but got {len(q_val_list)}."
                )

            i_targets = torch.randperm(int(self.n_members))[
                : self.config.model.n_in_target
            ]

            return torch.stack([q_val_list[i] for i in i_targets], dim=-1).min(-1)[0]
        elif reduce_type == "mean":
            return q_val_list.mean(0)
        else:
            raise ValueError(
                f"Unsupported reduce type: {reduce_type}. Use 'min' or 'mean'."
            )
        # [end-reduce-code]


# [start-redq-code]
class RandomizedEnsembledDoubleQLearning(ActorCritic):
    """
    REDQ agent implementation combining REDQCritic and SACActor.

    Args:
        config (MainConfig): Global configuration object.
        critic_type (type): Type of critic to use, defaults to REDQCritic.
        actor_type (type): Type of actor to use, defaults to SACActor.
    """

    _agent_name = "REDQ"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = REDQCritic,
        actor_type: type = SACActor,
    ) -> None:
        super().__init__(config, critic_type, actor_type)


# [end-redq-code]
