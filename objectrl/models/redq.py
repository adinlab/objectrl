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
