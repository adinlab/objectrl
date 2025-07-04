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

from objectrl.agents.base_agent import Agent
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.utils.utils import dim_check

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class DQNCritic(CriticEnsemble):
    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int):
        super().__init__(config, dim_state, dim_act)

        self.dim_act = dim_act
        self._explore_rate: float = config.model.critic.exploration_rate

    def update(
        self, state: torch.Tensor, action: torch.Tensor, y: torch.Tensor
    ) -> None:
        """
        Update critic networks using the provided Bellman targets.

        Args:
            state: State tensor.
            action: Action tensor.
            y: Bellman target values.
        Returns:
            None
        """
        self.optim.zero_grad()

        pred = self.Q(state, None)[range(state.shape[0]), action.int()][:, None]
        dim_check(pred, y)
        loss = self.loss(pred, y).mean()
        loss.backward()
        self.optim.step()
        self.iter += 1

    def act(
        self, state: torch.Tensor, target: bool = False, is_training: bool = True
    ) -> torch.Tensor:

        if is_training and torch.rand(1) < self._explore_rate:
            return torch.randint(self.dim_act, size=(1,), device=state.device)

        if target:
            return self.Q_t(state, None).argmax(dim=-1, keepdim=True)
        else:
            return self.Q(state, None).argmax(dim=-1, keepdim=True)

    def Q(self, state: torch.Tensor, action: None) -> torch.Tensor:
        # Indexing as there is only a single critic
        return self.model_ensemble(state)[0]

    def Q_t(self, state: torch.Tensor, action: None) -> torch.Tensor:
        # Indexing as there is only a single critic
        return self.model_ensemble(state)[0]

    @torch.no_grad()
    def get_bellman_target(
        self, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:

        target_value, _ = self.Q_t(next_state, None).max(-1, keepdim=True)

        dim_check(reward.unsqueeze(-1), target_value)
        y = reward.unsqueeze(-1) + self._gamma * target_value * (1 - done.unsqueeze(-1))

        return y


class DQN(Agent):
    def __init__(self, config: "MainConfig", critic_type: type[CriticEnsemble]) -> None:
        """
        Deep Q-Network

        Args:
            config (MainConfig): Global configuration object.
            critic_type (type[CriticEnsemble]): Type of critic ensemble to use.
        Returns:
            None
        """
        super().__init__(config)

        self.critic = critic_type(config, self.dim_state, self.dim_act)
        self.n_iter: int = 0

        # Requires discrete action spaces
        self._discrete_action_space = True

    def learn(self, max_iter: int = 1, n_epochs: int = 0) -> None:
        """
        Perform the learning process for the agent.

        Args:
            max_iter (int): Maximum number of iterations for learning.
            n_epochs (int): Number of epochs for training. If 0, random sampling is used.
        Returns:
            None
        """
        # Check if there is enough data in memory to sample a batch
        if self.config_train.batch_size > len(self.experience_memory):
            return None

        # Determine the number of steps and initialize the iterator
        n_steps = self.experience_memory.get_steps_and_iterator(
            n_epochs, max_iter, self.config_train.batch_size
        )

        for _ in range(n_steps):
            # Get batch using the internal iterator
            batch = self.experience_memory.get_next_batch(self.config_train.batch_size)

            bellman_target = self.critic.get_bellman_target(
                batch["reward"], batch["next_state"], batch["terminated"]
            )

            self.critic.update(batch["state"], batch["action"], bellman_target)

            # Update target networks
            if self.critic.has_target:
                self.critic.update_target()
            self.n_iter += 1

        return None

    def select_action(
        self, state: torch.Tensor, is_training: bool = True
    ) -> torch.Tensor:
        return self.critic.act(state, is_training=is_training)

    def reset(self) -> None:
        if self.critic._reset:
            self.critic.reset()
