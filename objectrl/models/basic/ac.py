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
from objectrl.models.basic.actor import Actor
from objectrl.models.basic.critic import CriticEnsemble

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class ActorCritic(Agent):
    """
    Base Actor-Critic agent combining an actor policy and a critic ensemble.
    This class serves as a foundation for various Actor-Critic algorithms.

    Args:
        config (MainConfig): Configuration object containing model hyperparameters.
        critic_type (type[CriticEnsemble]): Type of critic to use.
        actor_type (type[Actor]): Type of actor to use.

    Attributes:
        critic (CriticEnsemble): Critic network ensemble instance.
        actor (Actor): Actor network instance.
        policy_delay (int): Number of critic updates per actor update.
        n_iter (int): Iteration counter for training steps.
    """

    _agent_name = "AC"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type[CriticEnsemble],
        actor_type: type[Actor],
    ) -> None:
        """
        Initializes the ActorCritic agent with actor and critic networks.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type[CriticEnsemble]): Critic class type.
            actor_type (type[Actor]): Actor class type.
        Returns:
            None
        """
        super().__init__(config)

        self.critic = critic_type(config, self.dim_state, self.dim_act)
        self.actor = actor_type(config, self.dim_state, self.dim_act)
        self.policy_delay: int = config.model.policy_delay
        self.n_iter: int = 0

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
                batch["reward"], batch["next_state"], batch["terminated"], self.actor
            )
            self.critic.update(batch["state"], batch["action"], bellman_target)

            # Update the actor network periodically
            if self.n_iter % self.policy_delay == 0:
                self.actor.update(batch["state"], self.critic)
                if self.actor.has_target:
                    self.actor.update_target()

            # Update target networks
            if self.critic.has_target:
                self.critic.update_target()
            self.n_iter += 1

        return None

    @torch.no_grad()
    def select_action(
        self, state: torch.Tensor, is_training: bool = True
    ) -> torch.Tensor:
        """
        Select an action based on the current state.

        Args:
            state (torch.Tensor): The current state.
            is_training (bool): Whether the agent is in training mode.
        Returns:
            torch.Tensor: The selected action.
        """
        act_dict = self.actor.act(state, is_training=is_training)
        return act_dict

    def reset(self) -> None:
        """
        Reset the agent.

        Args:
            None
        Returns:
            None
        """
        if self.actor._reset:
            self.actor.reset()
        if self.critic._reset:
            self.critic.reset()
