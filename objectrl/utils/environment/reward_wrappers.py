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

import gymnasium as gym
import numpy as np


class PositionDelayWrapper(gym.RewardWrapper):
    """
    A Gymnasium wrapper that modifies the reward function based on position delay and control cost.
    This wrapper delays reward until the agent reaches a certain position (`position_delay`).
    It also penalizes large control signals to encourage smoother actions.

    Attributes:
        env (gym.Env): The environment to wrap.
        position_delay (float): Minimum x-position the agent must reach before receiving reward.
        ctrl_w (float): Weight for the control cost penalty term.
    """

    def __init__(
        self, env: gym.Env, position_delay: float = 2, ctrl_w: float = 0.001
    ) -> None:
        """
        Initialize the PositionDelayWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            position_delay (float): Minimum x-position the agent must reach before receiving reward.
            ctrl_w (float): Weight for the control cost penalty term.
        Returns:
            None
        """
        super().__init__(env)
        self.position_delay = position_delay
        self.ctrl_w = ctrl_w

    def step(self, action: np.ndarray) -> tuple:
        """
        Take a step in the environment, modifying the reward.
        The environment's reward is replaced with a custom one that combines delayed
        forward movement reward and a control cost.

        Args:
            action (np.ndarray): Action taken by the agent.
        Returns:
            tuple: (observation, modified_reward, terminated, truncated, info)
                - `info["x_pos"]`: Current x-position of the agent.
                - `info["action_norm"]`: Squared norm of the action.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["x_pos"] = self.data.qpos[0]
        info["action_norm"] = np.sum(np.square(action))
        return (
            observation,
            self.reward(observation, action),
            terminated,
            truncated,
            info,
        )

    def reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        Compute the modified reward based on position delay and control penalty.

        Args:
            observation: Current observation (unused here, but kept for compatibility).
            action (np.ndarray): Action taken by the agent.
        Returns:
            float: Modified reward value.
        """
        x_pos = self.data.qpos[0]
        x_vel = self.data.qvel[0]
        ctrl_cost = self.ctrl_w * np.sum(np.square(action))
        forward_reward = (x_pos >= self.position_delay) * x_vel
        rewards = forward_reward - ctrl_cost
        return rewards
