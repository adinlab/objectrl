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

import logging
import typing
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy
import torch

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class Logger:
    """
    Logger class for experiment tracking, result storage, and evaluation plotting.

    Args:
        result_path (str): Base directory where results will be stored.
        env_name (str): Name of the environment being used.
        model_name (str): Name of the model being trained.
        seed (int): Random seed for reproducibility.
        config (MainConfig, optional): Configuration object containing experiment parameters.

    Attributes:
        path (Path): Path to the directory where logs and plots are saved.
        eval_results (dict): Stores evaluation rewards for different training steps.
        logger (logging.Logger): Python logger instance configured to write to a file.
    """

    def __init__(
        self,
        result_path: Path,
        env_name: str,
        model_name: str,
        seed: int,
        config: Optional["MainConfig"] = None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = (
            result_path
            / env_name
            / model_name
            / f"seed_{str(seed).zfill(2)}"
            / timestamp
        )
        # Create data storage
        self.eval_results = {}

        # Setup logger
        self.logger = self.create_logger()
        self.logger.info(f"Experiment with seed no {seed}")
        # If user provided a config, log its details
        if config is not None:
            self.logger.info(f"Args: \n{config}")

    def create_logger(self) -> logging.Logger:
        """
        Sets up a file-based logger.

        Args:
            None
        Returns:
            logging.Logger: Configured logger object for recording logs.
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure logging
        logging.basicConfig(
            filename=self.path / "log.log",
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M:%S",
            level=logging.INFO,
            filemode="w",
        )

        logging.disable(logging.DEBUG)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        return logger

    def log(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to be logged.
        Returns:
            None
        """
        self.logger.info(message)

    def critical(self, message: str) -> None:
        """
        Logs a critical message (used for evaluation results).

        Args:
            message (str): The critical message to be logged.
        Returns:
            None
        """
        self.logger.critical(message)

    def __call__(self, message: str) -> None:
        """
        Allows the logger instance to be called like a function.

        Args:
            message (str): The message to be logged as info.
        Returns:
            None
        """
        self.log(message)

    def episode_summary(self, episode: int, steps: int, info: dict) -> None:
        """
        Logs a summary of a completed episode.

        Args:
            episode (int): Episode index.
            steps (int): Step count at episode end.
            info (dict): Dictionary containing reward and step information.
        Returns:
            None
        """
        reward = info["episode_rewards"][episode]
        self.log(
            f"Episode: {episode + 1:4d}\tN-steps: {steps:7d}\tReward: {reward:10.3f}"
        )

    def plot_rewards(self, rewards: numpy.ndarray, steps: numpy.ndarray) -> None:
        """
        Generates and saves a plot of normalized per-episode rewards.

        Args:
            rewards (numpy.ndarray): Array of rewards per episode.
            steps (numpy.ndarray): Array of steps per episode.
        Returns:
            None
        """
        plt.figure()
        plt.plot(steps, rewards)
        plt.xlabel("Steps")
        plt.ylabel("Normalized Per-Episode Reward")
        plt.savefig(self.path / "learning-curve.png")
        plt.close()

    def save(self, info: dict, episode: int, n_step: int) -> None:
        """
        Saves episode and step reward information and generates training curve plots.

        Args:
            info (dict): Dictionary containing episode and step rewards.
            episode (int): Current episode index.
            n_step (int): Current training step index.
        Returns:
            None
        """
        if not info:
            return

        episode_rewards = info["episode_rewards"][: episode + 1]
        episode_steps = info["episode_steps"][: episode + 1]
        step_rewards = info["step_rewards"][: n_step + 1]

        numpy.save(self.path / "episode_rewards.npy", episode_rewards)
        numpy.save(self.path / "step_rewards.npy", step_rewards)
        self.plot_rewards(episode_rewards, episode_steps)

    @staticmethod
    def IQM_reward_calculator(rewards: torch.Tensor) -> numpy.floating:
        """
        Computes the Interquartile Mean (IQM) of rewards.

        Args:
            rewards (torch.Tensor): Tensor of evaluation rewards.
        Returns:
            float: The IQM of the rewards.
        """
        q1 = numpy.percentile(rewards, 25)
        q3 = numpy.percentile(rewards, 75)
        rewards = rewards.numpy()
        return numpy.mean(rewards[(rewards >= q1) & (rewards <= q3)])

    def save_eval_results(self, n_step: int, rewards: torch.Tensor) -> None:
        """
        Saves evaluation results and logs IQM and mean rewards.

        Args:
            n_step (int): Step at which evaluation is performed.
            rewards (torch.Tensor): Array of reward values from evaluation episodes.
        Returns:
            None
        """
        iqm = self.IQM_reward_calculator(rewards)
        mean_reward = rewards.mean()

        self.critical(
            f"EVALUATION\tN-steps: {n_step:7d}\tMean_Reward: {mean_reward:10.3f}\tIQM_Reward: {iqm:10.3f}"
        )
        self.eval_results[n_step] = rewards
        numpy.save(self.path / "eval_results.npy", self.eval_results)

        # Plot evaluation curve
        self._plot_eval_curve()

    def _plot_eval_curve(self) -> None:
        """Helper method to plot evaluation curve.

        Args:
            None
        Returns:
            None
        """
        x = list(self.eval_results.keys())
        y_mean = numpy.array([self.eval_results[k].mean() for k in x])
        y_std = numpy.array([self.eval_results[k].std() for k in x])

        plt.figure()
        plt.plot(x, y_mean)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        plt.xlabel("Steps")
        plt.ylabel("Eval Reward")
        plt.savefig(self.path / "eval-curve.png")
        plt.close()
