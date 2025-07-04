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

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from objectrl.loggers.logger import Logger


@pytest.fixture
def dummy_logger(tmp_path):
    result_path = tmp_path
    env_name = "cheetah"
    model_name = "DummyAgent"
    seed = 42
    return Logger(result_path, env_name, model_name, seed)


def test_logger_initialization_creates_path_and_logger(tmp_path):
    logger = Logger(tmp_path, "cheetah", "DummyAgent", 42)
    assert logger.path.exists()
    assert logger.logger is not None
    assert logger.eval_results == {}


def test_logger_log_and_critical_calls_loggers(dummy_logger):
    with (
        patch.object(dummy_logger.logger, "info") as mock_info,
        patch.object(dummy_logger.logger, "critical") as mock_critical,
    ):
        dummy_logger.log("Test message")
        dummy_logger.critical("Critical message")

        mock_info.assert_called_with("Test message")
        mock_critical.assert_called_with("Critical message")


def test_logger_episode_summary_logs_correct_format(dummy_logger):
    dummy_info = {"episode_rewards": torch.tensor([10.0, 20.0, 30.0])}
    with patch.object(dummy_logger.logger, "info") as mock_info:
        dummy_logger.episode_summary(1, 123, dummy_info)
        mock_info.assert_called_once()
        message = mock_info.call_args[0][0]
        assert "Episode:" in message
        assert "2" in message  # episode index is 1, but it logs 1+1
        assert "N-steps:" in message
        assert "123" in message
        assert "Reward:" in message
        assert "20.000" in message


def test_logger_save_saves_arrays_and_plots(dummy_logger):
    dummy_info = {
        "episode_rewards": torch.tensor([10.0, 20.0, 30.0]),
        "episode_steps": torch.tensor([5, 10, 15]),
        "step_rewards": [("ep1", 0, 1.0)] * 15,
    }

    with (
        patch("numpy.save") as mock_save,
        patch.object(dummy_logger, "plot_rewards") as mock_plot,
    ):
        dummy_logger.save(dummy_info, episode=2, n_step=14)
        assert mock_save.call_count == 2  # episode_rewards + step_rewards
        mock_plot.assert_called_once()


def test_logger_save_eval_results_saves_and_logs(dummy_logger):
    rewards = torch.tensor([10.0, 20.0, 30.0])
    with (
        patch.object(dummy_logger, "critical") as mock_critical,
        patch("numpy.save") as mock_save,
        patch.object(dummy_logger, "_plot_eval_curve") as mock_plot,
    ):

        dummy_logger.save_eval_results(100, rewards)

        assert 100 in dummy_logger.eval_results
        mock_critical.assert_called_once()
        mock_save.assert_called_once()
        mock_plot.assert_called_once()


def test_logger_iqm_reward_calculator():
    rewards = torch.tensor([1, 2, 3, 4, 100])  # outlier on high end
    iqm = Logger.IQM_reward_calculator(rewards)
    # should exclude 1 and 100 (below Q1 and above Q3)
    assert 2 <= iqm <= 4


def test_logger_plot_eval_curve(dummy_logger):
    dummy_logger.eval_results = {
        10: torch.tensor([1.0, 2.0, 3.0]),
        20: torch.tensor([2.0, 3.0, 4.0]),
    }

    with patch("matplotlib.pyplot.savefig") as mock_save:
        dummy_logger._plot_eval_curve()
        mock_save.assert_called_once()
