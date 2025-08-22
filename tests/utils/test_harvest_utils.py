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

import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd

from objectrl.utils.harvest_utils import Harvester
from objectrl.config.config import HarvestConfig


@pytest.fixture
def harvester():
    config = HarvestConfig(
        logs_path=Path("/tmp/logs"),
        result_path=Path("/tmp/results"),
        env_names=["env1"],
        model_names=["modelA"],
        seeds=[0, 1],
        smoothing_window=3,
        width=10,
        height=5,
        dpi=100,
        verbose=False,
    )
    return Harvester(config)


@patch("objectrl.utils.harvest_utils.np.load")
def test_collect_results(mock_np_load, harvester):
    dummy_eval_results = {
        0: torch.tensor([1.0, 2.0, 3.0]),
        1: torch.tensor([2.0, 3.0, 4.0]),
        2: torch.tensor([3.0, 4.0, 5.0]),
    }
    mock_loaded_obj = MagicMock()
    mock_loaded_obj.item.return_value = dummy_eval_results
    mock_np_load.return_value = mock_loaded_obj

    harvester.get_result_file_path = MagicMock(
        return_value=Path("/dummy/path/eval_results.npy")
    )

    harvester.smooth_curve = lambda x: x

    harvester.collect_results()

    results_final = harvester.results["Final"]["env1"]["modelA"]
    results_iqm = harvester.results["IQM"]["env1"]["modelA"]
    results_aulc = harvester.results["AULC"]["env1"]["modelA"]

    assert len(results_final) == 2
    assert len(results_iqm) == 2
    assert len(results_aulc) == 2


@patch("matplotlib.pyplot.subplots")
def test_plot_results_and_plot_model_metrics(mock_subplots, harvester):
    class DummySpine:
        def __init__(self):
            self.visible = None

        def set_visible(self, visible):
            self.visible = visible

    class DummyAxis:
        def __init__(self):
            self.plots = []
            self.title = None
            self.xlabel = None
            self.ylabel = None
            self.legend_called = False
            self.grid_called = False
            self.limits = None
            self._top_spine = DummySpine()
            self._right_spine = DummySpine()

        def plot(self, x, y, label=None, linewidth=None):
            self.plots.append((x, y, label, linewidth))

        def fill_between(self, x, y1, y2, alpha=None):
            pass

        def set_title(self, title):
            self.title = title

        def set_xlabel(self, label):
            self.xlabel = label

        def set_ylabel(self, label):
            self.ylabel = label

        def legend(self):
            self.legend_called = True

        def grid(self):
            self.grid_called = True

        def set_xlim(self, left, right):
            self.limits = (left, right)

        @property
        def spines(self):
            return {"top": self._top_spine, "right": self._right_spine}

    dummy_fig = MagicMock()
    dummy_ax = DummyAxis()

    mock_subplots.side_effect = [
        (dummy_fig, dummy_ax),
        (dummy_fig, dummy_ax),
    ]

    harvester.curves["env1"]["modelA"]["seeds"] = [
        np.ones((3, 1)),
        np.ones((3, 1)) * 2,
    ]
    harvester.curves["env1"]["modelA"]["x"] = np.arange(3).reshape(-1, 1)
    harvester.results["Final"]["env1"]["modelA"] = [1.0, 1.5]
    harvester.results["IQM"]["env1"]["modelA"] = [1.1, 1.4]
    harvester.results["AULC"]["env1"]["modelA"] = [1.2, 1.6]

    harvester.plot_results()

    assert dummy_ax._top_spine.visible is False
    assert dummy_ax._right_spine.visible is False

    assert dummy_ax.title is not None
    assert dummy_ax.xlabel == "Timesteps"
    assert dummy_ax.ylabel == "Return"
    assert dummy_ax.legend_called
    assert dummy_ax.grid_called
    assert dummy_ax.limits is not None
