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

import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from objectrl.config.config import HarvestConfig

# Use Agg backend for matplotlib (non-interactive)
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


class Harvester:
    """
    Collects, processes, and visualizes evaluation results for multiple models
    across various environments.

    This class is designed to aggregate performance metrics like Final Return,
    Interquartile Mean (IQM), and Area Under the Learning Curve (AULC), and
    generate publication-ready plots and tables.

    Args:
        config (HarvestConfig): Configuration object containing paths, model/env names,
                                seeds, plotting parameters, and verbosity settings.
    Attributes:
        config (HarvestConfig): Configuration object with harvesting settings.
        metrics (list[str]): List of metrics to compute and visualize.
        results (dict[str, dict[str, dict[str, list[float]]]]): Nested dictionary to store metric results.
        curves (dict[str, dict[str, dict[str, Any]]]): Nested dictionary to store learning curves for each model and environment.
    """

    def __init__(self, config: HarvestConfig) -> None:
        """
        Initialize the Harvester with a configuration object.

        Args:
            config (HarvestConfig): Configuration containing paths, model/env names, seeds,
                                    plotting parameters, and verbosity settings.
        Returns:
            None
        """
        self.config: HarvestConfig = config
        self.metrics: list[str] = ["Final", "IQM", "AULC"]
        self.initialize_data_stores()

    def initialize_data_stores(self) -> None:
        """
        Prepare internal data structures to store metric results and learning curves.

        Args:
            None
        Returns:
            None
        """
        self.results: dict[str, dict[str, dict[str, list[float]]]] = {
            metric: {
                env: {model: [] for model in self.config.model_names}
                for env in self.config.env_names
            }
            for metric in self.metrics
        }

        self.curves: dict[str, dict[str, dict[str, Any]]] = {
            env: {model: {"seeds": [], "x": None} for model in self.config.model_names}
            for env in self.config.env_names
        }

    def get_result_file_path(
        self, env: str, model: str, seed: int
    ) -> os.PathLike | None:
        """
        Get the file path to the latest evaluation results for a specific setting.

        Args:
            env (str): Environment name.
            model (str): Model name.
            seed (int): Seed index.
        Returns:
            os.PathLike | None: Full path to result `.npy` file, or None if not found.
        """
        path = self.config.logs_path / env / model / f"seed_{str(seed).zfill(2)}"
        timestamps = [x.parts[-2] for x in path.glob("*/eval_results.npy")]
        newest = max(timestamps) if timestamps else None

        if newest:
            return path / newest / "eval_results.npy"
        else:
            if self.config.verbose:
                print(f"No results found for {env} {model} {seed}")
            return None

    def smooth_curve(self, x: np.ndarray) -> np.ndarray:
        """
        Smooth a learning curve using a moving average defined in config.

        Args:
            x (np.ndarray): Raw curve array (e.g., rewards over time).
        Returns:
            np.ndarray: Smoothed version of input array.
        Raises:
            ValueError: If window is invalid or larger than the array length.
        """
        if self.config.smoothing_window < 1:
            return x
        if self.config.smoothing_window % 2 == 0:
            raise ValueError("window_size must be odd for symmetric smoothing.")
        if len(x) < self.config.smoothing_window:
            raise ValueError("Input array is smaller than the smoothing window.")

        x = x.flatten()

        half_window = self.config.smoothing_window // 2
        padded = np.pad(
            x,
            (half_window, self.config.smoothing_window - half_window - 1),
            mode="edge",
        )
        kernel = np.ones(self.config.smoothing_window) / self.config.smoothing_window
        smoothed = np.convolve(padded, kernel, mode="valid").reshape(-1, 1)
        return smoothed

    def collect_results(self) -> None:
        """
        Load results from disk and compute summary metrics (Final, IQM, AULC).
        Also smooth and store all evaluation curves for each model and seed.

        Args:
            None
        Returns:
            None
        """
        for env in self.config.env_names:
            for model in self.config.model_names:
                for seed in self.config.seeds:
                    file_path = self.get_result_file_path(env, model, seed)
                    if file_path:
                        eval_results = np.load(file_path, allow_pickle=True).item()
                        x_axis = np.array(list(eval_results.keys())).reshape(-1, 1)
                        y_axis = (
                            torch.stack(list(eval_results.values()))
                            .numpy()
                            .mean(1)
                            .reshape(-1, 1)
                        )

                        final_results = eval_results[x_axis[-1, 0]].numpy()
                        final_score = np.mean(final_results)

                        q1 = np.percentile(final_results, 25)
                        q3 = np.percentile(final_results, 75)
                        iqm = np.mean(
                            final_results[(final_results >= q1) & (final_results <= q3)]
                        )
                        aulc = np.mean(y_axis)

                        self.results["Final"][env][model].append(final_score)
                        self.results["IQM"][env][model].append(iqm)
                        self.results["AULC"][env][model].append(aulc)

                        self.curves[env][model]["seeds"].append(
                            self.smooth_curve(y_axis)
                        )
                        if self.curves[env][model]["x"] is None:
                            self.curves[env][model]["x"] = x_axis

    def format_model_name(self, model: str) -> str:
        """
        Convert model name to uppercase for presentation.

        Args:
            model (str): Original model identifier.
        Returns:
            str: Formatted model name (uppercase).
        """
        return model.upper()

    def _plot_model_metrics(
        self,
        env: str,
        model: str,
        ax: Any,
        ax_env: Any,
        df_env: pd.DataFrame,
        df_all: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Plot a learning curve for a single model, add metrics to dataframes.

        Args:
            env (str): Name of the environment.
            model (str): Name of the model.
            ax (Any): Axes for combined figure (all environments).
            ax_env (Any): Axes for individual environment figure.
            df_env (pd.DataFrame): Environment-specific metrics table.
            df_all (pd.DataFrame): Aggregated table for all metrics.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Updated environment and global results.
        """
        y_axis = np.concatenate(self.curves[env][model]["seeds"], axis=1)
        x_axis = self.curves[env][model]["x"]
        mean = np.mean(y_axis, axis=1)
        std = np.std(y_axis, axis=1)

        label = (
            f"{self.format_model_name(model)}\t {np.mean(self.results['Final'][env][model]):.2f}"
            + r" $\pm$ "
            + f"{np.std(self.results['Final'][env][model]):.2f} ({y_axis.shape[-1]})"
        )

        for plotter in (ax, ax_env):
            plotter.plot(x_axis.flatten(), mean, label=label, linewidth=2)
            plotter.fill_between(x_axis.flatten(), mean - std, mean + std, alpha=0.2)

        final = (
            f"{np.mean(self.results['Final'][env][model]):.2f}"
            + r" $\pm$ "
            + f"{np.std(self.results['Final'][env][model]):.2f}"
        )
        iqm = (
            f"{np.mean(self.results['IQM'][env][model]):.2f}"
            + r" $\pm$ "
            + f"{np.std(self.results['IQM'][env][model]):.2f}"
        )
        aulc = (
            f"{np.mean(self.results['AULC'][env][model]):.2f}"
            + r" $\pm$ "
            + f"{np.std(self.results['AULC'][env][model]):.2f}"
        )

        row = {
            "model": self.format_model_name(model),
            "n_rep": y_axis.shape[-1],
            "Final": final,
            "IQM": iqm,
            "AULC": aulc,
        }

        df_env = pd.concat([df_env, pd.DataFrame([row])], ignore_index=True)
        df_all = pd.concat(
            [df_all, pd.DataFrame([{**{"env": env}, **row}])], ignore_index=True
        )
        return df_env, df_all

    def plot_results(self) -> None:
        """
        Generate all visualizations and metric tables:
        - One plot per environment.
        - One aggregate plot for all environments.
        - CSV and Markdown outputs.

        Args:
            None
        Returns:
            None
        """
        fig_all, ax_all = plt.subplots(
            len(self.config.env_names),
            1,
            figsize=(
                self.config.width,
                self.config.height * len(self.config.env_names),
            ),
            dpi=self.config.dpi,
        )
        if len(self.config.env_names) == 1:
            ax_all = [ax_all]
        df_all = pd.DataFrame(columns=["env", "model", "n_rep", *self.metrics])

        for env, ax in zip(self.config.env_names, ax_all, strict=True):
            fig_env, ax_env = plt.subplots(
                1,
                1,
                figsize=(self.config.width, self.config.height),
                dpi=self.config.dpi,
            )
            df_env = pd.DataFrame(columns=["model", "n_rep", *self.metrics])

            for model in self.config.model_names:
                if len(self.curves[env][model]["seeds"]) > 0:
                    df_env, df_all = self._plot_model_metrics(
                        env, model, ax, ax_env, df_env, df_all
                    )

            for fig, plotter in ((fig_all, ax), (fig_env, ax_env)):
                plotter.set_title(r"\texttt{" + f"{env.capitalize()}" + r"}")
                plotter.set_xlabel("Timesteps")
                plotter.set_ylabel(self.config.y_axis)
                plotter.legend()
                plotter.grid()
                plotter.set_xlim(
                    0, self.curves[env][self.config.model_names[0]]["x"].max() + 1
                )
                plotter.spines["top"].set_visible(False)
                plotter.spines["right"].set_visible(False)
                fig.tight_layout()

            os.makedirs(self.config.result_path / env, exist_ok=True)
            fig_env.savefig(
                self.config.result_path / env / f"{env}.png",
                bbox_inches="tight",
                dpi=self.config.dpi,
            )
            fig_env.savefig(
                self.config.result_path / env / f"{env}.pdf",
                bbox_inches="tight",
                dpi=self.config.dpi,
            )
            df_env.to_csv(self.config.result_path / env / f"{env}.csv", index=False)
            df_env.to_markdown(self.config.result_path / env / f"{env}.md", index=False)

            if self.config.verbose:
                print(f"Results for {env}:")
                print(df_env.to_markdown(index=False, tablefmt="github"))

        os.makedirs(self.config.result_path, exist_ok=True)
        fig_all.savefig(
            self.config.result_path / "evaluation_curves.png",
            bbox_inches="tight",
            dpi=self.config.dpi,
        )
        fig_all.savefig(
            self.config.result_path / "evaluation_curves.pdf",
            bbox_inches="tight",
            dpi=self.config.dpi,
        )
        df_all.to_csv(self.config.result_path / "results.csv", index=False)
        df_all.to_markdown(self.config.result_path / "results.md", index=False)

        if self.config.verbose:
            print(df_all.to_markdown(index=False, tablefmt="github"))

    def harvest(self) -> None:
        """
        Execute the full harvesting pipeline:
        - Collect results from files.
        - Compute statistics.
        - Generate plots and output tables.

        Args:
            None
        Returns:
            None
        """
        self.collect_results()
        self.plot_results()
