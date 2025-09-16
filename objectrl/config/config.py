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

"""
Configuration module for reinforcement learning experiments.

This module provides structured configuration management for setting up and
running reinforcement learning (RL) experiments. It defines dataclasses that
capture different components of an experiment setup, including:

- `NoiseConfig`: Optional Gaussian noise added to observations or actions of the RL agent.
- `EnvConfig`: Parameters specific to the RL environment.
- `TrainingConfig`: Training-related hyperparameters.
- `SystemConfig`: Runtime and system-level settings like device and seeds.
- `LoggingConfig`:  Logging, checkpointing, and result-saving options.
- `MainConfig`: Top-level config that combining all above configs and
                supports loading from external YAML or dict sources.
- `HarvestConfig`: Evaluation, visualization, and aggregation of multiple runs.

It integrates with dynamic model definitions from `model_configs`, and provides
a method (`MainConfig.from_config`) to construct complete configurations
programmatically from user-defined dictionaries.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy
import tyro

from objectrl.config.model import (
    ActorConfig,
    CriticConfig,
    ModelConfig,
)
from objectrl.config.model_configs import model_configs


# [start-noise-config]
@dataclass
class NoiseConfig:
    """
    Configuration for injecting Gaussian noise into actions and observations of the agent.
    Attributes:
        noisy_act (float): Standard deviation of noise added to actions. Default is 0.0 (no noise).
        noisy_obs (float): Standard deviation of noise added to observations. Default is 0.0 (no noise).
    """

    noisy_act: float = 0.0
    noisy_obs: float = 0.0


# [end-noise-config]


# [start-env-config]
@dataclass
class EnvConfig:
    """
    Configuration for setting up the reinforcement learning environment.
    Attributes:
        name (str): Environment name from a predefined set.
        noisy (NoiseConfig | None): Optional noise configuration.
        position_delay (float | None): Optional delay in position updates.
        control_cost_weight (float | ): Optional weight for control cost penalty.
        sparse_rewards (bool): Whether to use sparse rewards.
    """

    name: (
        Literal[
            "ant",
            "cartpole",
            "cheetah",
            "hopper",
            "humanoid",
            "reacher",
            "swimmer",
            "walker2d",
            "dmc-quadruped-run",
            "dmc-humanoid-run",
            "dmc-cheetah-run",
            "dmc-hopper-hop",
            "dmc-walker-run",
            "metaworld-window-close",
            "metaworld-window-open",
            "metaworld-drawer-close",
            "metaworld-drawer-open",
            "metaworld-reach",
            "metaworld-button-press-topdown",
            "metaworld-door-open",
        ]
        | str
    ) = "cheetah"
    noisy: NoiseConfig | None = None
    position_delay: float | None = None
    control_cost_weight: float | None = None
    sparse_rewards: bool = False


# [end-env-config]


# [start-training-config]
@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters.
    Attributes include learning rate, batch size, discount factor, buffer size,
    evaluation frequency, and more.
    """

    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    max_steps: int = 1_000_000
    warmup_steps: int = 10_000
    buffer_size: int = 1_000_000

    ### Training frequency settings
    reset_frequency: int = 0
    learn_frequency: int = 1
    max_iter: int = 1
    n_epochs: int = 0

    ### Evaluation settings
    eval_episodes: int = 10
    eval_frequency: int = 20_000
    # Run evaluations in parallel or sequentially
    parallelize_eval: bool = False

    optimizer: str = "Adam"


# [end-training-config]


# [start-system-config]
@dataclass
class SystemConfig:
    """
    Configuration for system-level execution.

    Attributes:
        num_threads (int): Number of threads (-1 for auto).
        seed (int): Random seed.
        random_seed (int): Let the config sample a random seed
        device (str): Runtime device ("cpu" or "cuda").
        storing_device ("cpu" or "cuda'): Device used for storing models/data. Store on the CPU if memory is a constraint
            otherwise prefer the gpu
    """

    num_threads: int = -1
    seed: int = 1
    # Initialize with a random seed
    random_seed: bool = False
    device: Literal["cpu", "cuda"] = "cuda"
    storing_device: Literal["cpu", "cuda"] = "cuda"

    def __post_init__(self):
        if self.random_seed:
            self.seed = numpy.random.randint(2**32)


# [end-system-config]


# [start-logging-config]
@dataclass
class LoggingConfig:
    """
    Configuration for logging experiment outputs.

    Attributes:
        result_path (str): Path to save experiment results.
        save_frequency (int): Save logs every N steps.
        save_params (bool): Whether to save model parameters at the end.
    """

    result_path: str = "../_logs"
    save_frequency: int = 20_000
    save_params: bool = False

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.result_path = Path(self.result_path)


# [end-logging-config]


# [start-main-config]
@dataclass
class MainConfig:
    """
    Main configuration combining environment, training, system, logging, and model settings.
    This class allows for central management and construction of experiment
    configurations and supports loading from dictionary or YAML files.
    """

    # Provide additional output
    verbose: bool = False
    # Show a progress bar
    progress: bool = False
    # An optional config path
    config: Path | None = None

    # Environmental config
    env: EnvConfig = field(default_factory=EnvConfig)
    # Training related configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Model related configuration. These cannot be changed via the CLI
    model: tyro.conf.Suppress[ModelConfig] = field(default_factory=ModelConfig)
    # model: ModelConfig = field(default_factory=ModelConfig)

    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MainConfig":
        """
        Override configuration values from a YAML file.

        Args:
            config (dict[str, Any]): Dictionary with optional keys:
                - 'env'
                - 'training'
                - 'system'
                - 'logging'
                - 'model' (required)
        Returns:
            MainConfig: A fully initialized configuration object.
        """

        config = copy.deepcopy(config)

        env_conf = config.pop("env", {})
        training_conf = config.pop("training", {})
        system_conf = config.pop("system", {})
        logging_conf = config.pop("logging", {})

        env = EnvConfig(**env_conf) if env_conf else EnvConfig()
        training = (
            TrainingConfig(**training_conf) if training_conf else TrainingConfig()
        )
        system = SystemConfig(**system_conf) if system_conf else SystemConfig()
        logging = LoggingConfig(**logging_conf) if logging_conf else LoggingConfig()

        model_conf = config.pop("model", {})
        assert model_conf, "Need to specify a model"
        model_name = model_conf["name"]
        assert model_name in model_configs.keys(), f"{model_name} is not available"

        if "actor" in model_configs[model_name].__annotations__:
            actor_conf = model_conf.pop("actor", {})
            actor = ActorConfig.from_config(actor_conf, model_name)
            model_conf["actor"] = actor
        if "critic" in model_configs[model_name].__annotations__:
            critic_conf = model_conf.pop("critic", {})
            critic = CriticConfig.from_config(critic_conf, model_name)
            model_conf["critic"] = critic

        model = model_configs[model_name](**model_conf)

        return cls(
            env=env,
            training=training,
            system=system,
            logging=logging,
            model=model,
            **config,
        )


# [end-main-config]


# [start-harvest-config]
@dataclass
class HarvestConfig:
    """
    Configuration for evaluation and visualization of experiments.

    Attributes:
        verbose (bool): Whether to provide verbose output.
        logs_path (str): Path to log files.
        result_path (str): Path to save results.
        env_names (list[str]): List of environment names to evaluate on.
        model_names (list[str]): List of model names to evaluate.
        seeds (list[int]): Random seeds to evaluate across.
        smoothing_window (int): Window size for reward smoothing.
        height (int): Plot height.
        width (int): Plot width.
        dpi (int): Plot DPI.
        y_axis (str): Label for the y-axis in plots.
    """

    # Provide additional output
    verbose: bool = True
    # Path to logs
    logs_path: str = "../_logs"
    # path to save results
    result_path: str = "../_results"
    # envs
    env_names: list[
        Literal[
            "ant",
            "cartpole",
            "cheetah",
            "hopper",
            "humanoid",
            "reacher",
            "swimmer",
            "walker2d",
            "dmc-quadruped-run",
            "dmc-humanoid-run",
            "dmc-cheetah-run",
            "dmc-hopper-hop",
            "dmc-walker-run",
            "metaworld-window-close",
            "metaworld-window-open",
            "metaworld-drawer-close",
            "metaworld-drawer-open",
            "metaworld-reach",
            "metaworld-button-press-topdown",
            "metaworld-door-open",
        ]
    ] = field(default_factory=lambda: ["cheetah"])

    # models
    models = Literal[tuple(model_configs.keys())]
    model_names: list[models] = field(default_factory=lambda: ["ddpg"])
    del models

    # seeds
    seeds: list[int] = field(default_factory=lambda: list(range(1, 11)))

    # smoothing window >= 1 if 1 then no smoothing
    # smoothing_window should be odd
    smoothing_window: int = 1

    # plotting
    height: int = 5
    width: int = 10
    dpi: int = 200
    # label for the y axis, e.g., "Return" or "Success Rate"
    y_axis: str = "Return"

    def __post_init__(self):
        """Convert log and result paths to Path objects."""

        self.logs_path = Path(self.logs_path)
        self.result_path = Path(self.result_path)


# [end-harvest-config]
