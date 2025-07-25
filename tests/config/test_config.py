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

import pytest
from pathlib import Path
from objectrl.config.config import (
    NoiseConfig,
    EnvConfig,
    TrainingConfig,
    SystemConfig,
    LoggingConfig,
    MainConfig,
    HarvestConfig,
)
from objectrl.config.model import model_configs


def test_noise_config_defaults():
    noise = NoiseConfig()
    assert noise.noisy_act == 0.0
    assert noise.noisy_obs == 0.0
    noise = NoiseConfig(noisy_act=0.1, noisy_obs=0.2)
    assert noise.noisy_act == 0.1
    assert noise.noisy_obs == 0.2


def test_logging_config_path_conversion():
    log = LoggingConfig(result_path="../_logs_test")
    assert isinstance(log.result_path, Path)
    assert str(log.result_path).endswith("_logs_test")


def test_env_config_defaults_and_override():
    env = EnvConfig()
    assert env.name == "cheetah"
    assert env.noisy is None
    assert env.position_delay is None
    assert env.control_cost_weight is None

    env2 = EnvConfig(
        name="hopper", noisy=None, position_delay=0.1, control_cost_weight=0.5
    )
    assert env2.name == "hopper"
    assert env2.position_delay == 0.1
    assert env2.control_cost_weight == 0.5


def test_training_config_defaults_and_override():
    train = TrainingConfig()
    assert train.learning_rate == 3e-4
    assert train.batch_size == 256
    assert train.gamma == 0.99
    assert train.max_steps == 1_000_000
    assert train.warmup_steps == 10_000
    assert train.optimizer == "Adam"

    train2 = TrainingConfig(learning_rate=1e-3, batch_size=128, optimizer="SGD")
    assert train2.learning_rate == 1e-3
    assert train2.batch_size == 128
    assert train2.optimizer == "SGD"


def test_system_config_defaults_and_override():
    system = SystemConfig()
    assert system.num_threads == -1
    assert system.seed == 1
    assert system.device == "cuda"
    assert system.storing_device == "cpu"

    system2 = SystemConfig(device="cpu", seed=42)
    assert system2.device == "cpu"
    assert system2.seed == 42


def test_env_config_name_literals():
    valid_names = [
        "ant",
        "cartpole",
        "cheetah",
        "hopper",
        "humanoid",
        "reacher",
        "swimmer",
        "custom_env",
    ]
    for name in valid_names:
        env = EnvConfig(name=name)
        assert env.name == name


def test_system_config_device_literals():
    for device in ["cpu", "cuda"]:
        sys_conf = SystemConfig(device=device)
        assert sys_conf.device == device


def test_harvest_config_path_conversion_and_defaults():
    harvest = HarvestConfig()
    assert isinstance(harvest.logs_path, Path)
    assert isinstance(harvest.result_path, Path)
    assert harvest.verbose is True
    assert "cheetah" in harvest.env_names


def test_main_config_from_config_minimal_valid():
    model_name = next(iter(model_configs))
    config_dict = {"model": {"name": model_name}}
    main_conf = MainConfig.from_config(config_dict)
    assert isinstance(main_conf, MainConfig)
    assert main_conf.model.name == model_name


def test_main_config_from_config_invalid_model():
    with pytest.raises(AssertionError):
        MainConfig.from_config({"model": {"name": "nonexistent_model"}})


def test_main_config_from_config_missing_model():
    with pytest.raises(AssertionError):
        MainConfig.from_config({})
