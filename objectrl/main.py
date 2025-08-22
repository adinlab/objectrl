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

import pprint
import sys
from pathlib import Path

import tyro

from objectrl.config.config import MainConfig
from objectrl.config.utils import (
    dict_to_dataclass,
    filter_model_args,
    get_cli_tyro,
    nested_asdict,
    print_tyro_help,
    setup_config,
)
from objectrl.experiments.control_experiment import ControlExperiment


def main(config: MainConfig) -> None:
    """
    Main entry point to run the control experiment training.

    Args:
        config (MainConfig): Configuration object containing
            all settings for the experiment.

    This function prints the config if verbose, creates a ControlExperiment
    instance, and starts training.
    """
    if config.verbose:
        pprint.pprint(config)
    exp = ControlExperiment(config)

    exp.train()


if __name__ == "__main__":
    """
    Command-line interface for experiment training.

    Supports:
    - --help_model MODEL_NAME: Print help for specific model parameters.
    - --config PATH: Load configuration from a YAML file.

    Combines command-line args, config files, and defaults to
    produce a final MainConfig for running the experiment.
    """
    config_path = None
    # Separate model parameter handling from the others.
    #   This is needed to account for the dynamic data class,
    #   which tyro does not support
    for i, arg in enumerate(sys.argv):
        # provide a help interface for the cli for model parameters
        if arg == "--help_model":
            assert i + 1 < len(
                sys.argv
            ), "--help_model needs a string specifying a model"
            model_name = sys.argv[i + 1]
            conf = MainConfig.from_config({"model": {"name": model_name}})
            print("# Configuration")
            tmp_data_class = dict_to_dataclass(nested_asdict(conf.model), model_name)
            tmp_data_class.__doc__ = (
                f"Avaliable parameters for {conf.model.__class__.__name__[:-6]}"
            )
            print_tyro_help(tmp_data_class)
            sys.exit(0)

        # (optional) provide a configuration yaml file
        elif arg == "--config":
            assert i + 1 < len(sys.argv), "--config needs a string specifying a path"
            config_path = Path(sys.argv[i + 1])
            break

    # Remove all cli model parameters and let the others be handled by tyro.
    filtered, model_args = filter_model_args(sys.argv)

    sys.argv = filtered

    tyro_config = tyro.cli(MainConfig, prog="ObjectRL")
    subset_tyro = get_cli_tyro(sys.argv, MainConfig)

    # Combine defaults, yaml, and cli (model + tyro)
    config = setup_config(config_path, model_args, tyro_config, subset_tyro)

    main(config)
