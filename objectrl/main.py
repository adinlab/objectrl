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
