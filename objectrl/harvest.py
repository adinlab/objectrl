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

import tyro

from objectrl.config.config import HarvestConfig
from objectrl.utils.harvest_utils import Harvester


def main(config: HarvestConfig):
    """
    Main entry point for the harvesting evaluation pipeline.

    Args:
        config (HarvestConfig): Configuration object containing
            all settings for harvesting, such as paths, models,
            environments, and verbosity.

    This function prints the configuration if verbose mode is enabled,
    creates a Harvester instance, and runs the harvesting process.
    """
    if config.verbose:
        pprint.pprint(config)

    harvester = Harvester(config)
    harvester.harvest()


if __name__ == "__main__":
    """
    Entry point when running this script directly.

    Parses command-line arguments to create a HarvestConfig,
    then calls the main harvesting function.
    """
    config = tyro.cli(HarvestConfig, prog="ObjectRL")

    main(config)
