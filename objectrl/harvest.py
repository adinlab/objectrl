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
