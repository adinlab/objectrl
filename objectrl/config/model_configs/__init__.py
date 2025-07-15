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

import importlib
import os
import pkgutil

"""
Auto-loader for model, actor, and critic configuration classes within the `models/` subpackage.
This module dynamically discovers and imports all Python modules in the current package directory,
and searches each for specific class names matching the following patterns:

- `<MODULE_NAME_UPPERCASE>Config`
- `<MODULE_NAME_UPPERCASE>ActorConfig`
- `<MODULE_NAME_UPPERCASE>CriticConfig`

Matching classes are added to the corresponding dictionaries:
- `model_configs` maps module names to their `<NAME>Config` class
- `actor_configs` maps module names to their `<NAME>ActorConfig` class
- `critic_configs` maps module names to their `<NAME>CriticConfig` class

These dictionaries can be used elsewhere for programmatically selecting configurations.
"""
# Get the package name dynamically
package_name = __name__

# Find and import all modules inside this subpackage (models/)
model_configs = {}
actor_configs = {}
critic_configs = {}
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)

    # Construct the expected class name (ModelNameConfig)
    class_name = f"{module_name.upper()}Config"
    actor_name = f"{module_name.upper()}ActorConfig"
    critic_name = f"{module_name.upper()}CriticConfig"

    # Check if the class exists in the module
    if hasattr(module, class_name):
        model_configs[module_name] = getattr(
            module, class_name
        )  # Store class reference

    # Check if the class exists in the module
    if hasattr(module, actor_name):
        actor_configs[module_name] = getattr(
            module, actor_name
        )  # Store class reference

    # Check if the class exists in the module
    if hasattr(module, critic_name):
        critic_configs[module_name] = getattr(
            module, critic_name
        )  # Store class reference
