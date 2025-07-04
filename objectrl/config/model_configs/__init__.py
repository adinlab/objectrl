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
