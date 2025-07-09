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
Utilities for dataclass handling, config merging, and dynamic configuration setup.

This module provides:
- Enhanced serialization for dataclasses with dynamic fields.
- Deep merge of nested dictionaries via `NestedDict`.
- Difference detection between configuration dictionaries.
- Decorator for improving `__repr__` output for dataclasses.
- Tools for creating default config dictionaries.
- A full configuration pipeline via `setup_config`.

Typical use case: parsing YAML configs, merging with CLI arguments,
and producing a structured configuration object for a training pipeline.
"""

import copy
import functools
import pprint
import sys
import typing
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import tyro
import yaml

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


def enhanced_asdict(obj: Any) -> dict:
    """
    Convert a dataclass instance to a dictionary, including both declared fields
    and dynamically added attributes.

    Args:
        obj (Any): A dataclass instance.
    Returns:
        dict: A dictionary containing all attributes of the dataclass.
    """
    if not is_dataclass(obj):
        raise TypeError(f"Expected a dataclass instance, got {type(obj).__name__}")

    # Get all attributes from the object's __dict__
    result = obj.__dict__.copy()

    # Recursively convert nested dataclasses
    for key, value in result.items():
        if is_dataclass(value):
            result[key] = enhanced_asdict(value)
        elif isinstance(value, list):
            result[key] = [
                enhanced_asdict(item) if is_dataclass(item) else item for item in value
            ]
        elif isinstance(value, dict):
            result[key] = {
                k: enhanced_asdict(v) if is_dataclass(v) else v
                for k, v in value.items()
            }

    return result


class NestedDict(dict):
    """
    Dictionary subclass that enables deep merging, i.e., nested dictionaries, using the `|` operator.

    Usage:
        merged = NestedDict(dict1) | dict2
    """

    def __or__(self, other: dict) -> dict:
        """
        Deep merge another dictionary into this one using the | operator.

        Args:
            other (dict): Dictionary to merge in.
        Returns:
            dict: A new dictionary with recursively merged values.
        """

        result = NestedDict(self.copy())

        for key, value in other.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = NestedDict(result[key]) | NestedDict(value)
            else:
                # For non-dict values or new keys, use standard behavior
                result[key] = value

        return result

    def __ror__(self, other) -> dict:
        """
        Reverse operator for merging when NestedDict is on the right-hand side.

        Args:
            other (dict): Dictionary to merge from.
        Returns:
            dict: Merged dictionary.
        """
        return NestedDict(other) | self


def get_tyro_diff(
    full_dict: dict[str, Any], sub_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Extract a subset of the original dictionary that matches the keys of the subset

    Args:
        full_dict (dict[str, Any]): Dictionary to subset.
        sub_dict (dict[str, Any]): Dictionary to extract from.
    Returns:
        dict[str, Any]: Subset of the original dictionary.
    """
    res_dict = {}
    for key, val in full_dict.items():
        if key not in sub_dict.keys():
            continue
        if isinstance(val, dict):
            tmp = get_tyro_diff(full_dict[key], sub_dict[key])
            if tmp:
                res_dict[key] = tmp
        else:
            res_dict[key] = val
    return res_dict


def diff_dict(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the difference between two dictionaries.

    Args:
        pre (dict[str, Any]): The original dictionary.
        post (dict[str, Any]): The updated dictionary.
    Returns:
        dict[str, Any]: Dictionary of changed values (keys with modified values).
    """
    res_dict = {}
    for key, val in pre.items():
        if key not in post.keys():
            continue
        if isinstance(val, dict):
            tmp = diff_dict(pre[key], post[key])
            if tmp:
                res_dict[key] = tmp
        else:
            if val != post[key]:
                res_dict[key] = post[key]

    return res_dict


def enhanced_repr(cls: type) -> Callable:
    """
    A wrapper that enhances __repr__ and __str__ methods for data classes with dynamically added attributes.

    Args:
        cls (type): A dataclass type.
    Returns:
        Callable: Decorated class with a better __repr__.
    """
    if not is_dataclass(cls):
        raise TypeError("Not a dataclass")

    orig_repr = cls.__repr__

    @functools.wraps(orig_repr)
    def custom_repr(self):
        all_attr = vars(self)
        pp = pprint.PrettyPrinter()
        class_name = self.__class__.__name__
        attrs_str = pp.pformat(all_attr)[1:-1]
        return f"{class_name}({attrs_str})"

    cls.__repr__ = custom_repr
    return cls


def create_field_dict(self) -> dict[str, Any]:
    """
    Create a dictionary of default values for a dataclass's fields.

    Args:
        cls (type): A dataclass type.
    Returns:
        Dict[str, Any]: Dictionary with field names and their default values.
    """
    ret_dict = {}
    for fld in fields(self):
        if fld.default is not MISSING:
            ret_dict[fld.name] = fld.default
        elif fld.default_factory is not MISSING:
            try:
                ret_dict[fld.name] = fld.default_factory()
            except Exception:
                ret_dict[fld.name] = (
                    f"<default_factory: {getattr(fld.default_factory, "__name__", repr(fld.default_factory))}>"
                )
        else:
            ret_dict[fld.name] = None
    return ret_dict


# This is adapted partially from tyro. As such it uses several internals
def get_cli_tyro(argv: list[str], config: type) -> dict[str, Any]:
    "Take a list of arguments and extract the keys"
    parser = tyro._cli.get_parser(config)  # type: ignore

    args = list(argv[1:])

    # Fix arguments. This will modify all option-style arguments replacing
    # underscores with hyphens, or vice versa if use_underscores=True.
    modified_args: dict[str, str] = {}
    # with tyro._strings.delimeter_context("_"):
    if True:
        for index, arg in enumerate(args):
            if not arg.startswith("--"):
                continue

            if "=" in arg:
                arg, _, val = arg.partition("=")
                fixed = "--" + tyro._strings.swap_delimeters(arg[2:]) + "=" + val  # type: ignore
            else:
                fixed = "--" + tyro._strings.swap_delimeters(arg[2:])  # type: ignore
            modified_args[fixed] = arg
            args[index] = fixed

    known_args = parser.parse_args(args)

    flat_dict = vars(known_args)

    def create_nested(flat_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Create a nested dictionary out of a flat one with keys separated by '.', and replace '-' with '_'
        """
        nested_dict = {}
        for key, value in flat_dict.items():
            # Ignore missing-type values
            if isinstance(value, tyro._singleton.NonpropagatingMissingType):  # type: ignore
                continue
            if isinstance(value, list) and all(
                isinstance(v, tyro._singleton.NonpropagatingMissingType) for v in value  # type: ignore
            ):
                continue
            parts = key.split(".")
            current = nested_dict
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            # If value is list of one element, unpack it
            current[parts[-1].replace("-", "_")] = None
        return nested_dict

    return create_nested(flat_dict)


def setup_config(
    config_path: Path | None,
    model_args: dict[str, Any],
    tyro_config: "MainConfig",
    subset_tyro: dict[str, Any],
) -> "MainConfig":
    """
    Load a config file (YAML), merge it with CLI arguments via Tyro, and return the final config.
    This function merges:
    - A YAML config file (if provided)
    - CLI generated model arguments
    - A Tyro-generated config object (e.g., from CLI)

    Args:
        config_path (Path | None): Path to a YAML configuration file.
        model_args (dict[str, Any]): Model arguments.
        tyro_config (MainConfig): A config object with Tyro/CLI overrides.
        subset_tyro (dict[str, Any]): A subset of tyro keys to load.
    Returns:
        MainConfig: The final combined and validated configuration object.
    Raises:
        Exception: If config file loading or merging fails.
    """
    from objectrl.config.config import MainConfig

    if config_path:
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading config file {config_path}: {e}")
            raise e
        # Override yaml_conf with cli parameters
        joint_conf = NestedDict(yaml_config) | NestedDict(model_args)
        pre_tyro_dict = enhanced_asdict(MainConfig.from_config(joint_conf))
    else:
        pre_tyro_dict = enhanced_asdict(MainConfig.from_config(model_args))

    tyro_dict = enhanced_asdict(tyro_config)
    # Get everything that tyro added
    diff_args = get_tyro_diff(tyro_dict, subset_tyro)

    # Combine the yaml file with the additions
    final_dict = NestedDict(pre_tyro_dict) | NestedDict(diff_args)

    # create the main config given the final config dict
    config = MainConfig.from_config(final_dict)
    assert config.model.name != "abstract", "No model specified"

    return config


def parse_value(value_str: str | None) -> bool | int | float | str | None:
    """
    Parse string value to appropriate Python type.
    Only detects booleans, integers, floats and returns strings otherwise.

    Args:
        value_str (str): A string to be parsed.
    Returns:
        bool | int | float | str: The parsed value.
    """
    if value_str is None:
        return None

    # Boolean parsing (case insensitive)
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    # Integer parsing
    try:
        # Check if it's a valid integer (no decimal point)
        if "." not in value_str and "e" not in value_str.lower():
            return int(value_str)
    except ValueError:
        pass

    # Float parsing
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string if no other type matches
    return value_str


def filter_model_args(argv: list[str]) -> tuple[list[str], dict[str, Any]]:
    """
    Extract all model arguments from the command line.

    Args:
        argv (list[str]): A list of command-line interface arguments.
    Returns:
        filtered_args (list[str]): A filtered list of CLI arguments excluding model arguments.
        model_dict (dict[str, Any]): A nested dictionary containing model-related parameters.
    """
    filtered_args = []
    model_dict = {}
    i = 0

    while i < len(argv):
        arg = argv[i]

        if arg.startswith("--model."):
            # Extract the nested key path
            key_path = arg[2:]  # Remove "--model." prefix

            # Get the value (next argument)
            assert i + 1 < len(argv) and not argv[i + 1].startswith(
                "--model."
            ), "every model key needs a value"
            value = parse_value(argv[i + 1])

            # Build nested dictionary
            current_dict = model_dict
            keys = key_path.split(".")

            # Navigate/create nested structure
            for j, key in enumerate(keys[:-1]):
                if key not in current_dict:
                    current_dict[key] = {}
                elif not isinstance(current_dict[key], dict):
                    # Conflict: trying to nest under a non-dict value
                    raise ValueError(
                        f"Conflict: --model.{'.'.join(keys[:j+1])} already has a value, cannot add nested key {'.'.join(keys)}"
                    )
                current_dict = current_dict[key]

            # Set the final value
            final_key = keys[-1]
            if final_key in current_dict:
                raise ValueError(f"Conflict: --model.{key_path} already has a value")

            current_dict[final_key] = value
            i += 2  # Skip both the flag and its value
        else:
            filtered_args.append(arg)
            i += 1

    return filtered_args, model_dict


def print_tyro_help(dataclass_type) -> None:
    """Print dataclass using tyro's exact help formatting.

    Args:
        dataclass_type (type): The dataclass type to print help for.
    Returns:
        None
    """

    # Capture help output by redirecting stderr
    old_stderr = sys.stderr
    try:
        help_output = StringIO()
        sys.stderr = help_output

        try:
            tyro.cli(
                dataclass_type, args=["--help"], config=(tyro.conf.FlagConversionOff,)
            )
        except SystemExit:
            pass

        help_text = help_output.getvalue()

    finally:
        sys.stderr = old_stderr

    print(help_text, end="")


def dict_to_dataclass(data: dict[str, Any], class_name: str = "DynamicClass") -> type:
    """
    Create a dataclass type at runtime from a (potentially) nested dictionary.

    Args:
        data (dict[str, Any]): Dictionary to convert into a dataclass.
        class_name (str): Name to assign to the generated dataclass type.
    Returns:
       type: Dynamically created dataclass type
    """
    annotations = {}
    defaults = {}

    for key, value in data.items():
        if isinstance(value, dict):
            # Create nested dataclass for dictionary values
            nested_class_name = f"{class_name}_{key.title()}"
            nested_class = dict_to_dataclass(value, nested_class_name)
            annotations[key] = nested_class
            defaults[key] = field(
                default_factory=lambda val=value, cls=nested_class: cls(
                    **copy.deepcopy(val)
                )
            )
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                # Handle list of dictionaries
                nested_class_name = f"{class_name}_{key.title()}Item"
                nested_class = dict_to_dataclass(value[0], nested_class_name)
                annotations[key] = f"list[{nested_class}]"
                defaults[key] = field(
                    default_factory=lambda val=value, cls=nested_class: [
                        cls(**item) for item in copy.deepcopy(val)
                    ]
                )
            elif (
                value
                and hasattr(value[0], "__class__")
                and not isinstance(value[0], str | int | float | bool)
            ):
                # Handle list of class instances
                annotations[key] = f"list[{type(value[0]).__name__}]"
                defaults[key] = field(
                    default_factory=lambda val=value: copy.deepcopy(val)
                )
            else:
                # Handle list of primitives
                annotations[key] = (
                    f"list[{type(value[0]).__name__}]" if value else list[Any]
                )
                defaults[key] = field(
                    default_factory=lambda val=value: copy.deepcopy(val)
                )
        elif hasattr(value, "__class__") and not isinstance(
            value, str | int | float | bool | type(None)
        ):
            # Handle class instances (including dataclasses)
            if is_dataclass(value):
                # For dataclass instances, use the type and the instance as default
                annotations[key] = type(value)
                defaults[key] = field(
                    default_factory=lambda val=value: copy.deepcopy(val)
                )
            else:
                # For other class instances
                annotations[key] = type(value)
                defaults[key] = field(
                    default_factory=lambda val=value: copy.deepcopy(val)
                )
        elif isinstance(value, type):
            # Handle class types (not instances)
            annotations[key] = type
            defaults[key] = value
        elif value is None:
            annotations[key] = Any
            defaults[key] = None
        else:
            # Handle primitive types (immutable)
            annotations[key] = type(value)
            defaults[key] = value

    # Create the dataclass dynamically
    namespace = {"__annotations__": annotations, **defaults}

    # Create class and apply dataclass decorator
    dynamic_class = type(class_name, (), namespace)
    return dataclass(dynamic_class)


def nested_asdict(obj) -> dict:
    """Convert dataclass to dict including all attributes, preserving nested structures

    Args:
        obj: A dataclass instance or any object.
    Returns:
        dict: A dictionary representation of the dataclass.
    """
    if is_dataclass(obj):
        # Get all instance attributes (including non-field ones)
        result = {}
        for key, value in obj.__dict__.items():
            if is_dataclass(value):
                # Recursively convert nested dataclasses
                result[key] = nested_asdict(value)
            elif isinstance(value, list | tuple):
                result[key] = type(value)(
                    nested_asdict(item) if is_dataclass(item) else item
                    for item in value
                )
            elif isinstance(value, dict):
                result[key] = {
                    k: nested_asdict(v) if is_dataclass(v) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        return result
    return obj
