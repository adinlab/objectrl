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
Model configuration classes for reinforcement learning agents.

This module provides structured dataclasses to define the architecture and
hyperparameters of actor and critic networks used in reinforcement learning
algorithms. It includes:

- `ActorConfig`: Configuration of the actor (policy) network.
- `CriticConfig`: Configuration of the critic (value function) network.
- `ModelConfig`: Wrapper for selecting and loading model-specific configurations.

All classes support loading from default model templates via `.from_config() and serializing to dictionaries via `.to_dict()`.
"""

from dataclasses import asdict, dataclass, fields
from typing import Literal

from objectrl.config.model_configs import actor_configs, critic_configs, model_configs
from objectrl.config.utils import create_field_dict, enhanced_repr


# [start-actor-config]
@enhanced_repr
@dataclass
class ActorConfig:
    """
    Configuration for the actor network in RL models.

    Attributes:
        depth (int): Number of hidden layers.
        width (int): Number of units per hidden layer.
        norm (bool): Enable normalization layers.
        activation (str): Activation function ('relu' or 'crelu'). User should add other activation functions if needed.
        has_target (bool): Whether to use a target actor network.
        n_actors (int): Number of parallel actor networks.
        reset (bool): Whether to reset the actor.
        n_heads (int): Number of heads in a multi-head actor.
    """

    depth: int = 3
    width: int = 256
    # Disable normalization in the actor network
    norm: bool = False
    # Activation function for the actor network
    activation: Literal["relu", "crelu"] = "relu"
    has_target: bool = False
    n_actors: int = 1
    reset: bool = False
    n_heads: int = 1
    max_grad_norm: float = 0.0  # Optional maximum gradient norm for clipping

    @classmethod
    def from_config(cls, config: dict, model_name: str) -> "ActorConfig":
        """
        Create an ActorConfig from a custom config dict and default model values.

        Args:
            config (dict): Custom configuration parameters.
            model_name (str): The name of the model whose defaults to load.
        Returns:
            ActorConfig: Initialized with both default and overridden settings.
        """
        config = create_field_dict(actor_configs[model_name]) | config

        known_names = {field.name for field in fields(cls)}
        known_attr = {k: v for k, v in config.items() if k in known_names}
        extra_attr = {k: v for k, v in config.items() if k not in known_names}

        instance = cls(**known_attr)

        for k, v in extra_attr.items():
            setattr(instance, k, v)
        return instance

    def to_dict(self) -> dict:
        """
        Convert the ActorConfig to a dictionary.

        Args:
            None
        Returns:
            dict: Dictionary representation of the config.
        """
        return asdict(self)


# [end-actor-config]


# [start-critic-config]
@enhanced_repr
@dataclass
class CriticConfig:
    """
    Configuration for the critic network in RL models.

    Attributes:
        depth (int): Number of hidden layers.
        width (int): Number of units per hidden layer.
        norm (bool): Normalization layers.
        activation (str): Activation function ('relu' or 'crelu'). User should add other activation functions if needed.
        n_members (int): Number of critic networks.
        reduce (str): Method for reducing outputs of an ensemble of critics.
        target_reduce (str): Method for reducing outputs of an ensemble of target critics.
        has_target (bool): Whether to use a target critic network.
        reset (bool): Whether to reset the critic.
    """

    depth: int = 3
    width: int = 256
    norm: bool = False
    activation: Literal["relu", "crelu"] = "relu"
    n_members: int = 2
    reduce: str = "min"
    target_reduce: str = "min"
    has_target: bool = True
    reset: bool = False
    max_grad_norm: float = 0.0  # Optional maximum gradient norm for clipping

    @classmethod
    def from_config(cls, config: dict, model_name: str) -> "CriticConfig":
        """
        Construct a CriticConfig from a user-defined config and base model name.

        Args:
            config (dict): Configuration overrides.
            model_name (str): Name of the model to fetch default critic settings from.

        Returns:
            CriticConfig: Populated configuration object.
        """
        config = create_field_dict(critic_configs[model_name]) | config

        known_names = {field.name for field in fields(cls)}
        known_attr = {k: v for k, v in config.items() if k in known_names}
        extra_attr = {k: v for k, v in config.items() if k not in known_names}

        instance = cls(**known_attr)

        for k, v in extra_attr.items():
            setattr(instance, k, v)

        return instance

    def to_dict(self):
        """
        Convert this CriticConfig to a dictionary.
        """
        return asdict(self)


# [end-critic-config]


# [start-model-config]
@dataclass
class ModelConfig:
    """
    Base configuration for a model.

    Attributes:
        name (str): Identifier for the model type.
    """

    name: str = "abstract"

    @classmethod
    def from_config(cls, config: dict, model_name: str) -> "ModelConfig":
        """
        Construct a ModelConfig from a user config and model definition.

        Args:
            config (dict): Configuration overrides.
            model_name (str): Model name to use as the default template.
        Returns:
            ModelConfig: A fully populated configuration object.
        """
        config = model_configs[model_name] | config

        known_names = {field.name for field in fields(cls)}
        known_attr = {k: v for k, v in config.items() if k in known_names}
        extra_attr = {k: v for k, v in config.items() if k not in known_names}

        instance = cls(**known_attr)

        for k, v in extra_attr.items():
            setattr(instance, k, v)

        return instance

    def to_dict(self) -> dict:
        """
        Convert this ModelConfig to a dictionary.

        Args:
            None
        Returns:
            dict: Dictionary representation of the config.
        """
        return asdict(self)


# [end-model-config]
