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

import copy
from collections import OrderedDict

import dm_env
import numpy as np
from dm_control import suite
from gymnasium import core, spaces

TimeStep = tuple[dict[str, np.ndarray], float, bool, bool, dict]


def dmc_spec2gym_space(
    spec: dm_env.specs.Array | dict | OrderedDict,  # type: ignore[attr-defined] // ignore the dm_env.specs error
) -> spaces.Space:
    """
    Convert a dm_env spec (Array or BoundedArray) into a Gymnasium Space.

    Args:
        spec (Union[dm_env.specs.Array, Dict, OrderedDict]):
            The dm_env spec to convert.

    Returns:
        spaces.Space: A corresponding Gymnasium space.
    """
    if isinstance(spec, (OrderedDict, dict)):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):  # type: ignore[attr-defined] // ignore the dm_env.specs error
        return spaces.Box(
            low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype
        )
    elif isinstance(spec, dm_env.specs.Array):  # type: ignore[attr-defined] // ignore the dm_env.specs error
        return spaces.Box(
            low=-float("inf"), high=float("inf"), shape=spec.shape, dtype=spec.dtype
        )
    else:
        raise NotImplementedError(f"Unsupported spec type: {type(spec)}")


class DMCEnv(core.Env):
    """
    A Gymnasium-compatible wrapper for DeepMind Control Suite environments.

    This class adapts dm_control environments to the Gymnasium API
    by exposing `observation_space`, `action_space`, and standard
    `step`/`reset`/`render` methods.

    Attributes:
        domain_name (str): The DMC domain name (e.g., "cartpole").
        task_name (str): The DMC task name (e.g., "swingup").
        action_space (spaces.Space): The Gymnasium action space.
        observation_space (spaces.Space): The Gymnasium observation space.
    """

    def __init__(
        self,
        domain_name: str | None = None,
        task_name: str | None = None,
        env: dm_env.Environment | None = None,
        task_kwargs: dict | None = None,
        environment_kwargs: dict | None = None,
    ) -> None:
        """
        Initialize the DMCEnv wrapper.

        Args:
            domain_name (Optional[str]): Name of the control suite domain.
            task_name (Optional[str]): Name of the task in the domain.
            env (Optional[dm_env.Environment]): Pre-created dm_env environment.
            task_kwargs (Optional[Dict]): Keyword arguments for task creation.
                Must include a 'random' seed for determinism.
            environment_kwargs (Optional[Dict]): Extra arguments for environment.
        """
        task_kwargs = {} if task_kwargs is None else task_kwargs

        assert (
            "random" in task_kwargs
        ), "Please specify a seed in task_kwargs['random'] for deterministic behaviour."
        assert env is not None or (
            domain_name is not None and task_name is not None
        ), "You must provide either an environment or domain and task names."

        if env is None:
            env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=True,
            )

        self._env: dm_env.Environment = env
        self.domain_name: str | None = domain_name
        self.task_name: str | None = task_name
        self.action_space: spaces.Space = dmc_spec2gym_space(self._env.action_spec())
        self.observation_space: spaces.Space = dmc_spec2gym_space(
            self._env.observation_spec()
        )

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying dm_env environment."""
        return getattr(self._env, name)

    def step(self, action: np.ndarray) -> TimeStep:
        """
        Take a step in the environment.

        Args:
            action (np.ndarray): Action to apply.

        Returns:
            TimeStep: A Gymnasium-style tuple:
                (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), "Action not in action_space."

        time_step = self._env.step(action)
        reward: float = time_step.reward or 0.0
        done: bool = time_step.last()
        obs: dict[str, np.ndarray] = time_step.observation

        info: dict = {}
        trunc: bool = done and (time_step.discount == 1.0)
        term: bool = done and (time_step.discount != 1.0)
        if trunc:
            info["TimeLimit.truncated"] = True

        return obs, reward, term, trunc, info

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """
        Reset the environment.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[Dict]): Extra reset options.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict]: Initial observation and info dict.
        """
        super().reset(seed=seed)
        time_step = self._env.reset()
        info: dict = {}
        return time_step.observation, info

    def render(
        self,
        mode: str = "rgb_array",
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
    ) -> np.ndarray:
        """
        Render the environment as an RGB array.

        Args:
            mode (str): Must be "rgb_array".
            height (int): Image height (default: 84).
            width (int): Image width (default: 84).
            camera_id (int): Camera ID to render from.

        Returns:
            np.ndarray: Rendered image of shape (H, W, 3).
        """
        assert mode == "rgb_array", f"Only support rgb_array mode, got {mode}."
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
