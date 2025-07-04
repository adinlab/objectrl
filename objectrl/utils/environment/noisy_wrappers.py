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

import gymnasium as gym
import numpy as np


class NoisyActionWrapper(gym.ActionWrapper):
    """
    A Gymnasium wrapper that injects noise into the agent's actions.
    For discrete action spaces, the action is randomly replaced with another action
    with a given probability. For continuous action spaces, Gaussian noise is added.

    Attributes:
        env (gym.Env): The environment to wrap.
        noise_act (float): Noise level for the action.
            - For discrete spaces: probability of replacing the action.
            - For continuous spaces: standard deviation of Gaussian noise.
    """

    def __init__(self, env: gym.Env, noise_act: float = 0.1) -> None:
        """
        Initialize the NoisyActionWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            noise_act (float): Noise level for the action.
                - For discrete spaces: probability of replacing the action.
                - For continuous spaces: standard deviation of Gaussian noise.
        Returns:
            None
        """
        super().__init__(env)
        self.noise_act = noise_act

    def step(self, action: np.ndarray) -> tuple:
        """
        Modify the action by injecting noise, then step the environment.

        Args:
            action: The original action chosen by the agent.
        Returns:
            Tuple: (obs, reward, terminated, truncated, info) after stepping the env.
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            if np.random.random() < self.noise_act:
                action = self.action_space.sample()
        else:
            eps = self.noise_act * np.random.randn(*action.shape)
            action = np.clip(
                action + eps, self.action_space.low, self.action_space.high
            )
        return self.env.step(action)


class NoisyObservationWrapper(gym.ObservationWrapper):
    """
    A Gymnasium wrapper that injects noise into observations.
    Adds Gaussian noise to array-based observations or to values in dictionary observations.

    Attributes:
        env (gym.Env): The environment to wrap.
        noise_obs (float): Standard deviation of Gaussian noise added to observations.
    """

    def __init__(self, env: gym.Env, noise_obs: float = 0.1) -> None:
        """
        Initialize the NoisyObservationWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            noise_obs (float): Standard deviation of Gaussian noise added to observations.
        Returns:
            None
        """
        super().__init__(env)
        self.noise_obs = noise_obs

    def observation(self, obs: np.ndarray | dict) -> np.ndarray | dict:
        """
        Apply Gaussian noise to the observation.

        Args:
            obs (np.ndarray or dict): The observation to be noised.
        Returns:
            np.ndarray or dict: The noisy observation.
        """
        if isinstance(obs, np.ndarray):
            eps = self.noise_obs * np.random.randn(*obs.shape)
            return obs + eps
        elif isinstance(obs, dict):
            noisy_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    eps = self.noise_obs * np.random.randn(*value.shape)
                    noisy_obs[key] = value + eps
                else:
                    noisy_obs[key] = value
            return noisy_obs
        else:
            raise ValueError("Unsupported observation type")
