Make Environment Utility
==========================

This module defines a utility function to create environments with support
for noise injection, reward shaping, action rescaling, and consistent seeding. 
Environment names are mapped to standardized strings, using Gymnasium IDs for MuJoCo tasks and custom identifiers for DM Control tasks.

Function
--------

.. autofunction:: objectrl.utils.make_env.make_env

Key Features
------------

- **Rescales** continuous action spaces to the range ``[-1, 1]``.
- Optionally adds:
  
  - **Noisy actions** via ``NoisyActionWrapper``
  - **Noisy observations** via ``NoisyObservationWrapper``
  - **Reward shaping** via ``PositionDelayWrapper``

- **Reproducibility** through seeding for:
  
  - Gym environment
  - Action and observation spaces
  - NumPy and PyTorch RNGs

Usage Example
-------------

.. code-block:: python

    env = make_env("HalfCheetah-v4", 0, config.env)
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)