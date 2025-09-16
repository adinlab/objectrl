DMC Wrappers
=============

Descriptions
------------

This module provides Gymnasium-compatible wrappers for DeepMind Control Suite (DMC) environments.
It allows DMC environments to be used interchangeably with Gymnasium-based reinforcement learning
algorithms by providing `observation_space`, `action_space`, and standard `step`/`reset`/`render` APIs.

Key features:

- Convert `dm_env` specifications to Gymnasium spaces (`Box` or `Dict`) via `dmc_spec2gym_space`.
- Wrap DMC tasks so they can be used with Gymnasium-compatible RL code.
- Provide standard Gym-style `step` and `reset` methods.
- Support deterministic seeding and optional rendering.

Functions
---------

.. autoclass:: objectrl.utils.environment.dmc_wrappers.DMCEnv
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:

Classes
-------

.. autoclass:: objectrl.utils.environment.dmc_wrappers.DMCEnv
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl
