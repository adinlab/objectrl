Reward Wrappers
===============

This module defines custom reward shaping techniques using Gymnasium’s :code:`RewardWrapper`.
These wrappers are used to modify the reward function to improve learning dynamics.

Descriptions
------------

- **PositionDelayWrapper**: Delays reward until the agent crosses a predefined position (`position_delay`). Also includes a control cost penalty to discourage erratic actions.

Classes
-------

.. autoclass:: objectrl.utils.environment.reward_wrappers.PositionDelayWrapper
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl
