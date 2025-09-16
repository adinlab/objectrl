MetaWorld Wrappers
====================

Descriptions
-------------

This module provides a Gymnasium wrapper for Meta-World environments
that replaces the environment's reward with the binary success signal.
The reward will be either `-1.0` (failure) or `0.0` (success).

Classes
-------

.. autoclass:: objectrl.utils.environment.metaworld_wrappers.SparsifyRewardWrapper
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl
