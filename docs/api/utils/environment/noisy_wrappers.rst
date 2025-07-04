Noisy Wrappers
==============

This module provides wrappers that introduce noise into the agent's interaction with the environment, either through action perturbations or observation corruption.

Descriptions
------------

- **NoisyActionWrapper**: Adds noise to the actions. For discrete actions, actions may be randomly replaced. For continuous actions, Gaussian noise is added.
- **NoisyObservationWrapper**: Adds Gaussian noise to observations. Supports both NumPy arrays and dictionaries.

Classes
-------

.. autoclass:: objectrl.utils.environment.noisy_wrappers.NoisyActionWrapper
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.utils.environment.noisy_wrappers.NoisyObservationWrapper
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl
