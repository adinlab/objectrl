Actor
===========

This module defines the actor network responsible for selecting actions based on the current state.

Key Points:

- The :class:`Actor` models the policy function, mapping states to actions.

- Supports stochastic or deterministic action outputs depending on architecture.

- Designed to work seamlessly with the critic networks for policy gradient or actor-critic algorithms.

- Includes device management and flexible architecture configurations.

- Provides methods to sample or compute actions and to evaluate policy log probabilities if applicable.

Usage Notes:

- The actor receives states as input and outputs actions compatible with the environment.

- Can be extended or customized by modifying the underlying network architecture or sampling strategy.

.. important::

   The actor network must output actions consistent with the environment's action space.
   Ensure proper action normalization or bounding if required.

.. attention::

   For custom policy distributions (e.g., discrete, Gaussian), ensure your `act` method correctly samples actions.

Here are the detailed methods and attributes.

.. autoclass:: objectrl.models.basic.actor.Actor
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl
