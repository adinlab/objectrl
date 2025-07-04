ActorCritic
=================

This module combines an actor and a critic into a single model to enable joint policy and value learning.

Key Points:

- Integrates the policy network (actor) with a value estimator (critic) for efficient reinforcement learning.

- Supports synchronized updates and coordinated forward passes.

- Provides access to actor and critic components separately when needed.

- Facilitates algorithms like DDPG, TD3, or SAC that rely on actor-critic architecture.

- Manages device placement and configuration consistency between components.

Usage Notes:

- The :class:`ActorCritic` class simplifies the training pipeline by encapsulating both networks.

- Enables computation of both action selections and value predictions from the same input states.

.. attention::

   The synchronization between actor and critic updates is crucial for stable training.

.. note::

   Target networks, if used, should be updated with care using soft updates (Polyak averaging).

Here are the detailed methods and attributes.

.. autoclass:: objectrl.models.basic.ac.ActorCritic
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

