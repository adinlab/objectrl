Critic and CriticEnsemble
=================================

This module defines the core critic network and its ensemble variant for Q-value estimation in reinforcement learning.

Key Points:

- The :class:`Critic` class estimates Q-values for state-action pairs with optional target networks for stabilized training.

- The :class:`CriticEnsemble` class manages an ensemble of critics to improve robustness by aggregating multiple Q-value estimates.

- Both classes support soft target network updates via Polyak averaging.

- The ensemble uses a generic :class:`Ensemble` container for parallelizing multiple critics efficiently.

- Access individual critics from the ensemble with indexing (e.g., `ensemble[0]`).

- Reduction methods (:class:`min` or :class:`mean`) for ensemble Q-values are configurable.

Usage Notes:

- The input to critics is prepared by concatenating state and action tensors.

- The target network, if enabled, must be initialized before use.

- The :class:`update` method applies the Bellman update to the ensemble.

- The :class:`get_bellman_target` method is abstract and must be implemented in subclasses to provide the target for training.

.. important::

   Always initialize the target network before training to stabilize updates.

.. warning::

   Using an improper reduction method on Q-values from multiple critics may destabilize training.
   Supported reductions are :class:`min` and :class:`mean`.

Here are the detailed methods and attributes for both classes.

Critic
-------

.. autoclass:: objectrl.models.basic.critic.Critic
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

CriticEnsemble
-----------------

.. autoclass:: objectrl.models.basic.critic.CriticEnsemble
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

