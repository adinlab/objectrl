Critic Networks
===============

This module defines various critic (value function) network architectures used in reinforcement learning, including deterministic, probabilistic, and Bayesian variants.

Descriptions
-------------

- **CriticNet**: A basic deterministic Q-network estimating expected returns.
- **ValueNet**: A simplified Q-network for value estimation, ignoring action input.
- **CriticNetProbabilistic**: Outputs both mean and uncertainty in Q-values.
- **BNNCriticNet**: A Bayesian Q-network using the local reparameterization trick.
- **EMstyle**: An encoder network for expectation-maximization-style latent modeling.
- **DQNNet**: A Q-network architecture designed for discrete action spaces (e.g., DQN).

Classes
-------

.. autoclass:: objectrl.nets.critic_nets.CriticNet
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.critic_nets.ValueNet
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.critic_nets.CriticNetProbabilistic
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.critic_nets.BNNCriticNet
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.critic_nets.EMstyle
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.critic_nets.DQNNet
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl


