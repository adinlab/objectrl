Actor Networks
==============

This module provides neural network architectures for policy learning in reinforcement learning.  
It includes both deterministic and probabilistic actor models.

Descriptions
-------------

- **ActorNetProbabilistic**: Outputs a squashed Gaussian distribution over actions. Useful in stochastic policies like Soft Actor-Critic (SAC).
- **ActorNet**: Outputs deterministic actions, typically used in deterministic policy gradients or for evaluation.

Classes
-------

.. autoclass:: objectrl.nets.actor_nets.ActorNetProbabilistic
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.actor_nets.ActorNet
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. note::
    
    Both classes support multiple output heads (:code:`n_heads`) for ensemble learning or multi-policy settings.

