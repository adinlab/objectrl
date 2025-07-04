Nets
====

.. toctree::
   :maxdepth: 2
   :hidden:

   layers/layers
   actor_nets
   critic_nets

The `nets` module contains the core neural network building blocks used across ObjectRL's algorithms. It is organized to facilitate easy customization of network architectures.

Module Structure
----------------

- **layers**: Contains reusable neural network layers and utilities such as Bayesian layers, normalization layers, and other custom components that can be used by actor and critic networks.

- **actor_nets.py**: Defines policy network architectures responsible for selecting actions given states. These networks encapsulate the actor's behavior and can be customized or extended for new algorithms.

- **critic_nets.py**: Defines value or Q-function network architectures responsible for estimating expected returns. Critic networks support various architectures, including standard MLPs and Bayesian models.

Purpose and Usage
-----------------

This organization allows you to:

- Easily swap or extend network architectures for actors and critics without modifying the core algorithm logic.
- Reuse custom layers across different network types.
- Implement novel network designs to experiment with uncertainty estimation, exploration bonuses, or other advanced features.