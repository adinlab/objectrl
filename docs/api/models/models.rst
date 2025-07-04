Models
======

This section documents the core model implementations used across the ObjectRL library.
Each model encapsulates a reinforcement learning agent and its underlying architecture, including actor, critic, loss functions, and update logic.

We divide the models into two main categories:

- **Basic Building Blocks**: General-purpose modules such as actors, critics, ensembles, and customizable loss functions.
- **Algorithm-Specific Models**: Full algorithm implementations.

.. note::

   This structure in ObjectRL allows users to flexibly mix and match model components
   for custom algorithm development.


.. attention::

   If you're implementing a new algorithm or extending an existing one, refer to the UML diagrams and configuration templates provided
   in each algorithm's documentation for structural guidance.


.. toctree::
    :maxdepth: 2
    :hidden:

    basic/basic
    ddpg
    drnd
    oac
    pbac
    ppo
    redq
    sac
    td3