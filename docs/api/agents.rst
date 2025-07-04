Agents
======

Overview
--------

The `agents/` module defines the `Agent` base class, which provides a common interface and core utilities for reinforcement learning agents.

This abstract base class:

- Manages experience replay
- Provides model saving and loading functionality
- Interfaces with environment and logger
- Requires implementation of core methods: `reset`, `learn`, and `select_action`

.. important::

   All reinforcement learning agents in this repository should inherit from :class:`Agent` to ensure consistency and reusability of code.

.. note::

   The :class:`Agent` class is designed to be environment-agnostic. It does not assume a specific observation or action space format.
   You are expected to handle environment-specific preprocessing in your agent subclasses.


Base Class: Agent
-----------------

.. autoclass:: objectrl.agents.base_agent.Agent
    :members:
    :undoc-members:
    :show-inheritance:

Extending Agent
^^^^^^^^^^^^^^^^

To create a new agent, inherit ``Agent`` and implement the required methods. See our example agents in the `objectrl/models/` directory for reference. Furthermore, see our example :doc:`Build Your Own Model </examples/tutorial_2>` for a practical example of extending the ``Agent`` class.

.. attention::

   When implementing your own agent, you **must** override the `reset`, `learn`, and `select_action` methods.
   These are the fundamental hooks that ObjectRL uses to interact with your agent.


Design Philosophy
^^^^^^^^^^^^^^^^^^

This interface was designed with flexibility in mind. It separates concerns cleanly:

- **Memory management** is delegated to `experience_memory`
- **Logging** is handled via a pluggable `Logger`
- **Training steps** are decoupled via `learn()` and `select_action()` to support both offline and online training

.. tip::

   If you're building a new algorithm, start by extending :class:`Agent` and reusing existing components like actors, critics, and loss functions from the `models/` module.
   This will drastically reduce boilerplate.



