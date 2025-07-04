API
====

.. toctree::
    :maxdepth: 3
    :hidden:
    
    agents
    config
    experiments
    loggers
    models/models
    nets/nets
    replay_buffers
    utils/index

.. figure:: ../_static/imgs/ac.png
   :width: 100%
   :align: center
   :alt: General Deep Actor Critic architecture.
   
   General Deep Actor Critic architecture.


.. raw:: html

   <p>The figure below illustrates the core design of ObjectRL.</p>


Overview
~~~~~~~~

ObjectRL's API is organized into components that allow for flexible experimentation and development of reinforcement learning algorithms. Key modules include:

- **Agents**: Implementations of various RL agents. Currently, it includes the base Agent class.
- **Configurations**: Flexible, dataclass-based configurations that control hyperparameters and architectural choices.
- **Experiments**: Scripts and utilities for running training and evaluation workflows.
- **Loggers**: Tools for experiment tracking and result visualization.
- **Models**: Algorithms for actors, critics, and actor-critics.
- **Nets**: Building blocks such as policy and value networks.
- **Replay Buffers**: Experience storage.
- **Utils**: Helper functions, data structures, and network utilities.

This structure is designed to separate concerns clearly, enabling rapid prototyping and easy customization.

Getting Started
~~~~~~~~~~~~~~~

If you’re new to ObjectRL, check out the :doc:`/getting_started` page for installation instructions and quick-start tutorials.

Further Reading
~~~~~~~~~~~~~~~

For practical use cases and advanced examples, see the :doc:`/examples/examples` page, and for in-depth algorithm references, stay in this section.