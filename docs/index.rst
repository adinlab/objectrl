ObjectRL Documentation!
====================================

.. toctree::
   :maxdepth: 4
   :caption: Contents
   :hidden:

   getting_started
   examples/examples
   api/api

Welcome to the official documentation of **ObjectRL**, a deep reinforcement learning library designed for research and rapid prototyping by the `ADIN Lab <https://adinlab.github.io>`_ at the University of Southern Denmark.
Whether you're building new RL agents, experimenting with architectures, or integrating novel exploration strategies, ObjectRL provides the structure and flexibility you need.

What You'll Find
-----------------

This documentation is organized into several key sections to help you get started, understand the API, and explore advanced use cases:

- **Getting Started**: Learn how to install ObjectRL, set up your environment, and run your first experiment.
- **Examples**: Step-by-step tutorials for implementing your own agents, modifying existing ones, and understanding the internal structure. Furthermore, explore advanced RL techniques through real-world examples such as exploration bonuses, uncertainty modeling, and ensemble aggregation.
- **API**: Comprehensive documentation for ObjectRL’s modules, classes, and configuration system.

Supported Algorithms
---------------------

ObjectRL is designed primarily as a research and rapid prototyping framework. Our initial focus has been on implementing deep actor-critic algorithms for continuous control tasks in the MuJoCo, DM Control, and MetaWorld environment suites. The object-oriented design, however, enables future extensions to value-based methods and discrete action settings.

Currently, the main algorithms supported include:

- **Deep Deterministic Policy Gradient (DDPG)**: An early actor-critic algorithm for continuous control tasks.
  See :doc:`DDPG</api/models/ddpg>` page for details.
- **Twin Delayed Deep Deterministic Policy Gradient (TD3)**: An improvement over DDPG addressing overestimation bias with delayed policy updates.
  See :doc:`TD3</api/models/td3>` page for details.
- **Soft Actor-Critic (SAC)**: A popular off-policy actor-critic method with entropy regularization, well-suited for continuous control.
  See :doc:`SAC</api/models/sac>` page for details.
- **Proximal Policy Optimization (PPO)**: A widely-used on-policy actor-critic method.
  See :doc:`PPO</api/models/ppo>` page for details.
- **Randomized Ensemble Double Q-Learning (REDQ)**: An ensemble-based algorithm that improves value estimation and exploration.
  See :doc:`REDQ</api/models/redq>` page for details.
- **Distributional Random Network Distillation (DRND)**: Integrates exploration bonuses with distributional value estimates.
  See :doc:`DRND</api/models/drnd>` page for details.
- **Optimistic Actor-Critic (OAC)**: An actor-critic method incorporating optimism for better exploration.
  See :doc:`OAC</api/models/oac>` page for details.
- **PAC-Bayesian Actor-Critic (PBAC)**: An actor-critic algorithm that leverages PAC-Bayesian theory to improve exploration strategies.
  See :doc:`PBAC</api/models/pbac>` page for details.
- **Bayesian Neural Network SAC (BNN-SAC)**: Extends SAC with Bayesian critics to quantify epistemic uncertainty (currently in examples).
  See :doc:`Example 3</examples/example_3>` page for details.
- **Deep Q-Network (DQN)**: A classic value-based method primarily for discrete action spaces (currently in examples).
  See :doc:`Tutorial 2</examples/tutorial_2>` page for details.

The library's current strength and focus lie in continuous control and actor-critic methods. However, value-based and discrete action algorithms are supported experimentally.
This foundation facilitates rapid algorithmic development and experimentation in research.

Citation
--------

If you use ObjectRL in your research, please consider citing the following paper:

.. code-block:: bibtex

   @article{baykal2025objectrl,
      title={ObjectRL: An Object-Oriented Reinforcement Learning Codebase}, 
      author={Baykal, Gulcin and  Akg{\"u}l, Abdullah and Haussmann, Manuel and Tasdighi, Bahareh and Werge, Nicklas and Wu Yi-Shan and Kandemir, Melih},
      year={2025},
      journal={arXiv preprint arXiv:2507.03487}
    }