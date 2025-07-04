Examples
================

ObjectRL is designed based on object-oriented design principles, making it easy to implement new reinforcement learning agents and experiment with novel ideas. This section contains hands-on tutorials and real-world examples to guide you through building and modifying agents using ObjectRL.

These examples illustrate:

- How to implement new algorithms step-by-step
- How to reuse and extend existing models with minimal changes
- How to integrate exploration strategies and uncertainty estimation into standard methods

Whether you're starting from scratch or adapting a state-of-the-art algorithm, these examples are designed to enhance your understanding and intuition. 

Available Tutorials and Examples
--------------------------------

- **Tutorial 1: Step-by-Step AC Implementation**  
  Learn how to implement a custom Actor-Critic agent from scratch, including writing your own agent class, networks, and configuration dataclasses. Ideal for understanding the full pipeline of ObjectRL.

- **Tutorial 2: Build Your Own Model**  
  Discover how to transition from Actor-Critic to value-based methods like DQN. This tutorial guides you through inheritance-based design to reduce boilerplate when modifying RL algorithms.

- **Example 1: Adapting SAC to DRND**  
  A practical case study showing how to modify Soft Actor-Critic to use an ensemble-based exploration bonus. Demonstrates how to plug in new loss terms, networks, and training procedures.

- **Example 2: Adapting SAC to REDQ**  
  Discover how modifying the critic ensemble aggregation strategy in SAC leads to REDQ. Demonstrates how overriding the reduction method can easily lead to a new model.

- **Example 3: Uncertainty-Aware SAC with Bayesian Neural Networks**  
  Shows how to incorporate Bayesian layers into the SAC framework to estimate epistemic uncertainty. Includes custom networks, configuration overrides, and a working comparison setup.

How to Use These Examples
--------------------------

Each page provides ready-to-run code snippets, configuration instructions, and explanations of design decisions. You are encouraged to:

- Copy and adapt these templates for your own research
- Mix components from multiple examples to explore new algorithmic ideas
- Extend the configuration files to suit more advanced workflows

All experiments can be run using:

.. code-block:: bash

    python objectrl/main.py --model.name <your_model_name> --env.name <env_name>

For more control and reproducibility, write your own YAML config files.


.. toctree::
    :maxdepth: 2
    :hidden:

    tutorial_1
    tutorial_2
    example_1
    example_2
    example_3