Utils
=====

This module provides a collection of utility scripts and helper classes designed to support
various aspects of reinforcement learning experiments, including:

- Environment wrappers that add noise to actions and observations, and modify rewards.
- Custom activation functions to enhance neural network expressiveness.
- Tools for collecting, processing, and visualizing evaluation results across models and environments.
- Functions to create and configure Gym environments with consistent seeding and wrapping.
- Neural network utilities, including MLP and Bayesian MLP architectures and optimizer/loss creators.
- General-purpose tensor conversion and shape-checking utilities for PyTorch and NumPy interoperability.

.. toctree::
   :maxdepth: 2
   :hidden:

   environment/environment
   custom_act
   harvest_utils
   make_env
   net_utils
   utils