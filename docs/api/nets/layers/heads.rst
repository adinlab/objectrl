Heads
=====

This module provides various head layers that convert network outputs into probability distributions over actions. These heads are commonly used in reinforcement learning to model stochastic or deterministic policies.

Descriptions
-------------

GaussianHead
~~~~~~~~~~~~

- Splits the input into mean and log-variance to form a `Normal` distribution.
- Returns a sample (`action`), its `log-probability`, and the underlying `distribution`.

SquashedGaussianHead
~~~~~~~~~~~~~~~~~~~~

- Outputs a `Tanh`-squashed Gaussian distribution for bounded action spaces.
- Supports both training (stochastic sampling) and evaluation (mean of samples) modes.
- Returns the `action`, `log-probability` (if training), and `distribution`.

CategoricalHead
~~~~~~~~~~~~~~~

- Converts logits into a `Categorical` distribution for discrete actions.
- Uses `softmax` to derive probabilities.
- Returns a sampled `action`, its `log-probability`, and the `distribution`.

DeterministicHead
~~~~~~~~~~~~~~~~~

- Pass-through head for deterministic policies.
- Directly returns the input as the `action`.

Classes
-------

.. autoclass:: objectrl.nets.layers.heads.GaussianHead
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.layers.heads.SquashedGaussianHead
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.layers.heads.CategoricalHead
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.layers.heads.DeterministicHead
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

Usage Example
-------------

.. code-block:: python

    from nets.layers.heads import GaussianHead

    head = GaussianHead(n=4)
    output = head(torch.randn(1, 8))  # 4 for mean, 4 for logvar
    print(output["action"].shape)  # torch.Size([1, 4])

Notes
-----

- These heads abstract the output distribution from policy networks.
- Useful in actor-critic methods for exploration and entropy regularization.
- :code:`SquashedGaussianHead` ensures outputs stay in [-1, 1] using :code:`TanhTransform`.