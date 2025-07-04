Network Utilities
=================

This module provides utility functions and neural network classes commonly used for actor-critic algorithms in reinforcement learning, including deterministic and Bayesian MLP architectures.

Functions
---------

.. autofunction:: objectrl.utils.net_utils.create_optimizer
.. autofunction:: objectrl.utils.net_utils.create_loss

Classes
-------

MLP
~~~
.. autoclass:: objectrl.utils.net_utils.MLP
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

BayesianMLP
~~~~~~~~~~~
.. autoclass:: objectrl.utils.net_utils.BayesianMLP
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

Notes
-----

- The :code:`create_optimizer` function dynamically selects and configures an optimizer from **torch.optim**.
- The :code:`create_loss` function supports both PyTorch and custom loss modules (e.g., from **objectrl.models.basic.loss**).
- The :code:`MLP` class supports `ReLU` and `CReLU` activations and optional `LayerNorm`.
- The :code:`BayesianMLP` supports multiple Bayesian linear layer types:
  
  - `"bbb"`: Bayes by Backprop
  - `"lr"`: Local Reparameterization
  - `"clt"`: Central Limit Theorem
  - `"cltdet"`: Deterministic CLT

Example
-------

.. code-block:: python

    net = MLP(128, 64, depth=3, width=256, act="crelu", has_norm=True)
    out = net(torch.randn(32, 128))

    bayesian_net = BayesianMLP(128, 64, depth=3, width=256, layer_type="lr")
    out_bnn = bayesian_net(torch.randn(32, 128))
