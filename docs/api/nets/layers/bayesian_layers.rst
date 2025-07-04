Bayesian Layers
===============

This module provides implementations of Bayesian neural network layers designed for uncertainty modeling in deep learning architectures. These layers support different variational inference techniques and activation moment propagation.

Detailed Descriptions
---------------------

BayesianLinear
~~~~~~~~~~~~~~~~

Abstract base class for Bayesian neural network layers.

- Defines core attributes like `weight_mu`, `weight_rho`, `bias_mu`, `bias_rho`.
- Supports optional bias, prior distributions for weights.
- Allows softplus transformation for standard deviation parameters.
- Includes MAP mode and KL divergence computation.

BBBLinear
~~~~~~~~~~~

Implements a Bayesian layer using Bayes by Backprop:

- Samples weights and biases during forward pass from learned posterior.
- In MAP mode, uses only mean parameters without sampling.

LRLinear
~~~~~~~~

Implements a Bayesian layer using the Local Reparameterization Trick:

- Samples output activations instead of weights for more efficient variance reduction.
- Propagates mean and variance through layers.
- Supports MAP mode.

CLTLinear
~~~~~~~~~

Implements Bayesian layer using Central Limit Theorem (CLT) approximations:

- Supports ReLU and CReLU activations.
- Propagates mean and variance analytically through the network.
- Supports input/output layer distinctions and MAP mode.

CLTLinearDet
~~~~~~~~~~~~

Deterministic version of CLTLinear:

- Disables uncertainty modeling by removing learned standard deviations.
- Overrides methods to raise errors for standard deviation and KL divergence calls.
- Supports MAP mode and variance propagation accordingly.

Usage Example
-------------

.. code-block:: python

    import torch
    from nets.layers.bayesian_layers import BBBLinear

    layer = BBBLinear(in_features=128, out_features=64, bias=True)
    x = torch.randn(32, 128)
    output = layer(x)
    print(output.shape)  # torch.Size([32, 64])

Notes
-----

- The :code:`map()` method in :code:`BayesianLinear` switches between MAP (deterministic) and sampling modes.
- The KL divergence calculation sums over all parameters for regularization.
- :code:`CLTLinear` uses moment matching and normal CDF/PDF to approximate nonlinear activations.
- The deterministic variant :code:`CLTLinearDet` raises errors if variance or KL methods are called.



Classes
-------

.. autoclass:: objectrl.nets.layers.bayesian_layers.BayesianLinear
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.layers.bayesian_layers.BBBLinear
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.layers.bayesian_layers.LRLinear
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.layers.bayesian_layers.CLTLinear
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.nets.layers.bayesian_layers.CLTLinearDet
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

