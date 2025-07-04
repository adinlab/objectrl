Custom Loss Functions
=====================

This module implements probabilistic and PAC-Bayesian loss functions for uncertainty-aware learning.

Included Losses:

- :class:`ProbabilisticLoss`: Base class for probabilistic losses supporting different reduction modes.

- :class:`PACBayesLoss`: Combines empirical risk with complexity regularization for PAC-Bayesian learning.

Extending
---------

.. important::
    
    If you need to implement a custom loss function, add it to this module for consistency and integration with existing training pipelines. Extend from :class:`ProbabilisticLoss` for probabilistic outputs or from PyTorch's standard losses otherwise.

Here are the detailed methods and attributes for the implemented losses.

ProbabilisticLoss
------------------

.. autoclass:: objectrl.models.basic.loss.ProbabilisticLoss
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

PACBayesLoss
------------------

.. autoclass:: objectrl.models.basic.loss.PACBayesLoss
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl
