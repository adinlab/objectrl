Custom Activation Functions
===========================

This module defines custom activation functions used in neural networks.

Description
-----------

**CReLU** (Concatenated ReLU) [#crelu]_ applies ReLU activation to both the input and its negation,
then concatenates the results along the last dimension:

.. math::

    \text{CReLU}(x) = \text{ReLU}([x, -x])

This doubles the number of features and enhances the representational capacity of the network.


Classes
-------

.. autoclass:: objectrl.utils.custom_act.CReLU
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl


.. rubric:: References

.. [#crelu] Abbas Z. et al., (2023), *Loss of Plasticity in Continual Deep Reinforcement Learning*