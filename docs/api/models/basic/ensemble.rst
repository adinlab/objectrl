Ensemble
=========

Generic neural network ensemble container designed to hold multiple model instances with shared architecture but independent parameters.

Key Points:

- Provides batched parameter storage for efficient parallel forward passes using :class:`torch.func`.

- Supports expanding input tensors to match ensemble size.

- Allows extraction of single ensemble members as standalone models.


- Useful as a building block for model ensembles such as the :class:`CriticEnsemble`.

.. note::

   Increasing the number of ensemble members improves robustness but increases computational cost.

.. warning::

   Incorrect parameter extraction when retrieving single ensemble members can lead to inconsistent behavior.

.. warning::

    Set :code:`sequential=True` if the underlying ensemble members are stateful, e.g., use batch normalization. The vectorized version using parallel forward passes will not update such layers.


Here are the detailed methods and attributes.

.. autoclass:: objectrl.models.basic.ensemble.Ensemble
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl