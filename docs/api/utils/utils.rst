Utility Functions
=================

This module contains simple utility functions to facilitate conversions between
NumPy arrays and PyTorch tensors, and shape validation.

Functions
---------

.. autofunction:: objectrl.utils.utils.totorch
.. autofunction:: objectrl.utils.utils.tonumpy
.. autofunction:: objectrl.utils.utils.toint
.. autofunction:: objectrl.utils.utils.dim_check

Function Details
----------------

totorch(x, dtype=torch.float32, device="cuda") -> torch.Tensor
    Converts a NumPy array or compatible object into a PyTorch tensor with specified dtype and device.

tonumpy(x) -> numpy.ndarray
    Converts a PyTorch tensor to a NumPy ndarray, moving it to CPU and detaching from the computation graph.

toint(x) -> int
    Converts a PyTorch tensor containing a single value to a Python integer.

dim_check(tensor1, tensor2) -> None
    Asserts that two tensors have the same shape. Raises AssertionError if they differ.

Example Usage
-------------

.. code-block:: python

    import numpy as np
    import torch

    arr = np.array([1, 2, 3])
    tensor = totorch(arr, dtype=torch.float32, device="cpu")
    back_to_np = tonumpy(tensor)
    integer_value = toint(torch.tensor(5))
    dim_check(tensor, torch.ones_like(tensor))  # no error
