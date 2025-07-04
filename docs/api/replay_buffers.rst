Replay Buffers
==============

Overview
--------

The `replay_buffers/` module contains the `ReplayBuffer` class, which provides an efficient and extensible experience replay mechanism for reinforcement learning agents.

This buffer:

- Stores experience tuples using TorchRL's **TensorDictReplayBuffer**  
- Uses **LazyMemmapStorage** for efficient CPU memory handling  
- Supports **random, index-based, and full-batch** sampling  
- Allows **metadata saving and loading** for checkpointing  
- Enables **epoch-based training** via internal iterators  

.. note::

   This replay buffer is compatible with both CPU and GPU workflows by using a `device` (for training) and a `storing_device` (for storage) during initialization.

ReplayBuffer Class
------------------

.. autoclass:: objectrl.replay_buffers.experience_memory.ReplayBuffer
    :members:
    :undoc-members:
    :show-inheritance:

ReplayBuffer Management
------------------------

Initialization
^^^^^^^^^^^^^^^

You must now initialize the replay buffer by **explicitly specifying the devices and buffer size**:

.. code-block:: python

    from objectrl.replay_buffers.experience_memory import ReplayBuffer
    buffer = ReplayBuffer(
        device=torch.device("cuda"),             # Training device
        storing_device=torch.device("cpu"),      # Storage device
        buffer_size=100_000                      # Maximum number of transitions
    )

Adding Experience
^^^^^^^^^^^^^^^^^^^^

You can add either individual experiences or batches in the form of TorchRL `TensorDict` objects:

.. code-block:: python

    buffer.add(single_experience)       # Single transition
    buffer.add_batch(batch_experience)  # Batch of transitions

Sampling
^^^^^^^^^^

You can sample in multiple ways:

- **Random batch**: 

  .. code-block:: python

      batch = buffer.sample_batch(64)

- **By index**:

  .. code-block:: python

      batch = buffer.sample_by_index([0, 5, 9])

- **Specific fields**:

  .. code-block:: python

      obs_act = buffer.sample_by_index_fields([0, 5], fields=["obs", "action"])

- **Entire buffer**:

  .. code-block:: python

      all_data = buffer.sample_all()

.. note::

   All sampled experiences are automatically moved to the training `device`.

Epoch-Based Training
^^^^^^^^^^^^^^^^^^^^^

The buffer supports epoch-based batch iteration using an internal iterator:

.. code-block:: python

    iterator = buffer.create_epoch_iterator(batch_size=64, n_epochs=5)

    for batch in iterator:
        train_step(batch)

You can also:

- Use :py:meth:`ReplayBuffer.get_next_batch` to fetch the next batch
- Use :py:meth:`ReplayBuffer.calculate_num_batches` to determine how many full batches you can draw
- Use :py:meth:`ReplayBuffer.get_steps_and_iterator` to prepare steps + iterator simultaneously:

  .. code-block:: python

      n_steps = buffer.get_steps_and_iterator(n_epochs=10, max_iter=0, batch_size=64)

Iteration-Based Training
^^^^^^^^^^^^^^^^^^^^^^^^^

When you set :code:`n_epochs=0`, the replay buffer will not initialize an internal epoch-based iterator. Instead, training proceeds using a fixed number of update iterations, and batches are sampled randomly at each step using :code:`sample_batch()` internally.

This mode is useful when:

- You do not want to cycle through the buffer in epochs
- You prefer fully stochastic sampling per training step
- Your training logic depends on a fixed number of updates (e.g., max_steps=1000)

.. code-block:: python

    n_steps = buffer.get_steps_and_iterator(
        n_epochs=0,         # Disables epoch-based iterator
        max_iter=1,         # Number of learning updates at each step
        batch_size=64
    )

.. note::

    When :code:`n_epochs=0`, :code:`get_next_batch()` uses random sampling instead of drawing from a predefined iterator.

Saving and Loading Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can persist buffer **metadata only** (not full contents) for checkpointing:

.. code-block:: python

    buffer.save("checkpoints/buffer")   # Saves to checkpoints/buffer.metadata
    buffer.load("checkpoints/buffer")   # Loads metadata only

This includes:

- `buffer_size`: Total buffer capacity
- `data_size`: Current number of stored transitions
- `pointer`: Insertion index for next sample

.. attention::

   The :code:`load()` method **does not restore experience data**. Only the metadata is recovered. To restore full contents, consider future serialization support or log replay externally.

Buffer Size and Status
^^^^^^^^^^^^^^^^^^^^^^^^

You can inspect buffer usage via:

- `len(buffer)` — number of stored transitions
- `buffer.size` — same as above, for convenience

.. code-block:: python

    print(len(buffer))        # e.g., 9500
    print(buffer.size)        # e.g., 9500
