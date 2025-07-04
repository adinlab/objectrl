Loggers
=======

Overview
--------

The ``loggers/`` module contains the ``Logger`` class, which is responsible for:

- Recording textual logs to disk
- Saving intermediate experiment results (e.g., rewards, steps)
- Plotting reward curves and evaluation metrics
- Computing statistical summaries (mean, IQM)
- Structuring output directories per run

This ensures that each experiment is reproducible and its outcomes are easy to inspect and analyze visually or numerically.

Logger
------------

.. autoclass:: objectrl.loggers.logger.Logger
    :members:
    :undoc-members:
    :show-inheritance:

Initialization
^^^^^^^^^^^^^^

The ``Logger`` is initialized using explicit arguments: ``result_path``, ``env_name``, ``model_name``, and ``seed``. It constructs a unique timestamped folder using this metadata and stores:

- Logs (``log.log``)
- Evaluation results (``eval_results.npy``)
- Visualizations (``learning-curve.png``, ``eval-curve.png``)
- Rewards and metrics (``episode_rewards.npy``, ``step_rewards.npy``)

.. code-block:: python

    from objectrl.loggers.logger import Logger
    from pathlib import Path

    logger = Logger(
        result_path="../_logs",
        env_name="cheetah",
        model_name="sac",
        seed=0,
        config=config  # Optional: logs config details if provided
    )
.. note::

    The ``config`` argument is optional. If provided, its content is logged for reproducibility under ``log.log``.

Logging Messages
^^^^^^^^^^^^^^^^^^

You can log messages via:

- ``logger.log(message)`` for standard info logs
- ``logger.critical(message)`` for critical highlights (e.g., evaluation results)
- ``logger(message)`` — uses ``__call__`` as shorthand for ``logger.log(...)``

Saving and Plotting Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During training, the logger tracks episode-level and step-level rewards. These are saved and plotted automatically at intervals.

.. code-block:: python

    logger.save(info, episode, step)
    logger.plot_rewards(rewards, steps)

The saved files include:

- ``episode_rewards.npy``: episode-level rewards
- ``step_rewards.npy``: raw reward values per step
- ``learning-curve.png``: training reward curve

Evaluation Logging
^^^^^^^^^^^^^^^^^^

During evaluation, rewards across multiple episodes are saved and summarized using:

- **Mean reward**
- **Interquartile Mean (IQM)** — a robust average that ignores extreme outliers

.. code-block:: python

    logger.save_eval_results(current_step, reward_tensor)

This also plots an ``eval-curve.png`` showing mean ± standard deviation over training steps.

Reward Statistics
^^^^^^^^^^^^^^^^^^^^

The ``IQM_reward_calculator`` statically computes the Interquartile Mean (middle 50%) of a set of rewards. This is often preferred in RL benchmarking to reduce the effect of outliers.

.. code-block:: python

    iqm = Logger.IQM_reward_calculator(rewards)

Directory Structure
^^^^^^^^^^^^^^^^^^^^^^

Each experiment creates an output folder under:

.. code-block::

    {result_path}/{env.name}/{model.name}/seed_{seed}/{timestamp}/

Example:

.. code-block::

    _logs/cheetah/sac/seed_01/2025-05-26_14-30-15/

This folder includes:

- Logs: ``log.log``
- Training plot: ``learning-curve.png``
- Evaluation plot: ``eval-curve.png``
- NumPy metrics: ``episode_rewards.npy``, ``step_rewards.npy``, ``eval_results.npy``


