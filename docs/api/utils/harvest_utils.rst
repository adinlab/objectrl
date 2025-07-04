Harvest Utilities
=================

This module provides the ``Harvester`` class to aggregate, analyze, and visualize evaluation
results across multiple reinforcement learning models and environments.

It computes metrics like Final Return, IQM (Interquartile Mean), and AULC (Area Under Learning Curve),
and generates publication-ready plots and CSV/Markdown tables.

Classes
-------

.. autoclass:: objectrl.utils.harvest_utils.Harvester
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

Key Features
------------

- Collect evaluation results from disk for multiple seeds.
- Compute metrics such as Final, IQM, AULC.
- Smooth learning curves using configurable moving average.
- Generate per-environment and global plots.
- Output results as ``.csv``, ``.md``, ``.png``, and ``.pdf``.

Typical Workflow
----------------

.. code-block:: python

    harvester = Harvester(config)
    harvester.harvest()

This will:

1. Load and process evaluation results.
2. Compute statistical summaries.
3. Save visualizations and tables in ``config.result_path``.
