Experiments
===========

Overview
--------

The `experiments/` module defines the logic for running reinforcement learning experiments. It contains reusable base classes and specialized experiment implementations that manage:

- Environment creation and initialization
- Agent instantiation
- Training loops
- Evaluation procedures
- Logging, saving, and resetting mechanisms

This design allows for flexible experimentation across different environments, training setups, and agent architectures.

Module Structure
^^^^^^^^^^^^^^^^^

- **base_experiment.py**: Contains the abstract base class `Experiment` that sets up the environment and agent. It provides a general interface (`train`, `test`) but does not implement training or evaluation logic.

- **control_experiment.py**: Defines `ControlExperiment`, a concrete implementation of `Experiment`, that implements the full training and evaluation lifecycle. It is suitable for continuous control tasks in Gym-like environments.

Base Experiment
---------------

.. literalinclude:: ../../objectrl/experiments/base_experiment.py
    :language: python
    :start-after: start-class
    :end-before: end-class
    :caption: Base Experiment initialization.

.. raw:: html

    <p>The <strong>Experiment</strong> class serves as a blueprint for all experiment types. It initializes the environment and agent using the provided <code>MainConfig</code> object. This base class is extended by other experiment types that implement concrete logic for <code>train()</code> and <code>test()</code> methods.</p>

Control Experiment
------------------

Training
^^^^^^^^^

.. literalinclude:: ../../objectrl/experiments/control_experiment.py
    :language: python
    :start-after: class ControlExperiment
    :end-before: def eval
    :caption: Training loop implementation in ControlExperiment.

.. raw:: html

    <p><strong>ControlExperiment</strong> defines a standard training loop for training reinforcement learning agents. It supports warmup phases, periodic evaluations, and conditional resets based on training progress. It also handles interaction with the environment and manages learning updates and model saving.</p>

Evaluation
^^^^^^^^^^^

.. literalinclude:: ../../objectrl/experiments/control_experiment.py
    :language: python
    :start-after: def eval
    :end-before: self.agent.train()
    :caption: Evaluation procedure in ControlExperiment.

.. raw:: html

    <p>During evaluation, the agent runs multiple episodes in a separate evaluation environment. It uses its learned policy (without exploration noise) to gather performance statistics, which are logged and saved for later analysis. Evaluation runs do not affect the training process.</p>

Experiment Management
----------------------

Configuration and Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experiments are initialized using a `MainConfig` object, typically loaded from a YAML file or data classes. This configuration:

- Specifies environment and training parameters
- Defines the model architecture
- Controls system-level behavior like device usage
- Sets logging and evaluation frequencies

The experiment automatically uses this configuration to construct all required components, making the system fully reproducible and easily tunable.

Usage Example
^^^^^^^^^^^^^^

Here’s a minimal example of how to instantiate and run an experiment:

.. code-block:: python

    from objectrl.config.config import MainConfig
    from objectrl.experiments.control_experiment import ControlExperiment

    config = MainConfig.from_config(config_dict, model_name="my_model")
    experiment = ControlExperiment(config)
    experiment.train()

    # Optional: run evaluation after training
    experiment.eval(config.training.max_steps)

Extending Experiments
^^^^^^^^^^^^^^^^^^^^^^

To create a new type of experiment (e.g., curriculum learning, adversarial training), subclass the `Experiment` base class:

.. code-block:: python

    class MyCustomExperiment(Experiment):
        def train(self):
            # Custom training logic
            pass

        def test(self):
            # Optional test logic
            pass

.. attention::

    When extending `Experiment`, you must implement the ``train`` method. Optionally, you can override ``test`, ``eval``, or ``run`` for more customized workflows.

This design allows for experimentation with novel agent–environment interactions while reusing shared logic for environment and agent setup.

