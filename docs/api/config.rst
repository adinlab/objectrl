Config
======

Overview
--------

The configuration system is organized into multiple Python files across the `config/` folder and its submodules. Each file serves a distinct role in defining the behavior and structure of your agent. Here's how the configuration is structured:

- **config.py** (in ``config/``):
  Contains the core configuration dataclasses that define the experiment setup. These include:
  
  - **NoiseConfig** – optional Gaussian noise added to observations or actions
  - **EnvConfig** – parameters specific to the RL environment
  - **TrainingConfig** – training-related hyperparameters
  - **SystemConfig** – runtime and system-level settings like device and seeds
  - **LoggingConfig** –  logging, checkpointing, and result-saving options
  - **MainConfig** – top-level config that combines all above configs and supports loading from external YAML or command-line overrides
  - **HarvestConfig** – evaluation, visualization, and aggregation of multiple runs

- **model.py** (in ``config/``):
  Defines model-specific components that control the neural network architectures and behavior of the agent. These include:
  
  - **ActorConfig** – configuration of the actor (policy) network  
  - **CriticConfig** – configuration of the critic (value function) network  
  - **ModelConfig** – wrapper for selecting and loading the appropriate model-specific configurations

- **utils.py** (in ``config/``):
  Provides advanced utilities for dynamic config composition and serialization. You normally do not need to modify this file unless developing core library extensions.


  - **Enhanced serialization**: ``enhanced_asdict`` and ``nested_asdict`` extend ``dataclasses.asdict`` to support dynamically added attributes and nested structures.
  - **Deep merging**: ``NestedDict`` enables recursive merging of nested configuration dictionaries using the ``|`` operator.
  - **Dynamic type conversion**: ``parse_value`` converts string inputs into appropriate Python types for CLI parsing.
  - **Configuration setup**: ``setup_config`` merges YAML, CLI, and Tyro-generated configs into a unified ``MainConfig`` object.
  - **CLI argument filtering**: ``filter_model_args`` separates model-specific arguments from general CLI input.
  - **Dynamic dataclass creation**: ``dict_to_dataclass`` builds dataclass types from nested dictionaries at runtime.
  - **Improved introspection**: ``enhanced_repr`` and ``create_field_dict`` support detailed string representations and introspection of dataclasses.
  - **Diff tracking**: ``diff_dict`` highlights configuration differences for debugging or logging.
  - **Tyro integration**: ``print_tyro_help`` formats help messages using Tyro's CLI interface.

- **model_configs/ folder**:  
  Holds model-specific overrides for **ActorConfig** and **CriticConfig**, making it easy to define distinct variants like TD3, SAC, etc.  
  These are referenced in the :code:`from_config()`` methods of the model classes, and their details are shared in model specific pages of the documentation.

  .. tip::

     For detailed documentation on each model-specific configuration (e.g., SAC, TD3), see the 
     :doc:`Models <models/models>` section of the documentation and select the model.

- **model_yamls/ folder**:  
  Contains YAML files that define the configurations for specific models. These files can be loaded dynamically to override default settings, allowing for flexible experimentation without modifying the codebase.

This structure allows for:

- Clear separation of concerns

- Easy composition of experiments

- Support for dynamic loading from YAML or command-line overrides

- Easy extensibility to add new models or new configuration sections

What follows is a detailed breakdown of each configuration class.

config.py File
---------------

Noise Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/config.py
    :language: python
    :start-after: [start-noise-config]
    :end-before: [end-noise-config]

.. raw:: html

    <p><strong>Noise</strong> configuration allows injecting Gaussian noise into either the agent's observations or actions. This is particularly useful for testing robustness, simulating real-world sensor or actuation errors, or inducing regularization effects during training.</p>


Environment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/config.py
    :language: python
    :start-after: [start-env-config]
    :end-before: [end-env-config]

.. raw:: html

    <p><strong>Environment</strong> configuration defines the environment-related settings used during training and evaluation. The <code>name</code> field specifies the environment (e.g., "cheetah", "hopper") while you can also use the original environment name (e.g., "HalfCheetah-v5", "Hopper-v5"), while optional fields like <code>noisy</code>, <code>position_delay</code>, and <code>control_cost_weight</code> allow fine-grained manipulation of the environment dynamics. This flexibility is useful for simulating real-world uncertainties or testing the agent's robustness under perturbed conditions.</p>

Training Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/config.py
    :language: python
    :start-after: [start-training-config]
    :end-before: [end-training-config]

.. raw:: html

    <p> <strong>Training</strong> configuration controls the learning process of the agent. It includes essential hyperparameters such as learning rate, batch size, and number of training steps. These parameters directly influence the stability, speed, and performance of the algorithm. With this configuration, you can easily tune experiments to meet different research goals or computational constraints. <code>parallelize_eval</code> initializes <code>eval_episodes</code> number of environments to evaluate in parallel instead of sequentially. If memory is not a bottleneck, this flag provides strong speed improvements as the number of evaluation episodes grows.</p>

System Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/config.py
    :language: python
    :start-after: [start-system-config]
    :end-before: [end-system-config]

.. raw:: html

    <p><strong>System</strong> configuration manages low-level runtime behavior and hardware settings. This includes control over the number of threads, random seed for reproducibility, and device selection (e.g., CPU or CUDA). These settings are especially useful for debugging, benchmarking, or deploying agents across heterogeneous hardware setups.</p>

Logging Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/config.py
    :language: python
    :start-after: [start-logging-config]
    :end-before: [end-logging-config]

.. raw:: html

    <p><strong>Logging</strong> configuration handles all aspects of experiment output and result storage. You can specify the path for saving logs, frequency of saving, and whether to persist the final model parameters. These options support systematic experiment tracking and simplify post-hoc analysis and reproducibility.</p>

Main Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/config.py
    :language: python
    :start-after: [start-main-config]
    :end-before: [end-main-config]

.. raw:: html

    <p><strong>Main</strong> configuration serves as the central entry point, composing sub-configurations such as <code>EnvConfig</code>, <code>TrainingConfig</code>, <code>SystemConfig</code>, <code>LoggingConfig</code>, and <code>ModelConfig</code>. It supports overriding values from external YAML files via <code>from_config</code>. This design ensures clarity, reproducibility, and ease of experiment management.</p>

Harvest Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/config.py
    :language: python
    :start-after: [start-harvest-config]
    :end-before: [end-harvest-config]

.. raw:: html

    <p><strong>Harvest</strong> configuration defines evaluation and result aggregation parameters. It supports running evaluations across multiple models and seeds and controls how results are saved and visualized.</p>

model.py File
----------------

Actor Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/model.py
    :language: python
    :start-after: [start-actor-config]
    :end-before: [end-actor-config]

.. raw:: html

    <p><strong>Actor</strong> configuration defines the architecture and behavior of the policy network. This level of control is essential for ablation studies and investigating architectural impacts on learning dynamics.</p>

Critic Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/model.py
    :language: python
    :start-after: [start-critic-config]
    :end-before: [end-critic-config]

.. raw:: html

    <p><strong>Critic</strong> configuration defines the structure of the value function estimator(s). This helps in adapting the value estimation to different algorithms (e.g., TD3 vs. SAC).</p>

Model Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../objectrl/config/model.py
    :language: python
    :start-after: [start-model-config]
    :end-before: [end-model-config]

.. raw:: html

    <p><strong>Model</strong> configuration acts as a lightweight entry class that dynamically delegates to the specific actor and critic configurations based on the selected model name. This abstraction allows for easy extension when introducing new algorithmic variants while preserving a consistent interface for configuration loading.</p>