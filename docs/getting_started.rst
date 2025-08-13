Getting Started
===============

Installation Guide
------------------

ObjectRL is tested with Python 3.12. We recommend using a virtual environment to keep dependencies isolated and manageable.

1. **Create and activate a virtual environment using conda:**

.. code-block:: bash

   conda create -n objectrl python=3.12 -y
   conda activate objectrl

Alternatively, you can use `venv` or other virtual environment tools.

2. **Install ObjectRL**

You have two options for installing ObjectRL: from `PyPI <https://pypi.org/project/objectrl/>`_ or directly from the `GitHub repository <https://github.com/adinlab/objectrl>`_ for the latest development version.

.. tabs::
   .. group-tab:: PyPI
      Install the latest stable release:

      .. code-block:: bash
      
         pip install objectrl

   .. group-tab:: GitHub
      Install the latest development version from source:

      .. code-block:: bash
      
         git clone https://github.com/adinlab/objectrl.git
         cd objectrl
         pip install -e .

3. **Install optional dependencies**

For additional features such as documentation generation, install optional dependencies:

.. tabs::
   .. group-tab:: PyPI
      .. code-block:: bash
      
         pip install objectrl['docs']

   .. group-tab:: GitHub
      .. code-block:: bash
      
         pip install -e .[docs]

Running Your First Experiment
-----------------------------

You can launch your first RL experiment directly from the command line.

1. **Run SAC on the default** ``cheetah`` **environment:**

   If installed from PyPI, use:

   .. code-block:: bash

      python -m objectrl.main --model.name sac
   
   If installed from GitHub, use:

   .. code-block:: bash

      python objectrl/main.py --model.name sac
   
   Other examples will assume the GitHub installation.

2. **Run DDPG on the** ``hopper`` **environment:**

   .. code-block:: bash

      python objectrl/main.py --model.name ddpg --env.name hopper

3. **Customize training parameters (e.g., train SAC on hopper for 100,000 steps and evaluate every 5 episodes):**

   .. code-block:: bash

      python objectrl/main.py\
         --model.name sac\
         --env.name hopper\
         --training.max_steps 100_000\
         --training.eval_episodes 5


Configuration Options
---------------------

For simple changes, CLI arguments are usually sufficient. However, for more complex configurations or reproducibility, you can create custom YAML file, e.g.,

.. code-block:: yaml
   :caption: ppo.yaml

   model:
      name: ppo
   training:
      warmup_steps: 0
      learn_frequency: 2048
      batch_size: 64
      n_epochs: 10

and use it at runtime via

.. code-block:: bash

   python objectrl/main.py --config ppo.yaml

This system enables clean separation of experiment configurations and code.

Command line arguments are able to overwrite these parameters which allows for a quick iteration of parameter tuning.
Assume you want to keep your ``ppo.yaml`` file, but test whether a different batch size improves performance.

.. code-block:: bash

   python objectrl/main.py --config ppo.yaml --training.batch_size 32


To get an overview of all non-model specific parameters run

.. code-block:: bash

   python objectrl/main.py --help

Model specific parameters, e.g., SAC's, are available via

.. code-block:: bash

   python objectrl/main.py --help_model sac

See :doc:`API</api/api>` for further details on the configuration.



Next Steps
----------

Once you have your environment set up and can run a simple experiment, explore the following documentation sections for deeper understanding:

- **Examples**: Hands-on tutorials and case studies for customizing and extending ObjectRL.
- **API**: Detailed reference documentation for classes, modules, and configuration options.

Troubleshooting
---------------

- If you encounter errors related to environment dependencies (e.g., MuJoCo), verify your installation and environment variables.
- Use ``pip list`` or ``conda list`` to check installed packages.
- Run experiments with ``--help`` to see available CLI options.
- Open an issue on our `GitHub repository. <https://github.com/adinlab/objectrl>`_ 