Tutorial 1: Step-by-Step AC Implementation
============================================

This tutorial walks through implementing a custom Actor-Critic (AC) algorithm named **MyAC** using ObjectRL.

We cover:

- Defining the MyAC agent, actor, and critic
- Customizing architecture via configuration
- Extending or replacing networks and extractors
- Registering and running the model
- Evaluating results from saved logs



1. Create the MyAC Agent
------------------------

The ``MyAC`` agent inherits from the base ``ActorCritic`` class in ``objectrl/models/basic/ac.py``. We define this custom agent and its specific actor and critic by creating a new file:
``objectrl/models/myac.py``
in which we define

- The agent: ``MyAC`` class (subclass of ``ActorCritic``)
- The actor: ``MyACActor`` (e.g., inheriting from ``Actor``)
- The critic: ``MyACCritic`` (e.g., inheriting from ``CriticEnsemble``)

.. code-block:: python

    from objectrl.models.basic.ac import ActorCritic
    from objectrl.models.basic.actor import Actor
    from objectrl.models.basic.critic import CriticEnsemble

    class MyACActor(Actor):
        pass

    class MyACCritic(CriticEnsemble):
        pass

    class MyAC(ActorCritic):
        _agent_name = "MyAC"

        def __init__(self, config, critic_type=MyACCritic, actor_type=MyACActor):
            super().__init__(config, critic_type, actor_type)


In this tutorial, we keep the behavior of these identical to their parents, but they could
overwrite or extend the behavior of the base actor-critic components as needed.

2. Define Model-Specific Configuration
--------------------------------------

We define their corresponding configuration dataclasses for each of the three classes in ``objectrl/config/model_configs/myac.py``.
In this example we'll define new architectures for actor and critic.


This file contains dataclasses that define:

- ``MyACActorConfig``
- ``MyACCriticConfig``
- ``MyACConfig`` (with references to the actor and critic config classes)

.. code-block:: python

    from dataclasses import dataclass, field

    from objectrl.models.myac import MyACActor, MyACCritic
    from objectrl.nets.actor_nets import MyActorNet
    from objectrl.nets.critic_nets import MyCriticNet

    @dataclass
    class MyACActorConfig:
        arch: type = MyActorNet

    @dataclass
    class MyACCriticConfig:
        arch: type = MyCriticNet

    @dataclass
    class MyACConfig:
        name: str = "myac"
        actor: MyACActorConfig = field(default_factory=MyACActorConfig)
        critic: MyACCriticConfig = field(default_factory=MyACCriticConfig)

These override and extend the default settings from ``objectrl/config/model.py`` and allow model-specific control over architecture and training behavior.

3. Define the Actor and Critic Networks
---------------------------------------

By default, ObjectRL uses the ``MLP`` class from ``objectrl/utils/net_utils.py`` as the feature extractor inside both actor and critic networks.

To use custom actor and critic architectures:

- Define ``MyActorNet`` in ``objectrl/nets/actor_nets.py``
- Define ``MyCriticNet`` in ``objectrl/nets/critic_nets.py``

.. code-block:: python
    :caption: objectrl/nets/actor_nets.py

    from objectrl.utils.net_utils import MLP
    import torch.nn as nn

    class MyActorNet(nn.Module):
        def __init__(self, dim_in, dim_out, depth, width, act, has_norm):
            super().__init__()
            self.net = MLP(dim_in, dim_out, depth, width, act, has_norm)

        def forward(self, x):
            return self.net(x)

.. code-block:: python
    :caption: objectrl/nets/critic_nets.py

    from objectrl.utils.net_utils import MLP
    import torch.nn as nn

    class MyCriticNet(nn.Module):
        def __init__(self, dim_in, dim_out, depth, width, act, has_norm):
            super().__init__()
            self.net = MLP(dim_in, dim_out, depth, width, act, has_norm)

        def forward(self, x):
            return self.net(x)

If you want to modify the **feature extractor**, write a new class (e.g., ``MyFeatureExtractor``) in:

``objectrl/utils/net_utils.py``

Then use this extractor inside your custom actor/critic networks. The names of the actor and critic networks are set in the model-specific config files mentioned above.

4. Register the Agent
----------------------

Once ``MyAC`` is implemented, you should register it in your model factory function by updating ``get_model``:

.. code-block:: python

    case "myac":
        return MyAC(config, critic.critic_type, actor.actor_type)

5. Run Training
---------------

You can train the model using ObjectRL’s command line interface:

.. code-block:: bash

    python objectrl/main.py --model.name "myac" --env.name "cheetah"

For most common settings (e.g., environment, training steps, seed), command-line arguments are an easy and effective way to configure your experiment.

For more structured or complex configurations, you can write a YAML configuration file:

.. code-block:: yaml
    :caption: myac.yaml

    model:
      name: myac
    env:
      name: cheetah
    training:
      max_steps: 100_000
    system:
      seed: 42

This allows you to override default values defined in the dataclass configuration while keeping experiments reproducible and clean.

6. Evaluate and Visualize
--------------------------

All logs and evaluation results are saved in the following directory structure:

::

    _logs/env_name/model_name/seed/timestamp/

This directory contains:

- ``log.log``: Logs of training progress
- ``learning-curve.png``: Training performance curve
- ``eval-curve.png``: Evaluation performance curve
- ``episode_rewards.npy, step_rewards.npy, eval_results.npy``: Numpy arrays with rewards and evaluation metrics

To inspect or visualize your experiment results, refer to these logs and images directly.