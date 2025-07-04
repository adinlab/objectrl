Proximal Policy Optimization (PPO)
==================================

.. raw:: html

   <span class="doc-tag">on-policy</span>
   <span class="doc-tag">policy optimization</span>
    <span class="doc-tag">clipping</span>

**Paper**: `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_

Pseudocode
----------

.. pdf-include:: ../../_static/pseudocodes/ppo.pdf
    :width: 100%


Configuration
----------------

.. literalinclude:: ../../../objectrl/config/model_configs/ppo.py
    :language: python
    :start-after: [start-config]
    :end-before: [end-config]
    :caption: Specific configuration for the PPO algorithm (in config/model_configs/).

UML Diagram
----------------

.. figure:: ../../_static/imgs/ppo.png
    :width: 100%
    :align: center
    :alt: UML diagram for the PPO algorithm.

    UML diagram for the PPO algorithm.

.. raw:: html

   <p>We use the UML diagram to illustrate the relationships between the classes in our PPO implementation.</p>
   <p>The diagram shows how the <code>PPOActor</code> and <code>PPOCritic</code> classes inherit from the base classes <code>Actor</code> and <code>CriticEnsemble</code>, respectively. <code>ProximalPolicyOptimization</code> class also inherits from <code>ActorCritic</code> class which inherits from <code>Agent</code>.</p>
   <p>We illustrate each class's crucial attributes and methods for PPO. Specifically:</p>
   <p><code>PPOActorNetProbabilistic</code> class implements a probabilistic actor network with Gaussian policies for continuous actions.</p>
   <p><code>loss()</code> and <code>update()</code> methods in <code>PPOActor</code> class implement the PPO clipped surrogate objective and actor updates.</p>
   <p><code>update()</code> method in <code>PPOCritic</code> class updates the critic using Bellman targets computed externally.</p>
   <p><code>ProximalPolicyOptimization</code> class handles the overall training loop including generalized advantage estimation (GAE) and batched updates.</p>


Classes
-------

.. autoclass:: objectrl.models.ppo.PPOActorNetProbabilistic
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.models.ppo.PPOActor
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.models.ppo.PPOCritic
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl

.. autoclass:: objectrl.models.ppo.ProximalPolicyOptimization
    :undoc-members:
    :show-inheritance:
    :private-members:
    :members:
    :exclude-members: _abc_impl