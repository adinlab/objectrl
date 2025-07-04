Example 2: Adapting SAC to REDQ
======================================

This example demonstrates how to convert Soft Actor-Critic (SAC) [#sac]_ to Randomized Ensembled
Double Q-learning (REDQ) [#redq]_ with minimal code changes.

REDQ extends SAC to improve the sample efficiency of model-free methods in continuous control tasks, 
where environment interactions are costly. The core idea is to increase the update-to-data (UTD) 
ratio, i.e., perform multiple gradient updates per environment step (:math:`\text{UTD} \gg 1`).
However, naively increasing UTD in SAC leads to instability. REDQ overcomes this by introducing
three key components: a large critic ensemble, a randomized pessimistic critic target, and an
averaged critic ensemble for policy learning.

The conversion from SAC to REDQ requires four modifications:

(i) *Increase the UTD ratio* (:math:`\text{UTD} \gg 1`),
(ii) *Maintain an ensemble of* :math:`N` *critics*,
(iii) *Compute the critic target using the minimum over a random subset of size* :math:`M < N` *from the critic ensemble*,
(iv) *Use the mean of the critic ensemble for actor updates*.

Each component builds upon the existing SAC implementation with minimal modifications.

See :doc:`../api/models/redq` for the full model API.
We will focus solely on the relevant changes compared to SAC in this use case.

REDQ Critics
~~~~~~~~~~~~

REDQ contains :math:`N` critics. To define the Bellman target for updating the critics,
REDQ samples a set :math:`\mathcal{M}` of :math:`M` distinct indices from :math:`\{1, 2, \ldots, N\}`.  
Then, compute the Q target :math:`y` (shared across all :math:`N` Q-functions) using the 
*minimum* over these :math:`M` critics:

.. math::

    y = r + \gamma \left( \min_{i \in \mathcal{M}} Q_{\phi_{\text{targ}, i}} \left( s', \tilde{a}' \right) 
        - \alpha \log \pi_{\theta} \left( \tilde{a}' \mid s' \right) \right), 
    \quad \tilde{a}' \sim \pi_{\theta} ( \cdot \mid s')

The policy :math:`\theta`, on the other hand, is updated using the *mean* of 
all :math:`N` critics in the ensemble, with gradient ascent:

.. math::

    \nabla_{\theta} \frac{1}{|B|} \sum_{s \in B} 
    \left( \frac{1}{N} \sum_{i=1}^{N} 
    Q_{\phi_i} \left( s, \tilde{a}_{\theta}(s) \right) 
    - \alpha \log \pi_{\theta} \left( \tilde{a}_{\theta}(s) \mid s \right) 
    \right), 
    \quad \tilde{a}_{\theta}(s) \sim \pi_{\theta}(\cdot \mid s)

This means we need to inherit a ``REDQCritic`` class from ``SACCritic`` with a modified ``reduce`` function:

.. literalinclude:: ../../objectrl/models/redq.py
    :language: python
    :start-after: [start-reduce-code]
    :end-before: [end-reduce-code]

REDQ ActorCritic and REDQ Actor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The REDQ actor is directly inherited from SAC without further modification:

.. literalinclude:: ../../objectrl/models/redq.py
    :language: python
    :start-after: [start-redq-code]
    :end-before: [end-redq-code]
    :emphasize-lines: 16

REDQ Config
~~~~~~~~~~~

Following the original REDQ implementation, we set the UTD ratio (corresponding to the hyperparameter ``policy_delay``) to 20.
REDQ maintains :math:`N = 10` critics, and to compute the Bellman target, it takes :math:`M = 2` random target critics.
As described above, REDQ uses the *minimum* of :math:`M = 2` critics for critic learning,
and the *mean* of all critics for actor training.
The relevant hyperparameters in the configuration dataclass can be modified as follows:

.. literalinclude:: ../../objectrl/config/model_configs/redq.py
    :language: python
    :start-after: [start-critic-config]
    :end-before: [end-critic-config]
    :emphasize-lines: 17-18

.. rubric:: References

.. [#sac] Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*
.. [#redq] Chen, X., et al. (2021). *Randomized Ensembled Double Q-Learning: Learning Fast Without a Model.*
