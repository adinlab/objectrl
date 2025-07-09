Example 1: Adapting SAC to DRND
======================================

Modifying existing reinforcement learning algorithms to incorporate new exploration mechanisms or
architectural changes is a common requirement in research and development. Our class structure
makes such adaptations straightforward and maintainable. This example demonstrates how to convert
Soft Actor-Critic (SAC) [#sac]_ to Distributional Random Network Distillation (DRND) [#drnd]_ with
minimal code changes.

DRND extends SAC by adding an exploration bonus based on uncertainty estimation through
ensemble disagreement. The key insight is that areas of the state-action space where an ensemble of
networks disagrees most likely represent unexplored regions worth investigating.

The conversion from SAC to DRND requires three main components:
the *exploration bonus mechanism*,
*modified actor and critic networks*,
and an *updated training procedure*.
Each component builds upon the existing SAC implementation with minimal
modifications.

See :doc:`../api/models/drnd` for the full model API.
We will focus solely on the relevant changes compared to SAC in this use case.

DRND Bonus
~~~~~~~~~~

DRND's exploration bonus consists of a learnable predictor network, :math:`f_\theta`, and a
fixed ensemble of target networks, :math:`\bar f_1,\ldots, \bar f_N`. The ensemble provides
uncertainty estimates through disagreement among its members.

The ensemble's first two moments are computed as:

.. math::

    \mu(x) &= \mathbb E[X] = \frac1N \sum_{i=1}^N \bar f_i(x),\\
    B_2(x) &= \mathbb E[X^2] = \frac1N \sum_{i=1}^N (\bar f_i(x))^2.



The configuration dataclass encapsulates all hyperparameters needed for the bonus network:

.. literalinclude:: ../../objectrl/config/model_configs/drnd.py
    :language: python
    :start-after: [start-bonus-config]
    :end-before: [end-bonus-config]

The network initialization leverages our existing utilities for consistent architecture across predictor
and ensemble members:

.. code-block:: python

    from utils.net_utils import MLP
    from models.basic.ensemble import Ensemble

    gen_net = lambda: MLP(
        dim_obs[0] + dim_act[0],
        bonus_conf.dim_out,
        bonus_conf.depth,
        bonus_conf.width,
        act=bonus_conf.activation,
        has_norm=bonus_conf.norm
    )

    predictor = gen_net()
    target_ensemble = Ensemble(
        bonus_conf.n_members,
        gen_net(),
        [gen_net() for _ in range(bonus_conf.n_members)]
    )
    optim_pred = torch.optim.Adam(predictor.parameters(), lr=bonus_conf.learning_rate)


The predictor network is trained to match randomly selected ensemble members, creating a
learning signal that captures epistemic uncertainty. The training objective minimizes the
mean squared error:

.. math::
    L(\theta) = ||f_\theta(x) - c(x)||^2,

where :math:`x=(s,a)` represents the concatenated state-action input.


.. code-block:: python

    def update_predictor(state: torch.Tensor, action: torch.Tensor) -> None:
        sa = torch.cat((state, action), -1)
        optim_pred.zero_grad()
        c = torch.randint(bonus_conf.n_members, ())
        c_target = target_ensemble[c](sa)
        pred = predictor(sa)
        loss = (pred - c_target).pow(2).mean()
        loss.backward()
        optim_pred.step()


The exploration bonus combines two terms: disagreement between the predictor and the ensemble mean, and
normalized variance across ensemble predictions:

.. math::
    b(x) = \lambda ||f_\theta(x) - \mu(x)||^2 + (1 - \lambda)\sqrt{\frac{(f_\theta(x)^2 - \mu(x)^2)}{B_2(x) - \mu(x)^2}},

where :math:`\lambda` is a scaling factor:

.. code-block:: python

    @torch.no_grad()
    def bonus(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat((state, action), -1)
        target_pred = target_ensemble(sa)
        mu = target_pred.mean(0)
        mu2 = mu.pow(2)
        B2 = target_pred.pow(2).mean(0)
        pred = predictor(sa)
        dim_check(pred, mu)
        fst = (pred - mu).pow(2).sum(1, keepdim=True)
        snd = torch.sqrt(((pred.pow(2) - mu2).abs() / (B2 - mu2))).mean(1, keepdim=True)
        return bonus_conf.scale_factor * fst + (1 - bonus_conf.scale_factor) * snd



DRND Actor
~~~~~~~~~~

The DRND actor extends the SAC actor by incorporating the exploration bonus into its loss function.
This modification encourages the policy to explore regions where the ensemble exhibits high
disagreement.
The required changes are highlighted in the code block below.


.. literalinclude:: ../../objectrl/models/drnd.py
    :language: python
    :start-after: [start-actor-code]
    :end-before: [end-actor-code]
    :emphasize-lines: 35, 36, 52


DRND Critics
~~~~~~~~~~~~

The DRND critics modify the Bellman target computation to include the exploration bonus as an
intrinsic reward. This ensures that the value function accounts for the exploration benefit of
different state-action pairs:

.. literalinclude:: ../../objectrl/models/drnd.py
    :language: python
    :start-after: [start-critic-code]
    :end-before: [end-critic-code]
    :emphasize-lines: 49, 50


DRND ActorCritic
~~~~~~~~~~~~~~~~

The main training loop requires minimal modifications to accommodate the predictor network updates.
The key changes involve passing the bonus ensemble to actor and critic updates and adding predictor
training:

.. literalinclude:: ../../objectrl/models/drnd.py
    :language: python
    :start-after: [start-drnd-code]
    :end-before: [end-drnd-code]
    :emphasize-lines: 48-54, 66


Summary
~~~~~~~

Converting SAC to DRND demonstrates the flexibility of ObjectRL's architecture.
The transformation required only:

    1. Adding the exploration bonus mechanism with ensemble-based uncertainty estimation
    2. Extending actor and critic classes to incorporate bonus terms in their loss functions
    3. Minimal training loop modifications to update the predictor network

The inheritance-based design allows us to reuse the majority of SAC's implementation while
cleanly extending functionality. This pattern generalizes to other algorithmic modifications,
making it straightforward to experiment with novel exploration strategies, different network
architectures, or alternative learning objectives.

The approach ensures that each component remains testable and maintainable while facilitating rapid prototyping of new ideas. This design philosophy significantly reduces development time for implementing state-of-the-art reinforcement learning algorithms.


.. rubric:: References

.. [#sac] Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*
.. [#drnd] Yang, K., et al. (2024). *Distributional Random Network Distillation for Exploration*
