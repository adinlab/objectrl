Tutorial 2: Build Your Own Model
=================================

The primary focus of this library centers on actor-critic mechanisms, which provide a robust foundation for
reinforcement learning algorithms. However, this design choice does not impose any fundamental limitations on
the framework's capabilities. The architecture allows for seamless implementation of value-based approaches,
including DQN-style methods. This tutorial demonstrates how to implement a Deep Q-Network (DQN) [#dqn]_ within
the existing framework, showcasing the library's flexibility and extensibility.

Background
~~~~~~~~~~

Deep Q-Networks combine deep learning with Q-learning, enabling the algorithm to handle high-dimensional
state spaces effectively. The core mathematical foundation relies on the Bellman equation for optimal
action-value functions:

.. math::

   Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{P}(s'|s,a)} \left[ r + \gamma \max_{a'} Q^*(s', a') \right].

In practice, DQN approximates this optimal Q-function using a neural network :math:`Q_\theta(s, a)` parameterized
by weights :math:`\theta`. The learning objective minimizes the temporal difference error:

.. math::

   L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( y - Q(s, a; \theta) \right)^2 \right],

where the target value is computed as:

.. math::

   y = r + \gamma \max_{a'} Q_{\bar \theta}(s', a').

Here, :math:`\bar\theta` represents the parameters of a target network that is updated periodically to stabilize
training. The use of the experience replay buffer :math:`\mathcal{D}` breaks the correlation between consecutive
samples, improving learning stability.


Building Blocks
~~~~~~~~~~~~~~~

To integrate DQN into the existing framework, we need to implement three key components following the library's 
design pattern. First, we implement the agent in ``models/dqn.py``, which will contain
the main DQN logic and coordination between components. Second, we need to create the corresponding
configuration file at ``config/model_configs/dqn.py`` to define hyperparameters and network architectures.
Finally, we must register the new model in the ``get_model`` factory function to make it accessible through
the standard interface.


The Agent
^^^^^^^^^


An agent requires implementation of three essential methods beyond
the default initialization: ``reset``, ``select_action``, and ``learn``. These methods define the
core behavioral interface that allows the agent to interact with environments and update its policy.

**Reset Method**

Resetting a DQN agent involves reinitializing the Q-value neural network, specifically the critic component.
The reset functionality is delegated to the critic's internal reset mechanism:

.. code-block:: python

    def reset(self) -> None:
        if self.critic._reset:
            self.critic.reset()

**Action Selection**

The action selection process in DQN is handled entirely by the Q-network, which we implement as part of the
critic component. This delegation pattern maintains a clean separation of concerns and allows the critic to
implement sophisticated exploration strategies such as epsilon-greedy action selection:

.. code-block:: python

    def select_action(self, state: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        return self.critic.act(state, is_training=is_training)


**Learning Process**

The learning mechanism performs gradient descent updates for a specified number of steps. Each iteration involves sampling a batch from the experience replay buffer, computing the Bellman target using the target network, and updating both the main Q-network and its target counterpart. The Bellman target computation follows the standard DQN formulation:

.. math::

   y_i = r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^-).

.. code-block:: python

    def learn(self, max_iter: int = 1, n_epochs: int = 0) -> None:
        """
        Perform the learning process for the agent.

        Args:
            max_iter (int): Maximum number of iterations for learning.
            n_epochs (int): Number of epochs for training. If 0, random sampling is used.
        """
        # Check if there is enough data in memory to sample a batch
        if self.config_train.batch_size > len(self.experience_memory):
            return None

        # Determine the number of steps and initialize the iterator
        n_steps = self.experience_memory.get_steps_and_iterator(
            n_epochs, max_iter, self.config_train.batch_size
        )

        for _ in range(n_steps):
            # Get batch using the internal iterator
            batch = self.experience_memory.get_next_batch(self.config_train.batch_size)

            bellman_target = self.critic.get_bellman_target(
                batch["reward"], batch["next_state"], batch["terminated"]
            )

            self.critic.update(batch["state"], batch["action"], bellman_target)

            # Update target networks
            if self.critic.has_target:
                self.critic.update_target()
            self.n_iter += 1

        return None





**Complete Agent Implementation**

The complete DQN agent class integrates all components and provides a clean interface for training and inference:

.. code-block:: python

    import torch

    from objectrl.agents.base_agent import Agent
    from objectrl.models.basic.critic import CriticEnsemble
    from objectrl.utils.utils import dim_check

    class DQN(Agent):
        def __init__(self, config: "MainConfig", critic_type: type[CriticEnsemble]) -> None:
            """
            Deep Q-Network
            """
            super().__init__(config)

            self.critic = critic_type(config, self.dim_state, self.dim_act)
            self.n_iter: int = 0

            # Requires discrete action spaces
            self._discrete_action_space = True

        def learn(self, max_iter: int = 1, n_epochs: int = 0) -> None:
            """
            Perform the learning process for the agent.

            Args:
                max_iter (int): Maximum number of iterations for learning.
                n_epochs (int): Number of epochs for training. If 0, random sampling is used.
            """
            # Check if there is enough data in memory to sample a batch
            if self.config_train.batch_size > len(self.experience_memory):
                return None

            # Determine the number of steps and initialize the iterator
            n_steps = self.experience_memory.get_steps_and_iterator(
                n_epochs, max_iter, self.config_train.batch_size
            )

            for _ in range(n_steps):
                # Get batch using the internal iterator
                batch = self.experience_memory.get_next_batch(self.config_train.batch_size)

                bellman_target = self.critic.get_bellman_target(
                    batch["reward"], batch["next_state"], batch["terminated"]
                )

                self.critic.update(batch["state"], batch["action"], bellman_target)

                # Update target networks
                if self.critic.has_target:
                    self.critic.update_target()
                self.n_iter += 1

            return None

        def select_action(
            self, state: torch.Tensor, is_training: bool = True
        ) -> torch.Tensor:
            return self.critic.act(state, is_training=is_training)

        def reset(self) -> None:
            if self.critic._reset:
                self.critic.reset()

The Critic
^^^^^^^^^^

While the original DQN algorithm employs a single critic network, our implementation extends the
``CriticEnsemble`` base class to facilitate future enhancements such as *Double DQN (DDQN)*. [#ddqn]_
This design choice provides flexibility for implementing ensemble methods and advanced variants without
requiring significant architectural changes.

**Adapting to State-Only Inputs**

The general ``CriticEnsemble`` framework assumes state-action pairs as inputs, but DQN operates on states alone,
outputting Q-values for all possible actions. We adapt the interface by modifying the Q-function methods to
accept only state inputs:

.. code-block:: python

    def Q(self, state: torch.Tensor, action: None) -> torch.Tensor:
        # Indexing as there is only a single critic
        return self.model_ensemble(state)[0]

    def Q_t(self, state: torch.Tensor, action: None) -> torch.Tensor:
        # Indexing as there is only a single critic
        return self.model_ensemble(state)[0]

**Bellman Target Computation**

The Bellman target computation implements the core DQN update rule, utilizing the target network to compute stable target values. The method handles batched operations efficiently and ensures proper tensor dimensions:

.. code-block:: python

    @torch.no_grad()
    def get_bellman_target(
        self, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:

        target_value, _ = self.Q_t(next_state, None).max(-1, keepdim=True)

        y = reward.unsqueeze(-1) + self._gamma * target_value * (1 - done.unsqueeze(-1))

        return y

**Network Updates**

The update mechanism performs standard gradient descent on the temporal difference error. It computes predictions for the taken actions and minimizes the mean squared error against the Bellman targets:

.. code-block:: python

    def update(
        self, state: torch.Tensor, action: torch.Tensor, y: torch.Tensor
    ) -> None:
        """
        Update critic networks using the provided Bellman targets.

        Args:
            state: State tensor.
            action: Action tensor.
            y: Bellman target values.
        """
        self.optim.zero_grad()

        pred = self.Q(state, None)[range(state.shape[0]), action.int()][:, None]
        dim_check(pred, y)
        loss = self.loss(pred, y).mean()
        loss.backward()
        self.optim.step()
        self.iter += 1




**Epsilon-Greedy Action Selection**

The action selection method implements epsilon-greedy exploration, balancing exploitation of learned
Q-values with exploration of random actions. During training, the agent selects random actions with probability
``_explore_rate``, otherwise, it chooses the action with the maximum Q-value:

.. code-block:: python

    def act(
        self, state: torch.Tensor, target: bool = False, is_training: bool = True
    ) -> torch.Tensor:

        if is_training and torch.rand(1) < self._explore_rate:
            return torch.randint(self.dim_act, size=(1,), device=state.device)

        if target:
            return self.Q_t(state, None).argmax(dim=-1, keepdim=True)
        else:
            return self.Q(state, None).argmax(dim=-1, keepdim=True)


**Network Architecture**

The critic network used by the DQN critic is a straightforward adaptation of the standard ``CriticNet``.
It implements a fully connected neural network that maps from state observations to Q-values for all possible actions:

.. code-block:: python

    class DQNNet(nn.Module):
        """
        Deterministic Critic Network (Q-network).
        Args:
            dim_state (int): Dimension of observation space.
            dim_act (int): Dimension of action space.
            depth (int): Number of hidden layers.
            width (int): Width of each hidden layer.
            act (str): Activation function to use.
            has_norm (bool): Whether to include normalization layers.
        """

        def __init__(
            self,
            dim_state: int,
            dim_act: int,
            depth: int = 3,
            width: int = 256,
            act: str = "relu",
            has_norm: bool = False,
        ) -> None:
            super().__init__()

            self.arch = MLP(dim_state, dim_act, depth, width, act=act, has_norm=has_norm)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the critic network.

            Args:
                x (Tensor): Concatenated observation and action tensor.
            """
            return self.arch(x)


**Complete Critic Implementation**

The complete DQN critic class integrates all the above components into a cohesive implementation that handles the core
value function learning:

.. code-block:: python

    class DQNCritic(CriticEnsemble):
        def __init__(
            self, config: "MainConfig", dim_state: tuple[int, ...], dim_act: tuple[int, ...]
        ):
            super().__init__(config, dim_state, dim_act)

            self.dim_act = dim_act
            self._explore_rate: float = config.model.critic.exploration_rate

        def update(
            self, state: torch.Tensor, action: torch.Tensor, y: torch.Tensor
        ) -> None:
            """
            Update critic networks using the provided Bellman targets.

            Args:
                state: State tensor.
                action: Action tensor.
                y: Bellman target values.
            """
            self.optim.zero_grad()

            pred = self.Q(state, None)[range(state.shape[0]), action.int()][:, None]
            dim_check(pred, y)
            loss = self.loss(pred, y).mean()
            loss.backward()
            self.optim.step()
            self.iter += 1

        def act(
            self, state: torch.Tensor, target: bool = False, is_training: bool = True
        ) -> torch.Tensor:

            if is_training and torch.rand(1) < self._explore_rate:
                return torch.randint(self.dim_act, size=(1,), device=state.device)

            if target:
                return self.Q_t(state, None).argmax(dim=-1, keepdim=True)
            else:
                return self.Q(state, None).argmax(dim=-1, keepdim=True)

        def Q(self, state: torch.Tensor, action: None) -> torch.Tensor:
            # Indexing as there is only a single critic
            return self.model_ensemble(state)[0]

        def Q_t(self, state: torch.Tensor, action: None) -> torch.Tensor:
            # Indexing as there is only a single critic
            return self.model_ensemble(state)[0]

        @torch.no_grad()
        def get_bellman_target(
            self, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor
        ) -> torch.Tensor:

            target_value, _ = self.Q_t(next_state, None).max(-1, keepdim=True)

            dim_check(reward.unsqueeze(-1), target_value)
            y = reward.unsqueeze(-1) + self._gamma * target_value * (1 - done.unsqueeze(-1))

            return y

The Config
^^^^^^^^^^

To integrate DQN into the configuration system, we need to add the appropriate configuration classes
to ``config/model_configs/dqn.py``. This configuration defines the network architecture, hyperparameters, and
component types used by the DQN implementation.

The configuration follows the library's standard pattern, defining both critic-specific and model-level parameters.
The critic configuration specifies the network architecture, exploration rate, and other critic-specific
hyperparameters, while the model configuration defines global settings such as the loss function and target
network update rate:

.. code-block:: python

    from dataclasses import dataclass, field

    from objectrl.models.dqn import DQNCritic
    from objectrl.nets.critic_nets import DQNNet

    @dataclass
    class DQNCriticConfig:
        arch: type = DQNNet
        critic_type: type = DQNCritic
        n_members: int = 1
        exploration_rate: float = 0.05

    @dataclass
    class DQNConfig:
        name: str = "dqn"
        loss: str = "MSELoss"
        # Polyak averaging rate
        tau: float = 0.005

        critic: DQNCriticConfig = field(default=DQNCriticConfig)

**Model Registration**

After implementing the agent, critic, and configuration components, the final step involves registering the DQN
model in the ``get_model`` factory function. This registration makes the DQN implementation accessible through the
standard model selection interface:

.. code-block:: python

        case "dqn":
            return DQN(config, critic.critic_type)

**Usage Example**

Once properly integrated, you can use the DQN implementation exactly like other models in the framework.
The following trains a DQN model on the classical `cartpole` environment.

.. code-block:: bash

    python main.py --model.name "dqn" --env.name "cartpole" --training.max-steps 50_000 --progress

This seamless integration showcases the framework's extensibility, allowing for easy addition of new
algorithms while maintaining consistency with the existing interface.

.. rubric:: References

.. [#dqn] Mnih, V. et al. (2013). *Playing Atari with Deep Reinforcement Learning*
.. [#ddqn] van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-Learning*


