# -----------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 ADIN Lab
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------------

from dataclasses import dataclass, field

from objectrl.models.dqn import DQNCritic
from objectrl.nets.critic_nets import DQNNet


@dataclass
class DQNCriticConfig:
    """Configuration for the DQN critic network.

    Attributes:
        arch (type): Neural network architecture class for the critic.
        critic_type (type): Critic class type.
        n_members (int): Number of critics in the ensemble.
        exploration_rate (float): Exploration rate for epsilon-greedy action selection.
    """

    arch: type = DQNNet
    critic_type: type = DQNCritic
    n_members: int = 1
    exploration_rate: float = 0.05


@dataclass
class DQNConfig:
    """Main DQN algorithm configuration class.

    Attributes:
        name (str): Algorithm identifier.
        loss (str): Loss function used for critic training.
        tau (float): Polyak averaging coefficient for target network updates.
        critic (DQNCriticConfig): Critic configuration.
    """

    name: str = "dqn"
    loss: str = "MSELoss"
    # Polyak averaging rate
    tau: float = 0.005

    critic: DQNCriticConfig = field(default=DQNCriticConfig)
