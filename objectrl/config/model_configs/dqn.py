# -----------------------------------------------------------------------------------
# ObjectRL: An Object-Oriented Reinforcement Learning Codebase
# Copyright (C) 2025 ADIN Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

    critic: DQNCriticConfig = field(default_factory=DQNCriticConfig)
