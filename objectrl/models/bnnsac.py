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

import typing

from objectrl.models.sac import SACActor, SACCritic, SoftActorCritic

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class BNNSoftActorCritic(SoftActorCritic):
    """
    BNN-Soft Actor-Critic agent with BNN-Critic ensemble
    """

    _agent_name = "BNN-SAC"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = SACCritic,
        actor_type: type = SACActor,
    ) -> None:
        """
        Initializes SAC agent.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type): Critic class type.
            actor_type (type): Actor class type.
        Returns:
            None
        """
        super().__init__(config, critic_type, actor_type)
