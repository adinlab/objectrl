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

import pytest
import torch
from types import SimpleNamespace
from pathlib import Path
from gymnasium.spaces import Box

from objectrl.models.oac import OACActor, OACCritic, OptimisticActorCritic


class DummyOACActorWrapper(torch.nn.Module):
    def __init__(self, dim_state, dim_act):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_state, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, dim_act * 2),
        )

    def forward(self, x, is_training=False):
        out = self.net(x)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        std = torch.exp(log_std.clamp(-5, 2))
        dist = torch.distributions.Normal(mu, std)
        transformed_dist = torch.distributions.TransformedDistribution(
            dist, torch.distributions.transforms.TanhTransform(cache_size=1)
        )
        action = transformed_dist.rsample() if is_training else torch.tanh(mu)
        return {
            "action": action,
            "dist": transformed_dist,
            "action_logprob": transformed_dist.log_prob(action).sum(
                dim=-1, keepdim=True
            ),
        }


@pytest.fixture
def dummy_config_oac():
    model = SimpleNamespace(
        tau=0.005,
        name="oac",
        policy_delay=1,
        loss="MSELoss",
        target_entropy=None,
        alpha=0.2,
        exploration=SimpleNamespace(beta_ub=1.0, delta=0.1),
        noise=SimpleNamespace(sigma_target=0.2, noise_clamp=0.3),
        actor=SimpleNamespace(
            arch=lambda dim_state, dim_act, **kwargs: DummyOACActorWrapper(
                dim_state, dim_act
            ),
            has_target=False,
            reset=True,
            depth=2,
            width=64,
            activation="relu",
            norm=False,
            lambda_actor=0.1,
            n_heads=1,
        ),
        critic=SimpleNamespace(
            target_reduce="min",
            reduce="min",
            n_members=2,
            has_target=True,
            reset=True,
            arch=lambda dim_state, dim_act, **kwargs: torch.nn.Sequential(
                torch.nn.Linear(dim_state + dim_act, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            ),
            depth=2,
            width=64,
            activation="relu",
            norm=False,
            lambda_critic=0.5,
        ),
    )

    env = SimpleNamespace(
        env=SimpleNamespace(
            action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=float),
            observation_space=Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
        ),
        name="DummyEnv",
    )

    training = SimpleNamespace(
        buffer_size=10000,
        gamma=0.99,
        optimizer="Adam",
        learning_rate=1e-3,
        batch_size=4,
    )

    system = SimpleNamespace(
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        seed=42,
    )

    logging = SimpleNamespace(result_path=Path("./_logs"))

    return SimpleNamespace(
        model=model,
        env=env,
        training=training,
        system=system,
        logging=logging,
        verbose=True,
    )


def test_oac_actor_loss(dummy_config_oac):
    actor = OACActor(dummy_config_oac, 4, 2)
    critic = OACCritic(dummy_config_oac, 4, 2)
    state = torch.randn(6, 4)
    loss = actor.loss(state, critic)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_oac_critic_bellman_target(dummy_config_oac):
    critic = OACCritic(dummy_config_oac, 4, 2)
    actor = OACActor(dummy_config_oac, 4, 2)
    reward = torch.ones(6)
    next_state = torch.randn(6, 4)
    done = torch.zeros(6)

    target = critic.get_bellman_target(reward, next_state, done, actor)

    assert isinstance(target, torch.Tensor)
    assert target.shape == (6, 1)


def test_oac_select_action(dummy_config_oac):
    agent = OptimisticActorCritic(dummy_config_oac)
    state = torch.randn(5, 4)

    act_dict_train = agent.select_action(state, is_training=True)
    act_dict_eval = agent.select_action(state, is_training=False)

    assert "action" in act_dict_train and "action" in act_dict_eval
    assert act_dict_train["action"].shape == (5, 2)
    assert act_dict_eval["action"].shape == (5, 2)
