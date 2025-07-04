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

import torch
import pytest
from types import SimpleNamespace
from gymnasium.spaces import Box
from pathlib import Path
from tensordict import TensorDict
from unittest.mock import patch

from objectrl.models.pbac import (
    PBACActor,
    PBACCritic,
    PACBayesLoss,
    PACBayesianAC,
)


class DummyPBACActorWrapper(torch.nn.Module):
    def __init__(self, dim_state, dim_act, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.dim_act = dim_act
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_state, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_heads * dim_act * 2),
        )

    def forward(self, x, is_training=True):
        out = self.model(x)
        return out.view(x.shape[0], self.n_heads, 2 * self.dim_act)


@pytest.fixture
def dummy_config_pbac():

    model = SimpleNamespace(
        name="pbac",
        gamma=0.99,
        tau=0.005,
        loss="PACBayesLoss",
        target_entropy=None,
        alpha=0.2,
        policy_delay=1,
        posterior_sampling_rate=2,
        sig2_lowerclamp=1e-4,
        lossparams=SimpleNamespace(
            prior_variance=1.0,
            bootstrap_rate=0.5,
            reduction="none",
            logvar_lower_clamp=-10.0,
            logvar_upper_clamp=2.0,
            complexity_coef=0.01,
        ),
        actor=SimpleNamespace(
            arch=lambda dim_state, dim_act, **kwargs: DummyPBACActorWrapper(
                dim_state, dim_act, n_heads=2
            ),
            has_target=False,
            reset=True,
            depth=2,
            width=64,
            activation="relu",
            norm=False,
            n_heads=2,
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
        ),
        storing_device=torch.device("cpu"),
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


def test_pbac_actor_act_behavior(dummy_config_pbac):
    actor = PBACActor(dummy_config_pbac, dim_state=4, dim_act=2)
    state = torch.randn(3, 4)
    dummy_action = torch.randn(3, 2)
    dummy_logprob = torch.randn(3)

    with patch(
        "objectrl.models.pbac.PBACActor.act",
        return_value={"action": dummy_action, "action_logprob": dummy_logprob},
    ):
        actor.is_episode_end = True
        actor.interaction_iter = 0
        out1 = actor.act(state, is_training=True)
        assert "action" in out1 and "action_logprob" in out1
        assert out1["action"].shape == (3, 2)

        actor.is_episode_end = False
        actor.interaction_iter = 1
        out2 = actor.act(state, is_training=True)
        assert out2["action"].shape == (3, 2)

        out3 = actor.act(state, is_training=False)
        assert out3["action"].shape == (3, 2)


def test_pbac_critic_get_bellman_target(dummy_config_pbac):
    critic = PBACCritic(dummy_config_pbac, dim_state=4, dim_act=2)
    actor = PBACActor(dummy_config_pbac, dim_state=4, dim_act=2)
    reward = torch.ones(3)
    next_state = torch.randn(3, 4)
    done = torch.zeros(3)

    dummy_action = torch.randn(3, 2)
    dummy_logprob = torch.randn(3)

    with patch.object(
        actor,
        "act",
        return_value={"action": dummy_action, "action_logprob": dummy_logprob},
    ):
        target = critic.get_bellman_target(reward, next_state, done, actor)

        assert target.shape == (
            2,
            3,
            3,
        ), f"Expected shape (2, 3, 3), got {target.shape}"
        assert torch.isfinite(target).all(), "Target contains non-finite values"


def test_pac_bayes_loss_forward(dummy_config_pbac):
    loss_fn = PACBayesLoss(dummy_config_pbac)
    q = torch.randn(3, 5)
    y = torch.randn(3, 5)

    loss = loss_fn(q, y)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_pbac_update_and_store_transition(dummy_config_pbac):
    agent = PACBayesianAC(dummy_config_pbac)
    state = torch.randn(4, 4)
    action = torch.randn(4, 2)
    target = torch.randn(4, 1)

    agent.critic.update(state, action, target)

    transition = TensorDict(
        {
            "terminated": torch.tensor(False),
            "truncated": torch.tensor(True),
        },
        batch_size=[],
    )
    agent.store_transition(transition)
    assert agent.actor.is_episode_end.item() is True

    transition2 = TensorDict(
        {
            "terminated": torch.tensor(False),
            "truncated": torch.tensor(False),
        },
        batch_size=[],
    )
    agent.store_transition(transition2)
    assert agent.actor.is_episode_end.item() is False
