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

import torch
import pytest
from types import SimpleNamespace
from pathlib import Path
from gymnasium.spaces import Box
from objectrl.models.ppo import (
    PPOActorNetProbabilistic,
    PPOActor,
    PPOCritic,
    ProximalPolicyOptimization,
)
from objectrl.nets.critic_nets import ValueNet


@pytest.fixture
def dummy_config():

    model = SimpleNamespace(
        name="ppo",
        loss="MSELoss",
        tau=0.0,
        policy_delay=1,
        max_grad_norm=0.5,
        clip_rate=0.2,
        GAE_lambda=0.95,
        normalize_advantages=True,
        entropy_coef=0.01,
        actor=SimpleNamespace(
            arch=PPOActorNetProbabilistic,
            actor_type=PPOActor,
            reset=True,
            has_target=False,
            depth=2,
            width=64,
            max_grad_norm=0.5,
            activation="relu",
            norm=False,
            n_heads=1,
        ),
        critic=SimpleNamespace(
            arch=ValueNet,
            critic_type=PPOCritic,
            reset=True,
            has_target=False,
            depth=2,
            n_members=1,
            width=64,
            max_grad_norm=0.5,
            activation="relu",
            norm=False,
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
        warmup_steps=0,
        gamma=0.99,
        optimizer="Adam",
        learning_rate=1e-3,
        batch_size=4,
        learn_frequency=1,
    )

    system = SimpleNamespace(
        device=torch.device("cpu"),
        storing_device=torch.device("cpu"),
        seed=42,
    )

    logging = SimpleNamespace(
        result_path=Path("./_logs"),
    )

    return SimpleNamespace(
        model=model,
        env=env,
        training=training,
        system=system,
        logging=logging,
        verbose=True,
    )


def test_ppo_actor_net_probabilistic_forward(dummy_config):
    net = PPOActorNetProbabilistic(dim_state=4, dim_act=2)
    dummy_input = torch.randn(3, 4)

    output_train = net(dummy_input, is_training=True)
    assert "dist" in output_train
    assert "action" in output_train
    assert "action_logprob" in output_train
    assert output_train["action"].shape == (3, 2)
    assert output_train["action_logprob"].shape == (3,)

    output_eval = net(dummy_input, is_training=False)
    assert torch.allclose(output_eval["action"], output_eval["dist"].mean)


def test_ppo_actor_loss_and_update(dummy_config):
    actor = PPOActor(dummy_config, dim_state=4, dim_act=2)
    actor.optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

    batch_size = 5
    state = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    old_logprob = torch.randn(batch_size)
    advantages = torch.randn(batch_size)

    loss = actor.loss(state, actions, old_logprob, advantages)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    actor.update(state, actions, old_logprob, advantages)


def test_ppo_critic_update(dummy_config):
    critic = PPOCritic(dummy_config, dim_state=4, dim_act=2)
    critic.optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    batch_size = 5
    state = torch.randn(batch_size, 4)
    target = torch.randn(batch_size, 1)

    critic.update(state, target)


def test_proximal_policy_optimization_init_and_learn(monkeypatch, dummy_config):
    class DummyMemory:
        def __init__(self):
            self.data = []
            self.storing_device = torch.device("cpu")

        def sample_all(self):
            batch_size = 4
            return {
                "reward": torch.zeros(batch_size, 1),
                "terminated": torch.zeros(batch_size, 1),
                "value": torch.zeros(batch_size, 1),
                "next_state_value": torch.zeros(batch_size, 1),
                "state": torch.randn(batch_size, 4),
                "next_state": torch.randn(batch_size, 4),
                "action_logprob": torch.zeros(batch_size),
                "action": torch.randn(batch_size, 2),
                "advantages": torch.zeros(batch_size),
                "returns": torch.zeros(batch_size),
            }

        def reset(self):
            pass

        def add_batch(self, batch):
            pass

        def get_steps_and_iterator(self, n_epochs, max_iter, batch_size):
            return 1

        def get_next_batch(self, batch_size):
            batch = self.sample_all()
            return batch

        def __len__(self):
            return 10

    agent = ProximalPolicyOptimization(dummy_config)
    agent.experience_memory = DummyMemory()

    actor_update_called = False
    critic_update_called = False

    def dummy_actor_update(*args, **kwargs):
        nonlocal actor_update_called
        actor_update_called = True

    def dummy_critic_update(*args, **kwargs):
        nonlocal critic_update_called
        critic_update_called = True

    agent.actor.update = dummy_actor_update
    agent.critic.update = dummy_critic_update

    agent.learn(max_iter=1, n_epochs=1)
    assert actor_update_called
    assert critic_update_called

    dummy_transition = {
        "state": torch.randn(1, 4),
        "action": torch.randn(1, 2),
        "reward": 0.0,
        "next_state": torch.randn(1, 4),
        "terminated": 0,
        "truncated": 0,
        "step": 0,
    }
    dummy_action_logprob = torch.tensor([0.0])

    agent.critic.Q = lambda x: torch.zeros(x.shape[0], 1)
    transition = agent.generate_transition(
        action_logprob=dummy_action_logprob, **dummy_transition
    )
    assert "next_state_value" in transition
    assert "value" in transition
    assert "action_logprob" in transition
