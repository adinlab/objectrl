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

import copy
import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F
import functools

from objectrl.models.basic.ensemble import Ensemble

# Mock for torch.nn.functional_call and torch.func.vmap if you don't have torch.func
try:
    import torch.func as thf
except ImportError:
    # Minimal mock versions to test logic - adjust as needed for your environment
    class thf:
        @staticmethod
        def stack_module_state(models):
            # Stack parameters and buffers into dicts of tensors
            params = {}
            buffers = {}
            for name, param in models[0].named_parameters():
                stacked = torch.stack([m.state_dict()[name] for m in models])
                params[name] = stacked
            for name, buffer in models[0].named_buffers():
                stacked = torch.stack([m.state_dict()[name] for m in models])
                buffers[name] = stacked
            return params, buffers

        @staticmethod
        def functional_call(base_model, state, inputs):
            params, buffers = state
            # Load params and buffers into base_model manually
            state_dict = {}
            for k, v in params.items():
                state_dict[k] = v
            for k, v in buffers.items():
                state_dict[k] = v
            base_model.load_state_dict(state_dict)
            return base_model(*inputs)

        @staticmethod
        def vmap(func, randomness=None):
            def wrapper(params, buffers, inputs):
                # naive batch loop over ensemble dim 0
                results = []
                for i in range(params[list(params.keys())[0]].shape[0]):
                    # Extract param slice for member i
                    param_slice = {k: v[i] for k, v in params.items()}
                    buffer_slice = {k: v[i] for k, v in buffers.items()}
                    results.append(func(param_slice, buffer_slice, inputs[i]))
                return torch.stack(results)

            return wrapper


# The Ensemble class as you provided should be imported here
# from your_module import Ensemble


# Simple test model to use as prototype
class SimpleModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def ensemble_instance():
    n_members = 3
    models = [SimpleModel() for _ in range(n_members)]
    prototype = copy.deepcopy(models[0])
    ensemble = Ensemble(
        n_members=n_members, prototype=prototype, models=models, device="cpu"
    )
    return ensemble


def test_initialization(ensemble_instance):
    ensemble = ensemble_instance
    assert ensemble.n_members == 3
    # Check params and buffers keys correspond to model state dict keys
    for name in ensemble.prototype.state_dict().keys():
        assert f"stacked-{name.replace('.', '_')}" in ensemble.state_dict()


def test_forward_output_shape(ensemble_instance):
    ensemble = ensemble_instance
    batch_size = 5
    input_dim = 4
    x = torch.randn(batch_size, input_dim)
    output = ensemble(x)
    # Output shape should be (n_members, batch_size, output_dim)
    assert output.shape[0] == ensemble.n_members
    assert output.shape[1] == batch_size
    assert output.shape[2] == ensemble.prototype.linear.out_features


def test_expand_method(ensemble_instance):
    ensemble = ensemble_instance
    # Test for input with less than 3 dims
    x = torch.randn(2, 4)
    expanded = ensemble.expand(x)
    assert expanded.shape[0] == ensemble.n_members
    assert expanded.shape[1] == 2
    assert expanded.shape[2] == 4

    # Test for input with 3 dims or more (should not expand)
    x_3d = torch.randn(ensemble.n_members, 2, 4)
    expanded_3d = ensemble.expand(x_3d)
    assert torch.equal(expanded_3d, x_3d)


def test_get_single_member_and_getitem(ensemble_instance):
    ensemble = ensemble_instance
    single0 = ensemble._get_single_member(0)
    single1 = ensemble._get_single_member(1)
    item0 = ensemble[0]
    item1 = ensemble[1]

    # The returned members should be instances of prototype type
    assert isinstance(single0, type(ensemble.prototype))
    assert isinstance(single1, type(ensemble.prototype))
    assert isinstance(item0, type(ensemble.prototype))
    assert isinstance(item1, type(ensemble.prototype))

    # Parameters of extracted member should match stacked params at index
    for name, param in single0.named_parameters():
        stacked_param = ensemble.params[name][0]
        assert torch.allclose(param, stacked_param)


def test_get_all_members(ensemble_instance):
    ensemble = ensemble_instance
    members = ensemble._get_all_members()
    assert isinstance(members, nn.ModuleList)
    assert len(members) == ensemble.n_members
    for i, member in enumerate(members):
        for name, param in member.named_parameters():
            stacked_param = ensemble.params[name][i]
            assert torch.allclose(param, stacked_param)
