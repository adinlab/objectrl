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

import pytest
from dataclasses import asdict
from objectrl.config import model
from objectrl.config.model_configs import model_configs, actor_configs, critic_configs
from objectrl.config.utils import create_field_dict


def test_actor_config_default_and_override():
    model_name = list(actor_configs.keys())[0]
    default_dict = create_field_dict(actor_configs[model_name])

    actor_cfg = model.ActorConfig.from_config({}, model_name)

    for key, val in default_dict.items():
        assert getattr(actor_cfg, key) == val

    override = {"depth": 5, "width": 512, "activation": "crelu"}
    actor_cfg2 = model.ActorConfig.from_config(override, model_name)
    for key, val in override.items():
        assert getattr(actor_cfg2, key) == val

    extra = {"extra_param": 123}
    actor_cfg3 = model.ActorConfig.from_config(extra, model_name)
    assert hasattr(actor_cfg3, "extra_param")
    assert actor_cfg3.extra_param == 123


def test_actor_config_to_dict():
    model_name = list(model_configs.keys())[0]
    actor_cfg = model.ActorConfig.from_config({}, model_name)
    d = actor_cfg.to_dict()
    assert isinstance(d, dict)
    assert "depth" in d
    assert d["depth"] == actor_cfg.depth


def test_critic_config_default_and_override():
    model_name = list(critic_configs.keys())[0]
    default_dict = create_field_dict(critic_configs[model_name])

    critic_cfg = model.CriticConfig.from_config({}, model_name)
    for key, val in default_dict.items():
        assert getattr(critic_cfg, key) == val

    override = {"depth": 4, "width": 128, "activation": "crelu"}
    critic_cfg2 = model.CriticConfig.from_config(override, model_name)
    for key, val in override.items():
        assert getattr(critic_cfg2, key) == val

    extra = {"extra_param": "abc"}
    critic_cfg3 = model.CriticConfig.from_config(extra, model_name)
    assert hasattr(critic_cfg3, "extra_param")
    assert critic_cfg3.extra_param == "abc"


def test_critic_config_to_dict():
    model_name = list(model_configs.keys())[0]
    critic_cfg = model.CriticConfig.from_config({}, model_name)
    d = critic_cfg.to_dict()
    assert isinstance(d, dict)
    assert "depth" in d
    assert d["depth"] == critic_cfg.depth


def test_model_config_default_and_override():
    model_name = list(model_configs.keys())[0]
    base_defaults_dict = create_field_dict(model_configs[model_name])

    import objectrl.config.model as model_module

    original_model_configs = model_module.model_configs
    try:
        patched_configs = dict(original_model_configs)
        patched_configs[model_name] = base_defaults_dict
        model_module.model_configs = patched_configs

        model_cfg = model.ModelConfig.from_config({}, model_name)
        assert hasattr(model_cfg, "name")

        override = {"name": "custom_name", "extra_param": 456}
        model_cfg2 = model.ModelConfig.from_config(override, model_name)
        assert model_cfg2.name == "custom_name"
        assert hasattr(model_cfg2, "extra_param")
        assert model_cfg2.extra_param == 456
    finally:
        model_module.model_configs = original_model_configs


def test_model_config_to_dict():
    model_cfg = model.ModelConfig(name="test_model")
    d = model_cfg.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == "test_model"
