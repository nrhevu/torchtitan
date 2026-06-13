# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

from torchtitan.config import ModelSourceConfig
from torchtitan.models.hf_config import (
    build_model_spec,
    build_model_spec_from_hf_config,
    resolve_hf_config_json_path,
)


def _dense_qwen3_config(**overrides):
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": True,
        "intermediate_size": 3072,
    }
    config.update(overrides)
    return config


def _moe_qwen3_config(**overrides):
    config = _dense_qwen3_config(
        architectures=["Qwen3MoeForCausalLM"],
        model_type="qwen3_moe",
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
    )
    config.update(overrides)
    return config


def _write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")


def test_build_qwen3_dense_spec_from_hf_assets_path(tmp_path):
    assets_dir = tmp_path / "Qwen3-0.6B"
    _write_config(assets_dir / "config.json", _dense_qwen3_config())

    model_source = ModelSourceConfig(
        type="hf_config",
        architecture="auto",
        hf_repo="Qwen/Qwen3-0.6B",
    )

    spec = build_model_spec(model_source, hf_assets_path=assets_dir)

    assert spec.name == "qwen3"
    assert spec.flavor == "hf_config"
    assert spec.model.dim == 1024
    assert spec.model.vocab_size == 151936
    assert spec.model.enable_weight_tying is True
    assert len(spec.model.layers) == 2
    assert spec.model.layers[0].attention.n_heads == 16
    assert spec.model.layers[0].attention.n_kv_heads == 8
    assert spec.model.layers[0].feed_forward.w1.out_features == 3072


def test_hf_repo_infers_local_assets_dir(tmp_path):
    config_path = tmp_path / "assets" / "hf" / "Qwen3-0.6B" / "config.json"
    _write_config(config_path, _dense_qwen3_config())

    model_source = ModelSourceConfig(type="hf_config", hf_repo="Qwen/Qwen3-0.6B")

    assert resolve_hf_config_json_path(model_source, base_dir=tmp_path) == config_path


def test_config_json_is_resolved_relative_to_base_dir(tmp_path):
    config_path = tmp_path / "configs" / "qwen3" / "config.json"
    _write_config(config_path, _dense_qwen3_config())

    model_source = ModelSourceConfig(
        type="hf_config",
        config_json="configs/qwen3/config.json",
    )

    spec = build_model_spec(model_source, base_dir=tmp_path)

    assert spec.model.dim == 1024
    assert spec.model.rope.max_seq_len == 4096


def test_hf_repo_without_local_config_errors(tmp_path):
    model_source = ModelSourceConfig(type="hf_config", hf_repo="Qwen/Qwen3-0.6B")

    with pytest.raises(FileNotFoundError, match="requires a local config.json"):
        build_model_spec(model_source, base_dir=tmp_path)


def test_build_qwen3_moe_spec_from_hf_config():
    model_source = ModelSourceConfig(type="hf_config", architecture="auto")

    spec = build_model_spec_from_hf_config(_moe_qwen3_config(), model_source)

    assert spec.name == "qwen3"
    assert spec.flavor == "hf_config"
    assert spec.model.layers[0].moe.num_experts == 128
    assert spec.model.layers[0].moe.router.top_k == 8
    assert spec.model.layers[0].moe.experts.hidden_dim == 768


def test_unsupported_hf_architecture_errors():
    model_source = ModelSourceConfig(type="hf_config", architecture="auto")
    hf_config = _dense_qwen3_config(architectures=["UnsupportedForCausalLM"])

    with pytest.raises(ValueError, match="Unsupported HF architecture"):
        build_model_spec_from_hf_config(hf_config, model_source)
