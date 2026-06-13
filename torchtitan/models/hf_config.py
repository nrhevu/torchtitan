# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build TorchTitan model specs from Hugging Face model configs."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any


HFConfigBuilder = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]

SUPPORTED_QWEN_DENSE_MODEL_TYPES = {"qwen3"}
SUPPORTED_QWEN_MOE_MODEL_TYPES = {"qwen3_moe"}


def _resolve_config_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or base_dir is None or candidate.exists():
        return candidate.resolve()
    return (Path(base_dir).expanduser() / candidate).resolve()


def _model_source_to_mapping(model_source: object) -> dict[str, Any]:
    if dataclasses.is_dataclass(model_source):
        return {
            field.name: getattr(model_source, field.name)
            for field in dataclasses.fields(model_source)
        }
    if isinstance(model_source, Mapping):
        return dict(model_source)
    raise TypeError("model_source must be a mapping or dataclass")


def load_hf_config_json(
    path: str | Path, *, base_dir: str | Path | None = None
) -> dict[str, Any]:
    config_path = _resolve_config_path(path, base_dir=base_dir)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("HF config.json must contain a JSON object")
    return data


def _repo_assets_config_path(
    repo_id: str, *, base_dir: str | Path | None = None
) -> Path:
    model_name = repo_id.rstrip("/").split("/")[-1]
    if not model_name:
        raise ValueError(f"Invalid model_source.hf_repo value: {repo_id!r}")
    return _resolve_config_path(
        Path("assets") / "hf" / model_name / "config.json", base_dir=base_dir
    )


def resolve_hf_config_json_path(
    model_source: object,
    *,
    base_dir: str | Path | None = None,
    hf_assets_path: str | Path | None = None,
) -> Path:
    source = _model_source_to_mapping(model_source)

    config_json = source.get("config_json")
    if config_json:
        return _resolve_config_path(str(config_json), base_dir=base_dir)

    if hf_assets_path:
        local_config = (Path(hf_assets_path).expanduser() / "config.json").resolve()
        if local_config.is_file():
            return local_config
        if source.get("hf_repo"):
            raise FileNotFoundError(
                "model_source.type=hf_config expected config.json at "
                f"hf_assets_path: {local_config}"
            )

    repo_id = source.get("hf_repo")
    if repo_id:
        inferred_config = _repo_assets_config_path(str(repo_id), base_dir=base_dir)
        if inferred_config.is_file():
            return inferred_config
        raise FileNotFoundError(
            "model_source.type=hf_config with hf_repo requires a local config.json. "
            f"Checked: {inferred_config}"
        )

    raise ValueError(
        "model_source.type=hf_config requires model_source.config_json, "
        "hf_assets_path/config.json, or model_source.hf_repo with local assets"
    )


def load_hf_config(
    model_source: object,
    *,
    base_dir: str | Path | None = None,
    hf_assets_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = resolve_hf_config_json_path(
        model_source, base_dir=base_dir, hf_assets_path=hf_assets_path
    )
    return load_hf_config_json(config_path)


def _supported_architectures() -> str:
    return ", ".join(sorted(HF_ARCHITECTURE_BUILDERS))


def select_architecture(hf_config: Mapping[str, Any], model_source: object) -> str:
    source = _model_source_to_mapping(model_source)
    requested = source.get("architecture", "auto")
    if requested and requested != "auto":
        architecture = str(requested)
    else:
        architectures = hf_config.get("architectures")
        if not isinstance(architectures, list) or not architectures:
            raise ValueError(
                "HF config must include a non-empty architectures list when "
                "model_source.architecture is absent or auto"
            )
        architecture = str(architectures[0])

    if architecture not in HF_ARCHITECTURE_BUILDERS:
        raise ValueError(
            f"Unsupported HF architecture {architecture!r}. "
            f"Supported architectures: {_supported_architectures()}"
        )
    return architecture


def build_model_spec_from_hf_config(
    hf_config: Mapping[str, Any], model_source: object
) -> Any:
    source = _model_source_to_mapping(model_source)
    architecture = select_architecture(hf_config, source)
    return HF_ARCHITECTURE_BUILDERS[architecture](hf_config, source)


def build_model_spec(
    model_source: object,
    *,
    base_dir: str | Path | None = None,
    hf_assets_path: str | Path | None = None,
) -> Any:
    hf_config = load_hf_config(
        model_source, base_dir=base_dir, hf_assets_path=hf_assets_path
    )
    return build_model_spec_from_hf_config(hf_config, model_source)


def _require_int(config: Mapping[str, Any], key: str) -> int:
    if key not in config or config[key] is None:
        raise ValueError(f"HF config is missing required field {key!r}")
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"HF config field {key!r} must be an integer")
    return value


def _require_number(config: Mapping[str, Any], key: str) -> int | float:
    if key not in config or config[key] is None:
        raise ValueError(f"HF config is missing required field {key!r}")
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"HF config field {key!r} must be numeric")
    return value


def _require_bool(config: Mapping[str, Any], key: str) -> bool:
    if key not in config:
        raise ValueError(f"HF config is missing required field {key!r}")
    value = config[key]
    if not isinstance(value, bool):
        raise TypeError(f"HF config field {key!r} must be a boolean")
    return value


def _validate_no_unsupported_qwen_features(config: Mapping[str, Any]) -> None:
    mlp_only_layers = config.get("mlp_only_layers", [])
    if mlp_only_layers:
        raise ValueError("Qwen HF configs with non-empty mlp_only_layers are not supported")

    use_sliding_window = config.get("use_sliding_window")
    if use_sliding_window:
        raise ValueError("Qwen HF configs with sliding-window attention are not supported")
    sliding_window = config.get("sliding_window")
    if use_sliding_window is None and sliding_window not in (None, False, 0):
        raise ValueError("Qwen HF configs with sliding-window attention are not supported")


def _validate_model_type(config: Mapping[str, Any], expected: set[str]) -> None:
    model_type = config.get("model_type")
    if model_type not in expected:
        expected_values = ", ".join(sorted(expected))
        raise ValueError(
            f"Unsupported Qwen model_type {model_type!r}; expected one of: {expected_values}"
        )


def _qwen_common_dims(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dim": _require_int(config, "hidden_size"),
        "n_layers": _require_int(config, "num_hidden_layers"),
        "n_heads": _require_int(config, "num_attention_heads"),
        "n_kv_heads": _require_int(config, "num_key_value_heads"),
        "head_dim": _require_int(config, "head_dim"),
        "vocab_size": _require_int(config, "vocab_size"),
        "rope_theta": float(_require_number(config, "rope_theta")),
        "max_seq_len": _require_int(config, "max_position_embeddings"),
        "enable_weight_tying": _require_bool(config, "tie_word_embeddings"),
    }


def _load_qwen3_components() -> SimpleNamespace:
    from torchtitan.distributed.pipeline_parallel import pipeline_llm
    from torchtitan.models.common import Embedding, Linear, RoPE
    from torchtitan.models.qwen3 import (
        _build_qwen3_layers,
        _build_qwen3_moe_layers,
        _EMBEDDING_INIT,
        _EMBEDDING_SKIP_INIT,
        _output_linear_init,
        _qwen3_norm,
        parallelize_qwen3,
    )
    from torchtitan.models.qwen3.model import Qwen3Model
    from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
    from torchtitan.protocols.model_spec import ModelSpec

    return SimpleNamespace(
        Embedding=Embedding,
        Linear=Linear,
        ModelSpec=ModelSpec,
        pipeline_llm=pipeline_llm,
        parallelize_qwen3=parallelize_qwen3,
        Qwen3Model=Qwen3Model,
        Qwen3StateDictAdapter=Qwen3StateDictAdapter,
        RoPE=RoPE,
        build_layers=_build_qwen3_layers,
        build_moe_layers=_build_qwen3_moe_layers,
        embedding_init=_EMBEDDING_INIT,
        embedding_skip_init=_EMBEDDING_SKIP_INIT,
        output_linear_init=_output_linear_init,
        qwen3_norm=_qwen3_norm,
    )


def _build_qwen3_spec(
    *,
    model_config: Any,
    components: SimpleNamespace,
) -> Any:
    return components.ModelSpec(
        name="qwen3",
        flavor="hf_config",
        model=model_config,
        parallelize_fn=components.parallelize_qwen3,
        pipelining_fn=components.pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=components.Qwen3StateDictAdapter,
    )


def _qwen3_base_model_config(
    dims: Mapping[str, Any],
    *,
    layers: list[Any],
    components: SimpleNamespace,
) -> Any:
    dim = int(dims["dim"])
    vocab_size = int(dims["vocab_size"])
    embedding_init = (
        components.embedding_skip_init
        if dims["enable_weight_tying"]
        else components.embedding_init
    )
    return components.Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=components.qwen3_norm(dim),
        enable_weight_tying=bool(dims["enable_weight_tying"]),
        tok_embeddings=components.Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=embedding_init,
        ),
        lm_head=components.Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=components.output_linear_init(dim),
        ),
        rope=components.RoPE.Config(
            dim=int(dims["head_dim"]),
            max_seq_len=int(dims["max_seq_len"]),
            theta=float(dims["rope_theta"]),
            backend="cos_sin",
        ),
        layers=layers,
    )


def qwen3_dense_model_spec(
    hf_config: Mapping[str, Any], model_source: Mapping[str, Any]
) -> Any:
    _validate_model_type(hf_config, SUPPORTED_QWEN_DENSE_MODEL_TYPES)
    _validate_no_unsupported_qwen_features(hf_config)
    dims = _qwen_common_dims(hf_config)
    hidden_dim = _require_int(hf_config, "intermediate_size")
    components = _load_qwen3_components()

    layers = components.build_layers(
        n_layers=dims["n_layers"],
        dim=dims["dim"],
        n_heads=dims["n_heads"],
        n_kv_heads=dims["n_kv_heads"],
        head_dim=dims["head_dim"],
        hidden_dim=hidden_dim,
        attn_backend=str(model_source.get("attn_backend", "sdpa")),
    )
    model_config = _qwen3_base_model_config(dims, layers=layers, components=components)
    return _build_qwen3_spec(model_config=model_config, components=components)


def qwen3_moe_model_spec(
    hf_config: Mapping[str, Any], model_source: Mapping[str, Any]
) -> Any:
    _validate_model_type(hf_config, SUPPORTED_QWEN_MOE_MODEL_TYPES)
    _validate_no_unsupported_qwen_features(hf_config)
    dims = _qwen_common_dims(hf_config)
    _require_int(hf_config, "intermediate_size")
    moe_hidden_dim = _require_int(hf_config, "moe_intermediate_size")
    num_experts = _require_int(hf_config, "num_experts")
    top_k = _require_int(hf_config, "num_experts_per_tok")
    components = _load_qwen3_components()

    layers = components.build_moe_layers(
        n_layers=dims["n_layers"],
        dim=dims["dim"],
        n_heads=dims["n_heads"],
        n_kv_heads=dims["n_kv_heads"],
        head_dim=dims["head_dim"],
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        attn_backend=str(model_source.get("attn_backend", "sdpa")),
        moe_comm_backend=str(model_source.get("moe_comm_backend", "standard")),
    )
    model_config = _qwen3_base_model_config(dims, layers=layers, components=components)
    return _build_qwen3_spec(model_config=model_config, components=components)


HF_ARCHITECTURE_BUILDERS: dict[str, HFConfigBuilder] = {
    "Qwen3ForCausalLM": qwen3_dense_model_spec,
    "Qwen3MoeForCausalLM": qwen3_moe_model_spec,
}
