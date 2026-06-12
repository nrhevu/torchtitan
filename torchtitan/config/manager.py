# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys
import warnings
from collections.abc import Mapping
from dataclasses import field, fields, is_dataclass, make_dataclass
from pathlib import Path
from typing import Any

import tyro

from torchtitan.tools.logging import logger


class ConfigManager:
    """
    Parses, merges, and validates a config from --module/--config,
    --config-file, and CLI sources.

    Configuration precedence:
        CLI args > YAML config file > config_registry function defaults

    --module selects the module (e.g., llama3, deepseek_v3).
    --config selects a config_registry function (e.g., llama3_debugmodel).
    --config-file selects a YAML file that points at a config_registry function
    and overlays config values.
    CLI arguments use the format <section>.<key> to override config values.
    """

    def __init__(self):
        self.register_tyro_rules(custom_registry)

    def parse_args(self, args: list[str] = sys.argv[1:]):
        loaded_config, args = self._load_config(args)
        config_cls = type(loaded_config)

        self.config = tyro.cli(
            config_cls, args=args, default=loaded_config, registry=custom_registry
        )

        self._validate_config()

        return self.config

    def _load_config(self, args: list[str]) -> tuple[object, list[str]]:
        """Load config from --config-file or from --module/--config.

        Returns (loaded_config, filtered_args) with config source args stripped.
        """
        module_name = None
        config_name = None
        config_file = None
        filtered_args = []

        i = 0
        while i < len(args):
            arg = args[i]

            # Handle --config-file=X, --config_file=X, and split forms.
            if arg.startswith(("--config-file=", "--config_file=")):
                config_file = arg.split("=", 1)[1]
            elif arg in ("--config-file", "--config_file"):
                if i + 1 < len(args):
                    config_file = args[i + 1]
                    i += 1
                else:
                    raise ValueError(f"{arg} requires a value")
            # Handle --module=X and --module X forms
            elif arg.startswith("--module="):
                module_name = arg.split("=", 1)[1]
            elif arg == "--module":
                if i + 1 < len(args):
                    module_name = args[i + 1]
                    i += 1
                else:
                    raise ValueError("--module requires a value")
            # Handle --config=X and --config X forms
            elif arg.startswith("--config="):
                config_name = arg.split("=", 1)[1]
            elif arg == "--config":
                if i + 1 < len(args):
                    config_name = args[i + 1]
                    i += 1
                else:
                    raise ValueError("--config requires a value")
            else:
                filtered_args.append(arg)

            i += 1

        if config_file is not None:
            if module_name is not None or config_name is not None:
                raise ValueError(
                    "--config-file cannot be combined with --module or --config"
                )
            return self._load_config_file(config_file), filtered_args

        if module_name is None:
            raise ValueError(
                "--module is required. Example: --module llama3 --config llama3_debugmodel"
            )
        if config_name is None:
            raise ValueError(
                "--config is required. Example: --module llama3 --config llama3_debugmodel"
            )

        return self._load_registry_config(module_name, config_name), filtered_args

    def _load_config_file(self, config_file: str) -> object:
        """Load a YAML config file that selects and overlays a registry config."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "--config-file requires PyYAML. Install it with `pip install PyYAML`."
            ) from exc

        config_path = Path(config_file).expanduser()
        with config_path.open(encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        if yaml_config is None:
            yaml_config = {}
        if not isinstance(yaml_config, Mapping):
            raise TypeError("--config-file YAML must contain a mapping at the top level")

        source = self._get_yaml_registry_source(yaml_config)
        if source is None:
            raise ValueError(
                "--config-file requires orch.torchtitan_config with module and config"
            )

        source_type = source.get("type", "registry")
        if source_type not in {"registry", "torchtitan_registry"}:
            raise ValueError(f"Unsupported torchtitan_config type: {source_type}")

        module_name = source.get("module")
        config_name = source.get("config")
        if not module_name or not config_name:
            raise ValueError("orch.torchtitan_config requires module and config")

        loaded_config = self._load_registry_config(str(module_name), str(config_name))
        metadata_keys = {
            "orch",
            "torchtitan_config",
            "config_source",
            "job",
            "primus_turbo",
        }
        payload = {
            key: value for key, value in yaml_config.items() if key not in metadata_keys
        }
        self._overlay_dataclass_config(loaded_config, payload)
        return loaded_config

    @staticmethod
    def _get_yaml_registry_source(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
        orch = config.get("orch", {})
        if orch is None:
            orch = {}
        if not isinstance(orch, Mapping):
            raise TypeError("orch must be a mapping")

        source = orch.get("torchtitan_config") or orch.get("config_source")
        if source is None:
            source = config.get("torchtitan_config") or config.get("config_source")
        if source is None:
            return None
        if not isinstance(source, Mapping):
            raise TypeError("orch.torchtitan_config must be a mapping")
        return source

    @classmethod
    def _overlay_dataclass_config(
        cls, config: object, data: Mapping[str, Any], *, path: str = ""
    ) -> None:
        if not is_dataclass(config):
            raise TypeError(
                f"Cannot overlay YAML mapping onto non-dataclass config at {path or '<root>'}"
            )

        field_by_name = {item.name: item for item in fields(config)}
        for raw_key, value in data.items():
            key = str(raw_key).replace("-", "_")
            field_path = f"{path}.{key}" if path else key
            if key not in field_by_name:
                raise ValueError(f"Unsupported YAML config field: {field_path}")

            current_value = getattr(config, key)
            if isinstance(value, Mapping) and is_dataclass(current_value):
                cls._overlay_dataclass_config(current_value, value, path=field_path)
            else:
                setattr(config, key, value)

    def _load_registry_config(self, module_name: str, config_name: str) -> object:
        from torchtitan.experiments import _supported_experiments

        # Validate module name
        from torchtitan.models import _supported_models

        all_supported = _supported_models | _supported_experiments

        module = None
        module_path = None

        # Import config_registry from module based on module specification
        if module_name in all_supported:
            # short module from supported module list  (search models first, then experiments)
            for prefix in ("torchtitan.models", "torchtitan.experiments"):
                module_path = f"{prefix}.{module_name}.config_registry"
                try:
                    module = importlib.import_module(module_path)
                    break
                except ImportError:
                    continue
            if module is None:
                raise ImportError(
                    f"Cannot import config_registry for module '{module_name}' "
                    f"from torchtitan.models or torchtitan.experiments"
                )
        else:
            # Fully qualified module path: try appending .config_registry first,
            # then fall back to importing directly (e.g., torchtitan.models.llama3
            # -> torchtitan.models.llama3.config_registry)
            for candidate in (f"{module_name}.config_registry", module_name):
                try:
                    module = importlib.import_module(candidate)
                    module_path = candidate
                    break
                except ImportError:
                    continue
            if module is None:
                raise ImportError(
                    f"Cannot import module '{module_name}' or "
                    f"'{module_name}.config_registry'. "
                    f"For shorthands, supported modules are: {sorted(all_supported)}"
                )

        # Get the config function
        config_fn = getattr(module, config_name, None)
        if config_fn is None or not callable(config_fn):
            available = [
                name
                for name in dir(module)
                if not name.startswith("_")
                and callable(getattr(module, name))
                and name[0].islower()
            ]
            raise ValueError(
                f"Config function '{config_name}' not found in {module_path}. "
                f"Available config functions: {available}"
            )

        loaded_config = config_fn()
        return loaded_config

    @staticmethod
    def _merge_configs(base, custom) -> type:
        """
        Merges a base config class with user-defined extensions.
        """
        warnings.warn(
            "ConfigManager._merge_configs is deprecated. "
            "Use Config subclasses with config_registry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result: list[str | tuple[str, Any] | tuple[str, Any, Any]] = []
        b_map = {f.name: f for f in fields(base)}
        c_map = {f.name: f for f in fields(custom)}

        for name, f in b_map.items():
            if (
                name in c_map
                and is_dataclass(f.type)
                and is_dataclass(c_map[name].type)
            ):
                m_type = ConfigManager._merge_configs(f.type, c_map[name].type)
                result.append((name, m_type, field(default_factory=m_type)))

            # Custom field overrides base type
            elif name in c_map:
                result.append((name, c_map[name].type, c_map[name]))

            # Only in Base
            else:
                result.append((name, f.type, f))

        # Only in Custom
        for name, f in c_map.items():
            if name not in b_map:
                result.append((name, f.type, f))

        return make_dataclass(f"Merged{base.__name__}", result, bases=(base,))

    def _validate_config(self) -> None:
        # TODO: temporary mitigation of BC breaking change in hf_assets_path
        #       tokenizer default path, need to remove later
        if not os.path.exists(
            self.config.hf_assets_path  # pyrefly: ignore[missing-attribute]
        ):
            logger.warning(
                f"HF assets path {self.config.hf_assets_path} does not exist!"
            )
            old_tokenizer_path = (
                "torchtitan/datasets/tokenizer/original/tokenizer.model"
            )
            if os.path.exists(old_tokenizer_path):
                self.config.hf_assets_path = (  # pyrefly: ignore[missing-attribute]
                    old_tokenizer_path
                )
                logger.warning(
                    f"Temporarily switching to previous default tokenizer path {old_tokenizer_path}. "
                    "Please download the new tokenizer files (python scripts/download_hf_assets.py) and update your config."
                )
        else:
            # Check if we are using tokenizer.model, if so then we need to alert users to redownload the tokenizer
            if self.config.hf_assets_path.endswith(  # pyrefly: ignore[missing-attribute]
                "tokenizer.model"
            ):
                raise Exception(
                    "You are using the old tokenizer.model, please redownload the tokenizer ",
                    "(python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer) ",
                    " and update your config to the directory of the downloaded tokenizer.",
                )

    @staticmethod
    def register_tyro_rules(registry: tyro.constructors.ConstructorRegistry) -> None:
        @registry.primitive_rule
        def list_str_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Support for comma separated string parsing"""
            if type_info.type != list[str]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="A,B,C,...",
                instance_from_str=lambda args: args[0].split(","),
                is_instance=lambda instance: all(isinstance(i, str) for i in instance),
                str_from_instance=lambda instance: [",".join(instance)],
            )


# Initialize the custom registry for tyro
custom_registry = tyro.constructors.ConstructorRegistry()


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Run this module directly to debug or inspect configuration parsing.
    #
    # Examples:
    #   Parse and print a config with CLI arguments:
    #     > python -m torchtitan.config.manager --module llama3 --config llama3_debugmodel
    #
    #   Show help message:
    #     > python -m torchtitan.config.manager --module llama3 --config llama3_debugmodel --help
    #
    # -----------------------------------------------------------------------------

    try:

        from rich import print as rprint
        from rich.pretty import Pretty

        config_manager = ConfigManager()
        config = config_manager.parse_args()

        rprint(Pretty(config))
    except ImportError:
        config_manager = ConfigManager()
        config = config_manager.parse_args()
        logger.info(config)
        logger.warning("rich is not installed, show the raw config")
