# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.config import ConfigManager
from torchtitan.experiments.ft.checkpoint import FTCheckpointManager
from torchtitan.experiments.ft.optimizer import FTOptimizersContainer
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer


class TestConfigManager(unittest.TestCase):
    def test_model_config_args(self):
        """--module and --config together load the correct config."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"
        assert config.training.steps == 10

    def test_model_config_args_equals_form(self):
        """--module=X --config=Y form works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module=llama3", "--config=llama3_debugmodel"]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_model_without_config_errors(self):
        """--module alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--config is required"):
            config_manager.parse_args(["--module", "llama3"])

    def test_config_without_model_errors(self):
        """--config alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args(["--config", "llama3_debugmodel"])

    def test_missing_both_errors(self):
        """No --module or --config raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args([])

    def test_invalid_model_errors(self):
        """--module with unknown module name raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(["--module", "nonexistent", "--config", "foo"])

    def test_invalid_config_errors(self):
        """--config with unknown function name lists available functions."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="Available config functions"):
            config_manager.parse_args(["--module", "llama3", "--config", "nonexistent"])

    def test_cli_overrides(self):
        """CLI args override config defaults."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.steps",
                "5",
            ]
        )
        assert config.training.steps == 5

    def test_config_file_loads_registry_config_and_yaml_overrides(self):
        """--config-file loads a registry config and applies YAML overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """\
orch:
  torchtitan_config:
    type: registry
    module: llama3
    config: llama3_debugmodel
training:
  steps: 7
  local_batch_size: 2
parallelism:
  data_parallel_shard_degree: 3
primus_turbo:
  enable_primus_turbo: false
""",
                encoding="utf-8",
            )

            config_manager = ConfigManager()
            config = config_manager.parse_args(["--config-file", str(config_path)])

        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"
        assert config.training.steps == 7
        assert config.training.local_batch_size == 2
        assert config.parallelism.data_parallel_shard_degree == 3

    def test_config_file_loads_hf_config_model_source_without_orch(self):
        """--config-file can build model_spec from HF config without orch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            assets_dir = root / "assets" / "hf" / "Qwen3-0.6B"
            assets_dir.mkdir(parents=True)
            (assets_dir / "config.json").write_text(
                json.dumps(
                    {
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
                ),
                encoding="utf-8",
            )
            (assets_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""\
model_source:
  type: hf_config
  architecture: auto
  hf_repo: Qwen/Qwen3-0.6B
  download:
    initial_load: true
    assets: [tokenizer, config]
hf_assets_path: {assets_dir}
training:
  steps: 7
""",
                encoding="utf-8",
            )

            config_manager = ConfigManager()
            config = config_manager.parse_args(
                ["--config-file", str(config_path), "--training.steps", "5"]
            )

        assert config.model_spec.name == "qwen3"
        assert config.model_spec.flavor == "hf_config"
        assert config.model_spec.model.dim == 1024
        assert len(config.model_spec.model.layers) == 2
        assert isinstance(config.dataloader, HuggingFaceTextDataLoader.Config)
        assert isinstance(config.loss, ChunkedCELoss.Config)
        assert config.training.steps == 5
        assert config.hf_assets_path == str(assets_dir)
        assert config.checkpoint.enable is True
        assert config.checkpoint.initial_load_in_hf is True
        assert config.checkpoint.initial_load_model_only is True
        assert config.model_source.download.assets == ["tokenizer", "config"]

    def test_config_file_loads_hf_config_fault_tolerant_trainer(self):
        """hf_config YAML with fault_tolerance uses FT trainer components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            assets_dir = root / "assets" / "hf" / "Qwen3-0.6B"
            assets_dir.mkdir(parents=True)
            config_json = assets_dir / "config.json"
            config_json.write_text(
                json.dumps(
                    {
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
                ),
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""\
model_source:
  type: hf_config
  architecture: auto
  config_json: {config_json}
  download:
    enabled: false
hf_assets_path: {assets_dir}
fault_tolerance:
  enable: true
  replica_id: 1
  group_size: 2
  process_group: gloo
checkpoint:
  enable_ft_dataloader_checkpoints: false
""",
                encoding="utf-8",
            )

            config_manager = ConfigManager()
            config = config_manager.parse_args(["--config-file", str(config_path)])

        assert isinstance(config, FaultTolerantTrainer.Config)
        assert isinstance(config.optimizer, FTOptimizersContainer.Config)
        assert isinstance(config.checkpoint, FTCheckpointManager.Config)
        assert config.fault_tolerance.enable is True
        assert config.fault_tolerance.replica_id == 1
        assert config.fault_tolerance.group_size == 2
        assert config.checkpoint.enable_ft_dataloader_checkpoints is False
        assert config.model_spec.name == "qwen3"
        assert config.model_spec.flavor == "hf_config"

    def test_hf_config_downloads_missing_assets(self):
        """hf_config YAML downloads missing local assets when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_manager = ConfigManager()
            config_manager._config_file_base_dir = root
            config_manager.config = Trainer.Config()
            config_manager.config.hf_assets_path = str(
                root / "assets" / "hf" / "Qwen3-0.6B"
            )
            config_manager.config.model_source.type = "hf_config"
            config_manager.config.model_source.hf_repo = "Qwen/Qwen3-0.6B"
            config_manager.config.model_source.download.assets = ["tokenizer", "config"]

            def fake_download(cmd):
                local_dir = Path(cmd[cmd.index("--local_dir") + 1]) / "Qwen3-0.6B"
                local_dir.mkdir(parents=True)
                (local_dir / "config.json").write_text("{}", encoding="utf-8")
                (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0)

            with patch("subprocess.run", side_effect=fake_download) as run:
                config_manager._prepare_model_source_assets(
                    config_manager.config.model_source
                )

        run.assert_called_once()

    def test_config_file_cli_overrides_yaml_values(self):
        """CLI args override values loaded from --config-file YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """\
orch:
  torchtitan_config:
    type: registry
    module: llama3
    config: llama3_debugmodel
training:
  steps: 7
""",
                encoding="utf-8",
            )

            config_manager = ConfigManager()
            config = config_manager.parse_args(
                ["--config-file", str(config_path), "--training.steps", "5"]
            )

        assert config.training.steps == 5

    def test_config_file_underscore_flag_form(self):
        """--config_file is accepted as an alias for --config-file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """\
orch:
  torchtitan_config:
    module: llama3
    config: llama3_debugmodel
""",
                encoding="utf-8",
            )

            config_manager = ConfigManager()
            config = config_manager.parse_args(["--config_file", str(config_path)])

        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_config_file_unknown_field_errors(self):
        """Unknown YAML fields fail with a clear nested field path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """\
orch:
  torchtitan_config:
    module: llama3
    config: llama3_debugmodel
training:
  not_a_real_field: 1
""",
                encoding="utf-8",
            )

            config_manager = ConfigManager()
            with pytest.raises(
                ValueError, match="Unsupported YAML config field: training.not_a_real_field"
            ):
                config_manager.parse_args(["--config-file", str(config_path)])

    def test_config_file_missing_registry_source_errors(self):
        """YAML config files must select a registry base config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """\
training:
  steps: 7
""",
                encoding="utf-8",
            )

            config_manager = ConfigManager()
            with pytest.raises(
                ValueError,
                match=(
                    "--config-file requires model_source.type=hf_config "
                    "or orch.torchtitan_config"
                ),
            ):
                config_manager.parse_args(["--config-file", str(config_path)])

    def test_config_file_missing_module_or_config_errors(self):
        """orch.torchtitan_config requires both module and config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """\
orch:
  torchtitan_config:
    module: llama3
""",
                encoding="utf-8",
            )

            config_manager = ConfigManager()
            with pytest.raises(
                ValueError, match="orch.torchtitan_config requires module and config"
            ):
                config_manager.parse_args(["--config-file", str(config_path)])

    def test_cli_override_dump_folder(self):
        """CLI args override config defaults for nested fields."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--dump_folder",
                "/tmp/test_tt/",
            ]
        )
        assert config.dump_folder == "/tmp/test_tt/"

    def test_parse_module_fqns_per_model_part(self):
        """module_fqns_per_model_part defaults to None."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.parallelism.module_fqns_per_model_part is None

    def test_parse_exclude_from_loading(self):
        """exclude_from_loading defaults to [] and can be overridden."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.checkpoint.exclude_from_loading == []

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--checkpoint.exclude_from_loading",
                "optimizer,lr_scheduler",
            ]
        )
        assert config.checkpoint.exclude_from_loading == [
            "optimizer",
            "lr_scheduler",
        ]

    def test_trainer_config_quantization_default(self):
        from torchtitan.components.quantization.utils import has_quantization

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert not has_quantization(config.model_spec.model)

    # TODO: remove this test when we remove the merge functionality
    def test_extend_trainer_config_directly(self):
        """Test that _merge_configs works to extend config types."""
        from dataclasses import dataclass

        @dataclass
        class CustomCheckpoint:
            convert_path: str = "/custom/path"
            fake_model: bool = True

        @dataclass
        class CustomTrainerConfig:
            checkpoint: CustomCheckpoint

        MergedTrainerConfig = ConfigManager._merge_configs(
            Trainer.Config, CustomTrainerConfig
        )

        # Verify the merged type has both base and custom fields
        merged = MergedTrainerConfig()
        assert hasattr(merged, "checkpoint")
        assert hasattr(merged.checkpoint, "convert_path")
        assert merged.checkpoint.convert_path == "/custom/path"
        assert merged.checkpoint.fake_model is True
        assert hasattr(merged, "model_spec")

    def test_flux_config_via_cli(self):
        """Test that --module flux --config flux_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "flux", "--config", "flux_debugmodel"]
        )
        assert config.model_spec.name == "flux"
        assert hasattr(config, "encoder")

    def test_deepseek_config(self):
        """Test that --module deepseek_v3 --config deepseek_v3_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "deepseek_v3", "--config", "deepseek_v3_debugmodel"]
        )
        assert config.model_spec.name == "deepseek_v3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_with_config_registry(self):
        """--module torchtitan.models.llama3.config_registry works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan.models.llama3.config_registry",
                "--config",
                "llama3_debugmodel",
            ]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_without_config_registry(self):
        """--module torchtitan.models.llama3 (auto-appends .config_registry)."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan.models.llama3",
                "--config",
                "llama3_debugmodel",
            ]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_invalid_errors(self):
        """--module with invalid FQN raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(
                [
                    "--module",
                    "torchtitan.models.nonexistent",
                    "--config",
                    "foo",
                ]
            )


if __name__ == "__main__":
    unittest.main()
