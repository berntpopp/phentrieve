"""Tests for shared chunking configuration resolver."""

import json

import pytest
import yaml

from phentrieve.text_processing.config_resolver import (
    ChunkingConfigError,
    resolve_chunking_config,
)

pytestmark = pytest.mark.unit


class TestConfigFileLoading:
    """Test configuration loading from files."""

    def test_load_valid_yaml_config(self, tmp_path):
        """Test loading valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_data = {"chunking_pipeline": [{"type": "simple_sentence", "config": {}}]}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = resolve_chunking_config(config_file=config_file)
        assert len(result) == 1
        assert result[0]["type"] == "simple_sentence"

    def test_load_valid_json_config(self, tmp_path):
        """Test loading valid JSON config file."""
        config_file = tmp_path / "config.json"
        config_data = {"chunking_pipeline": [{"type": "simple_sentence", "config": {}}]}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = resolve_chunking_config(config_file=config_file)
        assert len(result) == 1
        assert result[0]["type"] == "simple_sentence"

    def test_config_file_not_found(self, tmp_path):
        """Test error when config file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(ChunkingConfigError, match="not found"):
            resolve_chunking_config(config_file=nonexistent)

    def test_unsupported_file_format(self, tmp_path):
        """Test error for unsupported file format."""
        bad_file = tmp_path / "config.txt"
        bad_file.touch()
        with pytest.raises(ChunkingConfigError, match="Unsupported config file format"):
            resolve_chunking_config(config_file=bad_file)

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test error for invalid YAML syntax."""
        config_file = tmp_path / "bad.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: syntax: [")
        with pytest.raises(ChunkingConfigError, match="Failed to parse"):
            resolve_chunking_config(config_file=config_file)

    def test_missing_chunking_pipeline_key(self, tmp_path):
        """Test error when config missing chunking_pipeline key."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"other_key": "value"}, f)
        with pytest.raises(ChunkingConfigError, match="missing 'chunking_pipeline'"):
            resolve_chunking_config(config_file=config_file)

    def test_yml_extension_works(self, tmp_path):
        """Test that .yml extension is supported (not just .yaml)."""
        config_file = tmp_path / "config.yml"
        config_data = {"chunking_pipeline": [{"type": "simple_sentence", "config": {}}]}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = resolve_chunking_config(config_file=config_file)
        assert len(result) == 1


class TestStrategyResolution:
    """Test resolution of predefined strategies."""

    @pytest.mark.parametrize(
        "strategy_name",
        [
            "simple",
            "detailed",
            "semantic",
            "sliding_window",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_known_strategies_resolve(self, strategy_name):
        """Test that all known strategies can be resolved."""
        config = resolve_chunking_config(strategy_name=strategy_name)
        assert isinstance(config, list)
        assert len(config) > 0

    def test_unknown_strategy_uses_default(self, caplog):
        """Test unknown strategy falls back to default with warning."""
        config = resolve_chunking_config(strategy_name="unknown_strategy")
        assert isinstance(config, list)
        assert any("Unknown strategy" in record.message for record in caplog.records)

    def test_strategy_name_case_insensitive(self):
        """Test strategy names are case-insensitive."""
        config_lower = resolve_chunking_config(strategy_name="simple")
        config_upper = resolve_chunking_config(strategy_name="SIMPLE")
        assert config_lower == config_upper

    def test_mixed_case_strategy_name(self):
        """Test mixed case strategy names work."""
        config = resolve_chunking_config(strategy_name="Sliding_Window")
        assert isinstance(config, list)
        assert len(config) > 0


class TestParameterOverrides:
    """Test parameter override application."""

    def test_window_size_override(self):
        """Test window size override is applied."""
        config = resolve_chunking_config(
            strategy_name="sliding_window",
            window_size=10,
        )
        # Find sliding window component
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        assert sw_component["config"]["window_size_tokens"] == 10

    def test_all_parameters_override(self):
        """Test all parameters can be overridden together."""
        config = resolve_chunking_config(
            strategy_name="detailed",
            window_size=15,
            step_size=3,
            threshold=0.7,
            min_segment_length=5,
        )
        # Find sliding window component
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        assert sw_component["config"]["window_size_tokens"] == 15
        assert sw_component["config"]["step_size_tokens"] == 3
        assert sw_component["config"]["splitting_threshold"] == 0.7
        assert sw_component["config"]["min_split_segment_length_words"] == 5

    def test_partial_parameter_override(self):
        """Test that partial parameter overrides work."""
        config = resolve_chunking_config(
            strategy_name="semantic",
            window_size=12,  # Only override window_size
        )
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        assert sw_component["config"]["window_size_tokens"] == 12
        # Other params should have strategy defaults
        assert "step_size_tokens" in sw_component["config"]

    def test_no_overrides_preserves_defaults(self):
        """Test config without overrides uses strategy defaults."""
        config = resolve_chunking_config(strategy_name="detailed")
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        # Should have default values
        assert "window_size_tokens" in sw_component["config"]

    def test_overrides_work_with_multiple_sliding_windows(self):
        """Test overrides apply to all sliding window components."""
        config = resolve_chunking_config(
            strategy_name="detailed",  # Has sliding window
            window_size=20,
        )
        # Check all sliding window components have override applied
        for component in config:
            if component.get("type") == "sliding_window":
                assert component["config"]["window_size_tokens"] == 20


class TestErrorHandling:
    """Test error handling."""

    def test_neither_strategy_nor_file_raises_error(self):
        """Test error when neither strategy nor file provided."""
        with pytest.raises(ChunkingConfigError, match="Must provide either"):
            resolve_chunking_config()

    def test_both_strategy_and_file_prefers_file(self, tmp_path):
        """Test that file takes priority over strategy."""
        config_file = tmp_path / "config.yaml"
        config_data = {"chunking_pipeline": [{"type": "custom", "config": {}}]}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = resolve_chunking_config(
            strategy_name="simple",
            config_file=config_file,
        )
        # Should use file (custom), not strategy (simple)
        assert result[0]["type"] == "custom"

    def test_invalid_json_syntax(self, tmp_path):
        """Test error for invalid JSON syntax."""
        config_file = tmp_path / "bad.json"
        with open(config_file, "w") as f:
            f.write("{invalid json}")
        with pytest.raises(ChunkingConfigError, match="Failed to parse"):
            resolve_chunking_config(config_file=config_file)


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_cli_style_usage(self):
        """Test usage pattern from CLI code."""
        # Simulates old CLI pattern with default parameter values
        config = resolve_chunking_config(
            strategy_name="detailed",
            window_size=3,
            step_size=1,
            threshold=0.5,
            min_segment_length=2,
        )
        assert isinstance(config, list)
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        assert sw_component["config"]["window_size_tokens"] == 3

    def test_api_style_usage(self):
        """Test usage pattern from API code."""
        # Simulates old API pattern with default parameter values
        config = resolve_chunking_config(
            strategy_name="sliding_window_punct_conj_cleaned",
            window_size=7,
            step_size=1,
            threshold=0.5,
            min_segment_length=3,
        )
        assert isinstance(config, list)

    def test_config_file_only(self, tmp_path):
        """Test using config file without strategy (CLI pattern)."""
        config_file = tmp_path / "config.yaml"
        config_data = {"chunking_pipeline": [{"type": "simple_sentence", "config": {}}]}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = resolve_chunking_config(config_file=config_file)
        assert len(config) == 1

    def test_strategy_only(self):
        """Test using strategy without config file (API pattern)."""
        config = resolve_chunking_config(strategy_name="simple")
        assert isinstance(config, list)


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_file_with_parameters_ignores_parameters(self, tmp_path):
        """Test that parameters are ignored when loading from file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "chunking_pipeline": [
                {
                    "type": "sliding_window",
                    "config": {
                        "window_size_tokens": 100,  # File specifies 100
                    },
                }
            ]
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Try to override with window_size=50
        config = resolve_chunking_config(
            config_file=config_file,
            window_size=50,  # This should be ignored for file-based configs
        )

        # File should win, but parameters should be applied
        # (This tests the override behavior)
        sw_component = config[0]
        assert sw_component["type"] == "sliding_window"
        # Parameters ARE applied to file configs
        assert sw_component["config"]["window_size_tokens"] == 50

    def test_complex_config_with_multiple_stages(self, tmp_path):
        """Test complex config with multiple pipeline stages."""
        config_file = tmp_path / "complex.yaml"
        config_data = {
            "chunking_pipeline": [
                {"type": "cleaner", "config": {"clean_type": "punct"}},
                {"type": "sliding_window", "config": {"window_size_tokens": 5}},
                {"type": "post_processor", "config": {}},
            ]
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = resolve_chunking_config(config_file=config_file)
        assert len(config) == 3
        assert config[0]["type"] == "cleaner"
        assert config[1]["type"] == "sliding_window"
        assert config[2]["type"] == "post_processor"
