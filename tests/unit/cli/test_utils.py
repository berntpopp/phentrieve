"""Unit tests for CLI utility functions.

Tests for shared utilities used across CLI commands:
- load_text_from_input: Loading text from various sources
- resolve_chunking_pipeline_config: Config resolution logic

Following best practices:
- Pure function testing (no CLI runner needed)
- Mock file I/O and external dependencies
- Comprehensive edge case coverage
- Clear Arrange-Act-Assert structure
"""

import json
from pathlib import Path

import pytest
import typer
import yaml

from phentrieve.cli.utils import (
    load_text_from_input,
    resolve_chunking_pipeline_config,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for load_text_from_input()
# =============================================================================


class TestLoadTextFromInput:
    """Test load_text_from_input() function."""

    def test_loads_text_from_cli_argument(self):
        """Test loading text from command line argument."""
        # Arrange
        text_arg = "patient has seizures"
        file_arg = None

        # Act
        result = load_text_from_input(text_arg, file_arg)

        # Assert
        assert result == "patient has seizures"

    def test_loads_text_from_file(self, tmp_path):
        """Test loading text from a file."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_file.write_text("text from file", encoding="utf-8")

        # Act
        result = load_text_from_input(None, test_file)

        # Assert
        assert result == "text from file"

    def test_loads_text_from_stdin_when_available(self, mocker):
        """Test loading text from stdin when no CLI arg or file provided."""
        # Arrange
        mock_stdin = mocker.patch("sys.stdin")
        mock_stdin.isatty.return_value = False  # stdin has data
        mock_stdin.read.return_value = "text from stdin"

        # Act
        result = load_text_from_input(None, None)

        # Assert
        assert result == "text from stdin"
        mock_stdin.read.assert_called_once()

    def test_prioritizes_cli_arg_over_file(self, tmp_path):
        """Test CLI argument takes precedence over file."""
        # Arrange
        text_arg = "cli text"
        test_file = tmp_path / "test.txt"
        test_file.write_text("file text", encoding="utf-8")

        # Act
        result = load_text_from_input(text_arg, test_file)

        # Assert
        assert result == "cli text"  # CLI arg wins

    def test_prioritizes_cli_arg_over_stdin(self, mocker):
        """Test CLI argument takes precedence over stdin."""
        # Arrange
        text_arg = "cli text"
        mock_stdin = mocker.patch("sys.stdin")
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "stdin text"

        # Act
        result = load_text_from_input(text_arg, None)

        # Assert
        assert result == "cli text"  # CLI arg wins
        mock_stdin.read.assert_not_called()  # stdin not even checked

    def test_prioritizes_file_over_stdin(self, tmp_path, mocker):
        """Test file takes precedence over stdin."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_file.write_text("file text", encoding="utf-8")
        mock_stdin = mocker.patch("sys.stdin")
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "stdin text"

        # Act
        result = load_text_from_input(None, test_file)

        # Assert
        assert result == "file text"  # File wins
        mock_stdin.read.assert_not_called()

    def test_raises_exit_when_file_does_not_exist(self):
        """Test raises typer.Exit when file does not exist."""
        # Arrange
        nonexistent_file = Path("/nonexistent/file.txt")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            load_text_from_input(None, nonexistent_file)

        assert exc_info.value.exit_code == 1

    def test_raises_exit_when_no_input_provided(self, mocker):
        """Test raises typer.Exit when no input source available."""
        # Arrange
        mock_stdin = mocker.patch("sys.stdin")
        mock_stdin.isatty.return_value = True  # No stdin available

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            load_text_from_input(None, None)

        assert exc_info.value.exit_code == 1

    def test_raises_exit_when_text_is_empty(self):
        """Test raises typer.Exit when provided text is empty."""
        # Arrange
        text_arg = ""

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            load_text_from_input(text_arg, None)

        assert exc_info.value.exit_code == 1

    def test_raises_exit_when_text_is_only_whitespace(self):
        """Test raises typer.Exit when text is only whitespace."""
        # Arrange
        text_arg = "   \n\t  "

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            load_text_from_input(text_arg, None)

        assert exc_info.value.exit_code == 1

    def test_strips_and_validates_text_from_file(self, tmp_path):
        """Test validates empty text from file."""
        # Arrange
        test_file = tmp_path / "empty.txt"
        test_file.write_text("   \n  ", encoding="utf-8")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            load_text_from_input(None, test_file)

        assert exc_info.value.exit_code == 1


# =============================================================================
# Tests for resolve_chunking_pipeline_config()
# =============================================================================


class TestResolveChunkingPipelineConfig:
    """Test resolve_chunking_pipeline_config() function."""

    def test_loads_config_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        # Arrange
        config_data = {
            "chunking_pipeline": [
                {"type": "sentence", "config": {}},
                {"type": "sliding_window", "config": {"window_size_tokens": 3}},
            ]
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=config_file,
            strategy_arg="simple",  # Should be ignored
        )

        # Assert
        assert len(result) == 2
        assert result[0]["type"] == "sentence"
        assert result[1]["type"] == "sliding_window"

    def test_loads_config_from_yaml_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        # Arrange
        config_data = {
            "chunking_pipeline": [
                {"type": "sentence", "config": {}},
            ]
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=config_file,
            strategy_arg="simple",
        )

        # Assert
        assert len(result) == 1
        assert result[0]["type"] == "sentence"

    def test_loads_config_from_yml_extension(self, tmp_path):
        """Test loading configuration from .yml file."""
        # Arrange
        config_data = {"chunking_pipeline": [{"type": "sentence"}]}
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=config_file,
            strategy_arg="simple",
        )

        # Assert
        assert len(result) >= 1

    def test_raises_exit_when_config_file_not_found(self):
        """Test raises typer.Exit when config file does not exist."""
        # Arrange
        nonexistent_file = Path("/nonexistent/config.json")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            resolve_chunking_pipeline_config(
                chunking_pipeline_config_file=nonexistent_file,
                strategy_arg="simple",
            )

        assert exc_info.value.exit_code == 1

    def test_raises_exit_on_unsupported_file_format(self, tmp_path):
        """Test raises typer.Exit for unsupported config file format."""
        # Arrange
        config_file = tmp_path / "config.txt"  # Invalid extension
        config_file.write_text("some data", encoding="utf-8")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            resolve_chunking_pipeline_config(
                chunking_pipeline_config_file=config_file,
                strategy_arg="simple",
            )

        assert exc_info.value.exit_code == 1

    def test_uses_simple_strategy_when_no_config_file(self, mocker):
        """Test uses simple strategy when no config file provided."""
        # Arrange
        mock_get_simple = mocker.patch(
            "phentrieve.text_processing.config_resolver.get_simple_chunking_config",
            return_value=[{"type": "sentence"}],
        )

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=None,
            strategy_arg="simple",
        )

        # Assert
        mock_get_simple.assert_called_once()
        assert result[0]["type"] == "sentence"

    def test_uses_detailed_strategy_with_custom_params(self, mocker):
        """Test detailed strategy with custom sliding window parameters."""
        # Arrange
        mock_get_detailed = mocker.patch(
            "phentrieve.text_processing.config_resolver.get_detailed_chunking_config",
            return_value=[
                {"type": "sentence"},
                {"type": "sliding_window", "config": {}},
            ],
        )

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=None,
            strategy_arg="detailed",
            window_size=5,
            step_size=2,
            threshold=0.7,
            min_segment_length=3,
        )

        # Assert
        mock_get_detailed.assert_called_once()
        sliding_window_config = result[1]["config"]
        assert sliding_window_config["window_size_tokens"] == 5
        assert sliding_window_config["step_size_tokens"] == 2
        assert sliding_window_config["splitting_threshold"] == 0.7
        assert sliding_window_config["min_split_segment_length_words"] == 3

    def test_uses_semantic_strategy_with_custom_params(self, mocker):
        """Test semantic strategy with custom sliding window parameters."""
        # Arrange
        mock_get_semantic = mocker.patch(
            "phentrieve.text_processing.config_resolver.get_semantic_chunking_config",
            return_value=[
                {"type": "semantic"},
                {"type": "sliding_window", "config": {}},
            ],
        )

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=None,
            strategy_arg="semantic",
            window_size=4,
            step_size=1,
        )

        # Assert
        mock_get_semantic.assert_called_once()
        assert result[1]["config"]["window_size_tokens"] == 4

    def test_uses_sliding_window_strategy_with_params(self, mocker):
        """Test sliding_window strategy with custom parameters."""
        # Arrange
        mock_get_sliding = mocker.patch(
            "phentrieve.text_processing.config_resolver.get_sliding_window_config_with_params",
            return_value=[{"type": "sliding_window", "config": {}}],
        )

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=None,
            strategy_arg="sliding_window",
            window_size=6,
            step_size=3,
            threshold=0.6,
            min_segment_length=5,
        )

        # Assert - function called once (parameters applied via override mechanism)
        mock_get_sliding.assert_called_once()
        # Verify parameters were applied
        config = result[0]["config"]
        assert config["window_size_tokens"] == 6
        assert config["step_size_tokens"] == 3
        assert config["splitting_threshold"] == 0.6
        assert config["min_split_segment_length_words"] == 5

    @pytest.mark.parametrize(
        "strategy",
        [
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_updates_sliding_window_variants_with_params(self, strategy, mocker):
        """Test updating sliding window variants with custom parameters."""
        # Arrange
        mock_getter = mocker.patch(
            f"phentrieve.text_processing.config_resolver.get_{strategy}_config",
            return_value=[{"type": "sliding_window", "config": {}}],
        )

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=None,
            strategy_arg=strategy,
            window_size=7,
            step_size=2,
            threshold=0.8,
            min_segment_length=4,
        )

        # Assert
        mock_getter.assert_called_once()
        config = result[0]["config"]
        assert config["window_size_tokens"] == 7
        assert config["step_size_tokens"] == 2
        assert config["splitting_threshold"] == 0.8
        assert config["min_split_segment_length_words"] == 4

    def test_falls_back_to_default_on_unknown_strategy(self, mocker, caplog):
        """Test falls back to default config for unknown strategy."""
        # Arrange - mock the fallback config function
        mock_get_fallback = mocker.patch(
            "phentrieve.text_processing.config_resolver.get_sliding_window_punct_conj_cleaned_config",
            return_value=[{"type": "fallback"}],
        )

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=None,
            strategy_arg="unknown_strategy",
        )

        # Assert - should call fallback and log warning
        mock_get_fallback.assert_called_once()
        assert result[0]["type"] == "fallback"
        assert any("Unknown strategy" in record.message for record in caplog.records)

    def test_config_file_takes_precedence_over_strategy(self, tmp_path, mocker):
        """Test config file takes precedence over strategy argument."""
        # Arrange
        config_data = {"chunking_pipeline": [{"type": "from_file"}]}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        mock_get_simple = mocker.patch(
            "phentrieve.text_processing.config_resolver.get_simple_chunking_config"
        )

        # Act
        result = resolve_chunking_pipeline_config(
            chunking_pipeline_config_file=config_file,
            strategy_arg="simple",  # Should be ignored
        )

        # Assert
        mock_get_simple.assert_not_called()  # Strategy not used
        assert result[0]["type"] == "from_file"
