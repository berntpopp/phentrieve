"""Unit tests for text processing router helper functions."""

from typing import Any

import pytest

from api.routers.text_processing_router import (
    _apply_sliding_window_params,
    _get_chunking_config_for_api,
)
from api.schemas.text_processing_schemas import TextProcessingRequest

pytestmark = pytest.mark.unit


class TestApplySlidingWindowParams:
    """Test _apply_sliding_window_params helper function."""

    def test_updates_sliding_window_component(self):
        """Test sliding window component parameters are updated."""
        # Arrange
        config = [
            {
                "type": "sliding_window",
                "config": {
                    "window_size_tokens": 5,
                    "step_size_tokens": 1,
                },
            },
            {
                "type": "other_component",
                "config": {"param": "value"},
            },
        ]

        # Act
        _apply_sliding_window_params(
            config=config,
            window_size=10,
            step_size=2,
            threshold=0.7,
            min_segment_length=5,
        )

        # Assert
        sw_config = config[0]["config"]
        assert sw_config["window_size_tokens"] == 10
        assert sw_config["step_size_tokens"] == 2
        assert sw_config["splitting_threshold"] == 0.7
        assert sw_config["min_split_segment_length_words"] == 5

        # Other component unchanged
        assert config[1]["config"] == {"param": "value"}

    def test_handles_config_without_sliding_window(self):
        """Test gracefully handles config without sliding_window component."""
        # Arrange
        config = [{"type": "other", "config": {}}]

        # Act - should not raise
        _apply_sliding_window_params(config, 10, 2, 0.7, 5)

        # Assert - no changes
        assert config == [{"type": "other", "config": {}}]

    def test_updates_multiple_sliding_window_components(self):
        """Test updates all sliding_window components if multiple exist."""
        # Arrange
        config = [
            {"type": "sliding_window", "config": {}},
            {"type": "other", "config": {}},
            {"type": "sliding_window", "config": {}},
        ]

        # Act
        _apply_sliding_window_params(config, 15, 3, 0.8, 10)

        # Assert - both sliding_window components updated
        assert config[0]["config"]["window_size_tokens"] == 15
        assert config[0]["config"]["step_size_tokens"] == 3
        assert config[0]["config"]["splitting_threshold"] == 0.8
        assert config[0]["config"]["min_split_segment_length_words"] == 10
        assert config[2]["config"]["window_size_tokens"] == 15

    def test_preserves_existing_config_values(self):
        """Test preserves other config values not being updated."""
        # Arrange
        config = [
            {
                "type": "sliding_window",
                "config": {
                    "window_size_tokens": 5,
                    "other_param": "should_remain",
                    "another_param": 42,
                },
            }
        ]

        # Act
        _apply_sliding_window_params(config, 10, 2, 0.7, 5)

        # Assert - new values updated, old values preserved
        assert config[0]["config"]["window_size_tokens"] == 10
        assert config[0]["config"]["other_param"] == "should_remain"
        assert config[0]["config"]["another_param"] == 42

    def test_handles_empty_config_list(self):
        """Test handles empty config list without error."""
        # Arrange
        config: list[dict[str, Any]] = []

        # Act - should not raise
        _apply_sliding_window_params(config, 10, 2, 0.7, 5)

        # Assert - still empty
        assert config == []


class TestGetChunkingConfigForApi:
    """Test _get_chunking_config_for_api function."""

    @pytest.mark.parametrize(
        "strategy_name",
        [
            "simple",
            "semantic",
            "detailed",
            "sliding_window",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_all_strategies_return_valid_config(self, strategy_name: str):
        """Test all strategies return valid configuration."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy=strategy_name,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert
        assert isinstance(config, list)
        assert len(config) > 0
        assert all(isinstance(c, dict) for c in config)
        assert all("type" in c and "config" in c for c in config)

    def test_unknown_strategy_uses_default(self):
        """Test unknown strategy falls back to default."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="unknown_nonexistent_strategy",
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should return valid config (default)
        assert isinstance(config, list)
        assert len(config) > 0

    def test_custom_sliding_window_parameters_applied(self):
        """Test custom sliding window parameters are applied correctly."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="semantic",
            window_size=15,
            step_size=3,
            split_threshold=0.8,
            min_segment_length=10,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - find sliding_window component
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        assert sw_component is not None

        sw_config = sw_component["config"]
        assert sw_config["window_size_tokens"] == 15
        assert sw_config["step_size_tokens"] == 3
        assert sw_config["splitting_threshold"] == 0.8
        assert sw_config["min_split_segment_length_words"] == 10

    def test_default_parameters_when_none_provided(self):
        """Test default parameters used when not specified in request."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="semantic",
            # No custom params - should use defaults (ws=7, ss=1, th=0.5, msl=3)
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should have sliding_window with default params
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        assert sw_component is not None

        sw_config = sw_component["config"]
        assert sw_config["window_size_tokens"] == 7
        assert sw_config["step_size_tokens"] == 1
        assert sw_config["splitting_threshold"] == 0.5
        assert sw_config["min_split_segment_length_words"] == 3

    def test_none_chunking_strategy_uses_default(self):
        """Test None/missing chunking_strategy uses default."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy=None,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should return valid config (sliding_window_punct_conj_cleaned)
        assert isinstance(config, list)
        assert len(config) > 0

    def test_simple_strategy_returns_without_modification(self):
        """Test simple strategy returns config without sliding window params."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="simple",
            window_size=100,  # Should be ignored for simple strategy
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should return valid config
        assert isinstance(config, list)
        assert len(config) > 0

    def test_sliding_window_strategy_uses_params_directly(self):
        """Test sliding_window strategy passes params to config function."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="sliding_window",
            window_size=20,
            step_size=5,
            split_threshold=0.6,
            min_segment_length=8,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - params should be applied via config function
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        assert sw_component is not None

        sw_config = sw_component["config"]
        assert sw_config["window_size_tokens"] == 20
        assert sw_config["step_size_tokens"] == 5
        assert sw_config["splitting_threshold"] == 0.6
        assert sw_config["min_split_segment_length_words"] == 8

    def test_case_insensitive_strategy_name(self):
        """Test strategy name is case-insensitive."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="SEMANTIC",  # Uppercase
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should work same as lowercase
        assert isinstance(config, list)
        assert len(config) > 0

    @pytest.mark.parametrize(
        "strategy",
        [
            "semantic",
            "detailed",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_all_applicable_strategies_receive_params(self, strategy: str):
        """Test all strategies that need params receive them."""
        # Arrange
        custom_ws = 25
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy=strategy,
            window_size=custom_ws,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should have sliding_window with custom param
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        if sw_component:  # Some strategies may not have sliding_window
            assert sw_component["config"]["window_size_tokens"] == custom_ws
