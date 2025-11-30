"""Shared configuration resolution logic for text processing pipelines.

This module provides a unified interface for resolving chunking configuration
from multiple sources (files, predefined strategies) with parameter overrides.
Consolidates logic previously duplicated between CLI and API.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from phentrieve.config import (
    get_detailed_chunking_config,
    get_semantic_chunking_config,
    get_simple_chunking_config,
    get_sliding_window_cleaned_config,
    get_sliding_window_config_with_params,
    get_sliding_window_punct_cleaned_config,
    get_sliding_window_punct_conj_cleaned_config,
)
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)


class ChunkingConfigError(Exception):
    """Raised when chunking configuration resolution fails."""

    pass


def resolve_chunking_config(
    strategy_name: str | None = None,
    config_file: Path | None = None,
    window_size: int | None = None,
    step_size: int | None = None,
    threshold: float | None = None,
    min_segment_length: int | None = None,
) -> list[dict[str, Any]]:
    """
    Resolve chunking configuration from multiple sources.

    Priority:
    1. Config file (if provided)
    2. Strategy name lookup
    3. Raise error if neither provided

    After resolving base config, applies parameter overrides if provided.

    Args:
        strategy_name: Name of predefined strategy (e.g., "simple", "detailed", "semantic")
        config_file: Path to YAML/JSON config file
        window_size: Override window size (tokens) for sliding window strategies
        step_size: Override step size (tokens) for sliding window strategies
        threshold: Override similarity threshold (0.0-1.0) for splitting
        min_segment_length: Override minimum segment length (words)

    Returns:
        List of chunking pipeline configuration dictionaries

    Raises:
        ChunkingConfigError: If neither strategy_name nor config_file provided,
                            or if config file doesn't exist or has invalid format

    Examples:
        >>> # Load from config file
        >>> config = resolve_chunking_config(config_file=Path("config.yaml"))

        >>> # Use predefined strategy
        >>> config = resolve_chunking_config(strategy_name="simple")

        >>> # Use strategy with parameter overrides
        >>> config = resolve_chunking_config(
        ...     strategy_name="detailed",
        ...     window_size=7,
        ...     threshold=0.6
        ... )
    """
    # Step 1: Resolve base configuration
    base_config = _resolve_base_config(strategy_name, config_file)

    # Step 2: Apply parameter overrides
    if any(
        param is not None
        for param in [window_size, step_size, threshold, min_segment_length]
    ):
        base_config = _apply_parameter_overrides(
            base_config,
            window_size=window_size,
            step_size=step_size,
            threshold=threshold,
            min_segment_length=min_segment_length,
        )

    return base_config


def _resolve_base_config(
    strategy_name: str | None,
    config_file: Path | None,
) -> list[dict[str, Any]]:
    """
    Resolve base configuration from file or strategy name.

    Args:
        strategy_name: Strategy name
        config_file: Config file path

    Returns:
        Base configuration list

    Raises:
        ChunkingConfigError: If config resolution fails
    """
    # Priority 1: Config file
    if config_file is not None:
        return _load_config_from_file(config_file)

    # Priority 2: Strategy name
    if strategy_name is not None:
        return _get_strategy_config(strategy_name)

    # Error: Neither provided
    raise ChunkingConfigError(
        "Must provide either strategy_name or config_file for chunking configuration"
    )


def _load_config_from_file(config_file: Path) -> list[dict[str, Any]]:
    """
    Load chunking configuration from YAML or JSON file.

    Args:
        config_file: Path to config file

    Returns:
        Configuration list from file

    Raises:
        ChunkingConfigError: If file doesn't exist or has invalid format
    """
    if not config_file.exists():
        raise ChunkingConfigError(
            f"Config file not found: {_sanitize(str(config_file))}"
        )

    suffix = config_file.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ChunkingConfigError(
            f"Unsupported config file format: {_sanitize(suffix)}. Use .json, .yaml, or .yml"
        )

    try:
        with open(config_file, encoding="utf-8") as f:
            if suffix == ".json":
                config_data = json.load(f)
            else:  # .yaml or .yml
                config_data = yaml.safe_load(f)

        chunking_pipeline = config_data.get("chunking_pipeline")
        if chunking_pipeline is None:
            raise ChunkingConfigError(
                f"Config file missing 'chunking_pipeline' key: {_sanitize(str(config_file))}"
            )

        # Type checked: we know this is a list of dicts from the config structure
        return list(chunking_pipeline)

    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ChunkingConfigError(
            f"Failed to parse config file {_sanitize(str(config_file))}: {_sanitize(str(e))}"
        )


def _get_strategy_config(strategy_name: str) -> list[dict[str, Any]]:
    """
    Get predefined chunking strategy configuration by name.

    Args:
        strategy_name: Strategy name (e.g., "simple", "detailed", "semantic")

    Returns:
        Strategy configuration list

    Note:
        Returns default config for unknown strategy names (with warning).
    """
    strategy_name = strategy_name.lower()

    # Map strategy names to config functions
    strategy_map: dict[str, Callable[[], list[dict[str, Any]]]] = {
        "simple": get_simple_chunking_config,
        "detailed": get_detailed_chunking_config,
        "semantic": get_semantic_chunking_config,
        "sliding_window": get_sliding_window_config_with_params,
        "sliding_window_cleaned": get_sliding_window_cleaned_config,
        "sliding_window_punct_cleaned": get_sliding_window_punct_cleaned_config,
        "sliding_window_punct_conj_cleaned": get_sliding_window_punct_conj_cleaned_config,
    }

    if strategy_name in strategy_map:
        config_func = strategy_map[strategy_name]
        return list(config_func())
    else:
        logger.warning(
            "Unknown strategy '%s', using sliding_window_punct_conj_cleaned",
            _sanitize(strategy_name),
        )
        return list(get_sliding_window_punct_conj_cleaned_config())


def _apply_parameter_overrides(
    config: list[dict[str, Any]],
    window_size: int | None = None,
    step_size: int | None = None,
    threshold: float | None = None,
    min_segment_length: int | None = None,
) -> list[dict[str, Any]]:
    """
    Apply parameter overrides to sliding window components in config.

    Modifies config in-place by updating parameters for any components
    with type='sliding_window'.

    Args:
        config: Chunking pipeline configuration (modified in-place)
        window_size: Window size in tokens (if provided)
        step_size: Step size in tokens (if provided)
        threshold: Similarity threshold for splitting (if provided)
        min_segment_length: Minimum segment length in words (if provided)

    Returns:
        Modified configuration (same object as input)
    """
    # Build override dict with only provided parameters
    overrides: dict[str, int | float] = {}
    if window_size is not None:
        overrides["window_size_tokens"] = window_size
    if step_size is not None:
        overrides["step_size_tokens"] = step_size
    if threshold is not None:
        overrides["splitting_threshold"] = threshold
    if min_segment_length is not None:
        overrides["min_split_segment_length_words"] = min_segment_length

    # Apply overrides to all sliding_window components
    if overrides:
        for component in config:
            if component.get("type") == "sliding_window" and "config" in component:
                component["config"].update(overrides)

    return config
