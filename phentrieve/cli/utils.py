"""Utility functions for the Phentrieve CLI.

This module contains shared utility functions used by the CLI commands.
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict

import typer


def load_text_from_input(text_arg: Optional[str], file_arg: Optional[Path]) -> str:
    """Load text from command line argument, file, or stdin.

    Args:
        text_arg: Text provided as a command line argument
        file_arg: Path to a file to read text from

    Returns:
        The loaded text content

    Raises:
        typer.Exit: If no text is provided or if the file does not exist
    """
    raw_text = None

    if text_arg is not None:
        raw_text = text_arg
    elif file_arg is not None:
        if not file_arg.exists():
            typer.secho(f"Error: File {file_arg} does not exist.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        with open(file_arg, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        # Read from stdin if available
        if not sys.stdin.isatty():
            raw_text = sys.stdin.read()
        else:
            typer.secho(
                "Error: No text provided. Please provide text as an argument, "
                "via --input-file, or through stdin.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    if not raw_text or not raw_text.strip():
        typer.secho("Error: Empty text provided.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    return raw_text


def resolve_chunking_pipeline_config(
    chunking_pipeline_config_file: Optional[Path],
    strategy_arg: str,
    window_size: int = 3,
    step_size: int = 1,
    threshold: float = 0.5,
    min_segment_length: int = 2,
) -> List[Dict]:
    """
    Resolve the chunking pipeline configuration from a file or a strategy name.

    Args:
        chunking_pipeline_config_file: Optional path to a config file
        strategy_arg: Strategy name to use if no config file is provided
        window_size: Window size for sliding window chunker (tokens)
        step_size: Step size for sliding window chunker (tokens)
        threshold: Similarity threshold for sliding window chunker
        min_segment_length: Minimum segment length for sliding window chunker (words)

    Returns:
        List of chunker configurations

    Raises:
        typer.Exit: If the config file does not exist or has an invalid
            format
    """
    from phentrieve.config import (
        get_default_chunk_pipeline_config,
        get_simple_chunking_config,
        get_detailed_chunking_config,
        get_semantic_chunking_config,
        get_sliding_window_config_with_params,
        get_sliding_window_cleaned_config,
        get_sliding_window_punct_cleaned_config,
        get_sliding_window_punct_conj_cleaned_config,
    )

    chunking_pipeline_config = None

    # 1. First priority: Config file if provided
    if chunking_pipeline_config_file is not None:
        if not chunking_pipeline_config_file.exists():
            typer.secho(
                f"Error: Config file {chunking_pipeline_config_file} does not exist.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        suffix = chunking_pipeline_config_file.suffix.lower()
        with open(chunking_pipeline_config_file, "r", encoding="utf-8") as f:
            if suffix == ".json":
                config_data = json.load(f)
            elif suffix in (".yaml", ".yml"):
                config_data = yaml.safe_load(f)
            else:
                typer.secho(
                    f"Error: Unsupported config file format: {suffix}."
                    " Use .json, .yaml, or .yml",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

        chunking_pipeline_config = config_data.get("chunking_pipeline", None)

    # 2. Second priority: Strategy parameter
    if chunking_pipeline_config is None:
        if strategy_arg == "simple":
            chunking_pipeline_config = get_simple_chunking_config()
        elif strategy_arg == "detailed":
            # Get the detailed config first
            chunking_pipeline_config = get_detailed_chunking_config()
            # Find and update the sliding_window component with the specified parameters
            for component in chunking_pipeline_config:
                if component.get("type") == "sliding_window":
                    component["config"] = {
                        "window_size_tokens": window_size,
                        "step_size_tokens": step_size,
                        "splitting_threshold": threshold,
                        "min_split_segment_length_words": min_segment_length,
                    }
        elif strategy_arg == "semantic":
            # Get the semantic config first
            chunking_pipeline_config = get_semantic_chunking_config()
            # Find and update the sliding_window component with the specified parameters
            for component in chunking_pipeline_config:
                if component.get("type") == "sliding_window":
                    component["config"] = {
                        "window_size_tokens": window_size,
                        "step_size_tokens": step_size,
                        "splitting_threshold": threshold,
                        "min_split_segment_length_words": min_segment_length,
                    }
        elif strategy_arg == "sliding_window":
            chunking_pipeline_config = get_sliding_window_config_with_params(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                min_segment_length=min_segment_length,
            )

        elif strategy_arg == "sliding_window_cleaned":
            # Get the cleaned config and update the sliding window parameters if provided
            chunking_pipeline_config = get_sliding_window_cleaned_config()
            # Find the sliding window config in the pipeline and update its parameters
            for stage in chunking_pipeline_config:
                if stage.get("type") == "sliding_window" and "config" in stage:
                    # Only update if the parameter was explicitly provided (not None)
                    if window_size is not None:
                        stage["config"]["window_size_tokens"] = window_size
                    if step_size is not None:
                        stage["config"]["step_size_tokens"] = step_size
                    if threshold is not None:
                        stage["config"]["splitting_threshold"] = threshold
                    if min_segment_length is not None:
                        stage["config"][
                            "min_split_segment_length_words"
                        ] = min_segment_length
                    break
        elif strategy_arg == "sliding_window_punct_cleaned":
            # Get the punctuation+cleaned config and update sliding window parameters if provided
            chunking_pipeline_config = get_sliding_window_punct_cleaned_config()
            # Find the sliding window config in the pipeline and update its parameters
            for stage_config_item in chunking_pipeline_config:
                if (
                    stage_config_item.get("type") == "sliding_window"
                    and "config" in stage_config_item
                ):
                    if window_size is not None:  # Check if CLI arg was provided
                        stage_config_item["config"]["window_size_tokens"] = window_size
                    if step_size is not None:
                        stage_config_item["config"]["step_size_tokens"] = step_size
                    if threshold is not None:
                        stage_config_item["config"]["splitting_threshold"] = threshold
                    if min_segment_length is not None:
                        stage_config_item["config"][
                            "min_split_segment_length_words"
                        ] = min_segment_length
                    break  # Found and updated the sliding_window component
        elif strategy_arg == "sliding_window_punct_conj_cleaned":
            # Get the punctuation+conjunction+cleaned config and update sliding window parameters
            chunking_pipeline_config = get_sliding_window_punct_conj_cleaned_config()
            # Find the sliding window config in the pipeline and update its parameters
            for stage_config_item in chunking_pipeline_config:
                if (
                    stage_config_item.get("type") == "sliding_window"
                    and "config" in stage_config_item
                ):
                    if window_size is not None:  # Check if CLI arg was provided
                        stage_config_item["config"]["window_size_tokens"] = window_size
                    if step_size is not None:
                        stage_config_item["config"]["step_size_tokens"] = step_size
                    if threshold is not None:
                        stage_config_item["config"]["splitting_threshold"] = threshold
                    if min_segment_length is not None:
                        stage_config_item["config"][
                            "min_split_segment_length_words"
                        ] = min_segment_length
                    break  # Found and updated the sliding_window component
        else:
            typer.secho(
                f"Warning: Unknown strategy '{strategy_arg}'. "
                f"Using default configuration.",
                fg=typer.colors.YELLOW,
            )

    # 3. Final fallback: Default configuration
    if chunking_pipeline_config is None:
        chunking_pipeline_config = get_default_chunk_pipeline_config()

    return chunking_pipeline_config
