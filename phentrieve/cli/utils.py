"""Utility functions for the Phentrieve CLI.

This module contains shared utility functions used by the CLI commands.
"""

import sys
from pathlib import Path
from typing import Optional

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
        with open(file_arg, encoding="utf-8") as f:
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
) -> list[dict]:
    """
    Resolve the chunking pipeline configuration from a file or a strategy name.

    This is a CLI-specific wrapper around the shared config resolver that converts
    ChunkingConfigError exceptions to typer.Exit for CLI error handling.

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
        typer.Exit: If the config file does not exist or has an invalid format
    """
    from phentrieve.text_processing.config_resolver import (
        ChunkingConfigError,
        resolve_chunking_config,
    )

    try:
        return resolve_chunking_config(
            strategy_name=strategy_arg,
            config_file=chunking_pipeline_config_file,
            window_size=window_size,
            step_size=step_size,
            threshold=threshold,
            min_segment_length=min_segment_length,
        )
    except ChunkingConfigError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
