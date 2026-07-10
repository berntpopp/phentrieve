"""Tests for the multi-model LLM benchmark helper."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_llm_model_benchmarks as runner  # noqa: E402


def test_load_models_combines_arguments_and_file(tmp_path: Path) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text(
        "\n".join(
            [
                "# OpenRouter models",
                "meta-llama/llama-3.1-70b-instruct",
                "",
                "google/gemini-3.1-flash-lite",
            ]
        ),
        encoding="utf-8",
    )

    models = runner.load_models(
        model_args=["anthropic/claude-sonnet-4-6"],
        models_file=models_file,
    )

    assert models == [
        "anthropic/claude-sonnet-4-6",
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemini-3.1-flash-lite",
    ]


def test_safe_model_slug_removes_path_separators() -> None:
    assert (
        runner.safe_model_slug("meta-llama/llama-3.1-70b-instruct")
        == "meta-llama__llama-3.1-70b-instruct"
    )


def test_build_command_passes_model_as_cli_parameter(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"

    command = runner.build_command(
        test_file=Path("tests/data/benchmarks/german/tiny_v1.json"),
        model="meta-llama/llama-3.1-70b-instruct",
        provider="openrouter",
        env_file=Path(".env"),
        output_dir=output_dir,
        extra_args=["--language", "en"],
    )

    assert command[:4] == ["uv", "run", "--env-file", ".env"]
    assert "--llm-provider" in command
    assert command[command.index("--llm-provider") + 1] == "openrouter"
    assert "--llm-model" in command
    assert (
        command[command.index("--llm-model") + 1] == "meta-llama/llama-3.1-70b-instruct"
    )
    assert command[command.index("--output-path") + 1] == str(
        output_dir / "meta-llama__llama-3.1-70b-instruct.json"
    )
    assert command[-2:] == ["--language", "en"]
