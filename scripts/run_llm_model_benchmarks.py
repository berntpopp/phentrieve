#!/usr/bin/env python3
"""Run the LLM benchmark for multiple provider/model targets."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def safe_model_slug(model: str) -> str:
    return (
        model.strip()
        .replace("\\", "__")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def load_models(
    *,
    model_args: list[str] | None = None,
    models_file: Path | None = None,
) -> list[str]:
    models: list[str] = []
    for model in model_args or []:
        normalized = model.strip()
        if normalized:
            models.append(normalized)

    if models_file is not None:
        for line in models_file.read_text(encoding="utf-8").splitlines():
            normalized = line.strip()
            if normalized and not normalized.startswith("#"):
                models.append(normalized)

    return models


def build_command(
    *,
    test_file: Path,
    model: str,
    provider: str,
    env_file: Path | None,
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> list[str]:
    model_slug = safe_model_slug(model)
    command = ["uv", "run"]
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    command.extend(
        [
            "phentrieve",
            "benchmark",
            "llm",
            "--test-file",
            str(test_file),
            "--llm-provider",
            provider,
            "--llm-model",
            model,
            "--output-path",
            str(output_dir / f"{model_slug}.json"),
            "--checkpoint-path",
            str(output_dir / f"{model_slug}.checkpoint.json"),
            "--artifacts-dir",
            str(output_dir / model_slug),
        ]
    )
    command.extend(extra_args or [])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phentrieve benchmark llm for multiple models."
    )
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument(
        "--provider",
        default="openrouter",
        help="LLM provider passed to phentrieve benchmark llm.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model id to benchmark. Repeat for multiple models.",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        help="Text file with one model id per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Env file containing provider API keys. Use --no-env-file to disable.",
    )
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        help="Do not pass an env file to uv run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/llm-model-benchmarks"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print benchmark commands without executing them.",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed after -- to phentrieve benchmark llm.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra_args = list(args.benchmark_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    models = load_models(model_args=args.model, models_file=args.models_file)
    if not models:
        raise SystemExit("No models configured. Pass --model or --models-file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_file = None if args.no_env_file else args.env_file

    for model in models:
        command = build_command(
            test_file=args.test_file,
            model=model,
            provider=args.provider,
            env_file=env_file,
            output_dir=args.output_dir,
            extra_args=extra_args,
        )
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)  # noqa: S603 - no shell invocation.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
