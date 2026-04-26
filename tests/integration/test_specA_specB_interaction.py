"""Cross-spec test: Spec A (CLI profiles) + Spec B (adaptive_rechunking).

Verifies that:

1. A profile setting ``adaptive_rechunking.{enabled,quality_threshold}``
   in ``phentrieve.yaml`` is propagated to the resolved
   ``AdaptiveRechunkingConfig`` passed to ``run_full_text_service``.
2. Explicit ``--adaptive-rechunking-quality-threshold`` on the command
   line overrides the profile value while leaving other profile-supplied
   knobs (here, ``enabled=True``) intact.

The full-text service itself is patched out; this test exercises only
the resolution chain and CLI plumbing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config
from phentrieve.utils import load_user_config

pytestmark = pytest.mark.integration


def _clear_yaml_cache() -> None:
    """Clear the LRU caches that back YAML / user-config loading so each
    test sees its own ``phentrieve.yaml`` written in ``tmp_path``.
    """
    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


@pytest.fixture(autouse=True)
def _yaml(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Write a phentrieve.yaml in CWD that defines a profile carrying
    an ``adaptive_rechunking`` block, and clear the cached config so the
    test sees the freshly written file.
    """
    (tmp_path / "phentrieve.yaml").write_text(
        "profiles:\n"
        "  german_recall:\n"
        "    command: text process\n"
        "    language: de\n"
        "    adaptive_rechunking:\n"
        "      enabled: true\n"
        "      quality_threshold: 0.6\n"
    )
    monkeypatch.chdir(tmp_path)
    _clear_yaml_cache()
    yield
    _clear_yaml_cache()


@pytest.fixture
def app() -> Any:
    """The top-level Typer app, imported lazily to avoid invoking it at
    module import time (``phentrieve/__main__.py`` calls ``app()`` on
    import).
    """
    from phentrieve.cli import app as a

    return a


@patch("phentrieve.cli.text_commands.run_full_text_service")
def test_profile_provides_adaptive_config(
    mock_run: Any, app: Any, tmp_path: Any
) -> None:
    """A profile-supplied ``adaptive_rechunking`` block flows through to
    the resolved ``AdaptiveRechunkingConfig`` even with no CLI flags.
    """
    mock_run.return_value = {
        "meta": {"extraction_backend": "standard"},
        "processed_chunks": [],
        "aggregated_hpo_terms": [],
    }
    input_file = tmp_path / "in.txt"
    input_file.write_text("Patient.")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "-i",
            str(input_file),
            "--profile",
            "german_recall",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
    assert cfg.enabled is True
    assert cfg.quality_threshold == 0.6


@patch("phentrieve.cli.text_commands.run_full_text_service")
def test_explicit_flag_overrides_profile_adaptive(
    mock_run: Any, app: Any, tmp_path: Any
) -> None:
    """Explicit ``--adaptive-rechunking-quality-threshold`` overrides the
    profile-supplied threshold; other profile values pass through.
    """
    mock_run.return_value = {
        "meta": {"extraction_backend": "standard"},
        "processed_chunks": [],
        "aggregated_hpo_terms": [],
    }
    input_file = tmp_path / "in.txt"
    input_file.write_text("Patient.")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "-i",
            str(input_file),
            "--profile",
            "german_recall",
            "--adaptive-rechunking-quality-threshold",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
    assert cfg.quality_threshold == 0.5  # CLI wins
    assert cfg.enabled is True  # from profile
