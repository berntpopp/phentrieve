"""Tests that --profile works at both root and per-command placement.

Spec A Phase 7: ``phentrieve --profile X query ...`` and
``phentrieve query --profile X ...`` must resolve identically. With both
placements, the subcommand-level value wins. With ``PHENTRIEVE_PROFILE`` set
in the environment, an explicit ``--profile`` flag still wins.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from phentrieve.cli import app


def _clear_yaml_cache() -> None:
    from phentrieve.config import _load_yaml_config
    from phentrieve.utils import load_user_config

    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


@pytest.fixture(autouse=True)
def _yaml_with_two_profiles(tmp_path, monkeypatch):
    (tmp_path / "phentrieve.yaml").write_text(
        "profiles:\n"
        "  fast_query_a:\n"
        "    command: query\n"
        "    num_results: 5\n"
        "  fast_query_b:\n"
        "    command: query\n"
        "    num_results: 8\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PHENTRIEVE_PROFILE", raising=False)
    _clear_yaml_cache()
    yield
    _clear_yaml_cache()


class TestProfilePlacement:
    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_root_placement(self, mock_orchestrate):
        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(app, ["--profile", "fast_query_a", "query", "TEXT"])
        assert result.exit_code == 0, result.output
        assert mock_orchestrate.called
        assert mock_orchestrate.call_args.kwargs["num_results"] == 5

    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_subcommand_placement(self, mock_orchestrate):
        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(app, ["query", "TEXT", "--profile", "fast_query_a"])
        assert result.exit_code == 0, result.output
        assert mock_orchestrate.called
        assert mock_orchestrate.call_args.kwargs["num_results"] == 5

    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_subcommand_wins_over_root(self, mock_orchestrate):
        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--profile",
                "fast_query_a",
                "query",
                "TEXT",
                "--profile",
                "fast_query_b",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_orchestrate.called
        # b wins (subcommand-level beats root-level).
        assert mock_orchestrate.call_args.kwargs["num_results"] == 8

    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_explicit_flag_beats_both(self, mock_orchestrate):
        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "query",
                "TEXT",
                "--profile",
                "fast_query_a",
                "--num-results",
                "20",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_orchestrate.called
        assert mock_orchestrate.call_args.kwargs["num_results"] == 20

    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_envvar_overridden_by_explicit_flag(self, mock_orchestrate, monkeypatch):
        """PHENTRIEVE_PROFILE supplies a value, but explicit --profile wins."""
        mock_orchestrate.return_value = []
        monkeypatch.setenv("PHENTRIEVE_PROFILE", "fast_query_a")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["query", "TEXT", "--profile", "fast_query_b"],
        )
        assert result.exit_code == 0, result.output
        assert mock_orchestrate.called
        assert mock_orchestrate.call_args.kwargs["num_results"] == 8
