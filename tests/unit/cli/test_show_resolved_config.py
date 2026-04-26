"""Tests for the --show-resolved-config debug flag.

Spec A's Observability section requires a global `--show-resolved-config`
flag that prints the resolved option set (with source labels from
`ctx.obj['resolved_sources']` plus user-passed flags labeled as
`<flag> (commandline)`) to **stderr** before the command body runs. The
command must still execute (it is not a dry-run).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner


def _clear_yaml_cache() -> None:
    from phentrieve.config import _load_yaml_config
    from phentrieve.utils import load_user_config

    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Isolate from any user / repo phentrieve.yaml during tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PHENTRIEVE_PROFILE", raising=False)
    _clear_yaml_cache()
    yield
    _clear_yaml_cache()


@pytest.fixture
def app():
    from phentrieve.cli import app as a

    return a


class TestShowResolvedConfig:
    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_output_to_stderr_not_stdout(self, mock_orchestrate, app):
        """The resolved-config table goes to stderr, not stdout."""
        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(app, ["--show-resolved-config", "query", "TEXT"])
        assert result.exit_code == 0, result.output
        # Header lands on stderr (separate from stdout in Click >= 8.2).
        assert "Resolved configuration" in result.stderr
        assert "Resolved configuration" not in result.stdout
        # Source-label arrows present.
        assert "<-" in result.stderr

    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_command_still_executes(self, mock_orchestrate, app):
        """The flag is observability-only - the command body still runs."""
        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(app, ["--show-resolved-config", "query", "TEXT"])
        assert result.exit_code == 0, result.output
        assert mock_orchestrate.called

    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_explicit_flag_labeled_commandline(self, mock_orchestrate, app):
        """A user-passed flag is labeled with `(commandline)` and the value."""
        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--show-resolved-config",
                "query",
                "TEXT",
                "--num-results",
                "7",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "(commandline)" in result.stderr
        # The numeric value is rendered in the table as well.
        assert "7" in result.stderr
        # And the user's value flowed through to the orchestrator.
        assert mock_orchestrate.call_args.kwargs["num_results"] == 7

    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_source_labels_match_resolved_sources(
        self, mock_orchestrate, tmp_path, monkeypatch, app
    ):
        """Source labels in the rendered table reflect the sidecar source map.

        With a profile that sets ``num_results: 5``, the rendered line for
        ``num_results`` must carry the ``profile:<name>`` label - matching
        ``ctx.obj['resolved_sources']['num_results']`` produced by the eager
        --profile callback.
        """
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n  show_cfg_test:\n    command: query\n    num_results: 5\n"
        )
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()

        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--show-resolved-config",
                "--profile",
                "show_cfg_test",
                "query",
                "TEXT",
            ],
        )
        assert result.exit_code == 0, result.output
        # Profile-sourced label format `profile:<name>` is in the table.
        assert "profile:show_cfg_test" in result.stderr
        # And the `num_results` row carries that label specifically.
        num_results_lines = [
            line for line in result.stderr.splitlines() if "num_results" in line
        ]
        assert num_results_lines, result.stderr
        assert any("profile:show_cfg_test" in line for line in num_results_lines)

    @patch("phentrieve.retrieval.query_orchestrator.orchestrate_query")
    def test_flag_off_means_no_output(self, mock_orchestrate, app):
        """Without the flag, the resolved-config table is not printed."""
        mock_orchestrate.return_value = []
        runner = CliRunner()
        result = runner.invoke(app, ["query", "TEXT"])
        assert result.exit_code == 0, result.output
        assert "Resolved configuration" not in result.stderr
        assert "Resolved configuration" not in result.stdout
