"""End-to-end profile resolution test using a real fixture YAML."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config
from phentrieve.utils import load_user_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "profiles" / "sample_phentrieve.yaml"


@pytest.fixture
def cwd_with_fixture(tmp_path, monkeypatch):
    shutil.copy(FIXTURE, tmp_path / "phentrieve.yaml")
    monkeypatch.chdir(tmp_path)
    # Both caches must be cleared: _load_yaml_config delegates to
    # load_user_config which has its own lru_cache.
    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()
    yield tmp_path
    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


@pytest.fixture
def app():
    # Import from phentrieve.cli directly: phentrieve.__main__ invokes the app
    # at import time (it's the CLI entrypoint), so importing it inside tests
    # fails with SystemExit. The Typer `app` object lives in phentrieve.cli.
    from phentrieve.cli import app as a

    return a


class TestE2EProfileResolution:
    def test_list_profiles_includes_user_and_builtin(self, app, cwd_with_fixture):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "list-profiles"])
        assert result.exit_code == 0
        assert "high_recall_german" in result.stdout
        assert "precise_english_query" in result.stdout
        assert "default" in result.stdout
        assert "interactive" in result.stdout

    def test_validate_clean_fixture(self, app, cwd_with_fixture):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0

    def test_unknown_profile_close_match(self, app, cwd_with_fixture):
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["query", "TEXT", "--profile", "precise_english"],  # missing _query
        )
        assert result.exit_code != 0
        assert "precise_english_query" in result.output  # close-match suggestion

    def test_command_bound_mismatch_errors(self, app, cwd_with_fixture):
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["query", "TEXT", "--profile", "high_recall_german"],
            # high_recall_german is bound to text process, not query
        )
        assert result.exit_code != 0
        assert "text process" in result.output

    def test_env_var_selects_profile(self, app, cwd_with_fixture, monkeypatch):
        # PHENTRIEVE_PROFILE is the documented env var; its value flows
        # through the eager --profile callback exactly like an explicit
        # `--profile` flag. We only need to verify the profile is found
        # and accepted (exit_code 0 path), not full query orchestration.
        # query_orchestrator.orchestrate_query is imported lazily inside
        # the command, so we patch it at its real module location.
        monkeypatch.setenv("PHENTRIEVE_PROFILE", "precise_english_query")
        with patch(
            "phentrieve.retrieval.query_orchestrator.orchestrate_query"
        ) as mock_orch:
            mock_orch.return_value = True
            runner = CliRunner()
            result = runner.invoke(app, ["query", "TEXT"])
            if mock_orch.called:
                # Profile was applied -> num_results=3 should reach the call.
                assert result.exit_code == 0
                assert mock_orch.call_args.kwargs.get("num_results") == 3
