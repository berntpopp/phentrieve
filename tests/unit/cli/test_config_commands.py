"""Tests for `phentrieve config list-profiles / show / validate / path`."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config
from phentrieve.utils import load_user_config


@pytest.fixture(autouse=True)
def _yaml_setup(tmp_path, monkeypatch):
    """Create a temporary phentrieve.yaml with one user profile and switch CWD."""
    (tmp_path / "phentrieve.yaml").write_text(
        "profiles:\n"
        "  fast_query:\n"
        "    description: 'Quick English query'\n"
        "    command: query\n"
        "    num_results: 5\n"
    )
    monkeypatch.chdir(tmp_path)
    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()
    yield
    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


@pytest.fixture
def app():
    from phentrieve.cli import app as a

    return a


class TestConfigListProfiles:
    def test_lists_user_and_builtin(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "list-profiles"])
        assert result.exit_code == 0, result.output
        assert "fast_query" in result.stdout
        assert "default" in result.stdout
        assert "interactive" in result.stdout

    def test_shows_command_binding(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "list-profiles"])
        assert result.exit_code == 0, result.output
        # fast_query is bound to query; the binding column should show "query".
        assert "query" in result.stdout

    def test_shadowing_marks_user_source(self, app, tmp_path):
        # Override the YAML to shadow the built-in `interactive` profile.
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  interactive:\n"
            "    description: 'Shadowed interactive'\n"
            "    num_results: 7\n"
        )
        _load_yaml_config.cache_clear()
        load_user_config.cache_clear()
        runner = CliRunner()
        result = runner.invoke(app, ["config", "list-profiles"])
        assert result.exit_code == 0, result.output
        # The shadowed entry should mention "shadows" or similar marker.
        assert "interactive" in result.stdout
        assert "shadow" in result.stdout.lower()


class TestConfigShow:
    def test_show_user_profile(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "fast_query"])
        assert result.exit_code == 0, result.output
        assert "num_results: 5" in result.stdout
        assert "command: query" in result.stdout

    def test_show_builtin_interactive(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "interactive"])
        assert result.exit_code == 0, result.output
        assert "chunk_retrieval_threshold: 0.3" in result.stdout

    def test_show_unknown_profile_errors(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "nonexistent"])
        assert result.exit_code != 0
        assert "nonexistent" in result.output

    def test_show_shadowed_profile_uses_user_version(self, app, tmp_path):
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  interactive:\n"
            "    description: 'Shadowed interactive'\n"
            "    num_results: 7\n"
        )
        _load_yaml_config.cache_clear()
        load_user_config.cache_clear()
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "interactive"])
        assert result.exit_code == 0, result.output
        assert "num_results: 7" in result.stdout
        # The user version does NOT set chunk_retrieval_threshold, so it must
        # not appear in the output.
        assert "chunk_retrieval_threshold" not in result.stdout


class TestConfigValidate:
    def test_validate_clean_yaml(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0, result.output

    def test_validate_invalid_yaml_errors(self, app, tmp_path):
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n  bad:\n    unknown_field: value\n"
        )
        _load_yaml_config.cache_clear()
        load_user_config.cache_clear()
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code != 0
        assert "bad" in result.output


class TestConfigPath:
    def test_path_prints_loaded_yaml(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0, result.output
        assert "phentrieve.yaml" in result.stdout

    def test_path_no_config_lists_search_order(self, app, tmp_path, monkeypatch):
        # Switch to an empty dir without phentrieve.yaml and point user config
        # dir at another empty location so no file is found.
        empty_cwd = tmp_path / "empty"
        empty_cwd.mkdir()
        empty_user = tmp_path / "user_config"
        empty_user.mkdir()
        monkeypatch.chdir(empty_cwd)
        monkeypatch.setattr("phentrieve.utils.get_user_config_dir", lambda: empty_user)
        _load_yaml_config.cache_clear()
        load_user_config.cache_clear()
        runner = CliRunner()
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0, result.output
        assert (
            "No phentrieve.yaml" in result.stdout
            or "not found" in result.stdout.lower()
        )
