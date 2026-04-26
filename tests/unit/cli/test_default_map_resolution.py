"""Tests proving Click's default_map resolution combined with the sidecar
source map (Spec A Task 8).

The eager --profile callback merges values from the profile, top-level
YAML, and fallback constants into ``ctx.default_map``. ``ParameterSource``
alone cannot distinguish those layers - they all report as ``DEFAULT_MAP``.
The sidecar map at ``ctx.obj['resolved_sources']`` provides the fine-grained
label. These tests assert BOTH signals on representative paths.
"""

from __future__ import annotations

import textwrap

import click
import typer
from click.core import ParameterSource
from typer.testing import CliRunner


def _clear_config_caches() -> None:
    """Clear every LRU cache that backs phentrieve YAML config loading.

    Both ``phentrieve.config._load_yaml_config`` AND
    ``phentrieve.utils.load_user_config`` are cached separately; clearing
    only one leaves the other returning stale state across tests.
    """
    from phentrieve.config import _load_yaml_config
    from phentrieve.utils import load_user_config

    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


def _make_app(captured: dict):
    """Build a tiny Typer app whose subcommand records both signals."""
    from phentrieve.cli._profile import apply_profile_callback

    app = typer.Typer()

    @app.callback()
    def root(
        ctx: typer.Context,
        profile: str = typer.Option(
            "default",
            "--profile",
            callback=apply_profile_callback,
            is_eager=True,
        ),
    ):
        ctx.ensure_object(dict)

    @app.command()
    def cmd(
        ctx: typer.Context,
        language: str | None = typer.Option(None, "--language"),
        chunk_retrieval_threshold: float | None = typer.Option(
            None, "--chunk-retrieval-threshold"
        ),
        num_results: int | None = typer.Option(None, "--num-results"),
    ):
        captured["language"] = language
        captured["chunk_retrieval_threshold"] = chunk_retrieval_threshold
        captured["num_results"] = num_results
        captured["sources"] = dict(ctx.obj.get("resolved_sources", {}))
        captured["language_param_source"] = ctx.get_parameter_source("language")
        captured["chunk_param_source"] = ctx.get_parameter_source(
            "chunk_retrieval_threshold"
        )
        captured["num_results_param_source"] = ctx.get_parameter_source("num_results")

    return app


class TestDefaultMapResolution:
    """Each invariant proves: function body sees the right value, and BOTH
    ``ParameterSource`` and ``resolved_sources`` agree on its provenance."""

    def test_constant_when_nothing_is_set(self, monkeypatch, tmp_path):
        """No profile (built-in 'default'), no YAML, no flag.
        Expectation: function-body fallback receives the resolved value
        from ``default_map`` populated with the const-source label.
        """
        monkeypatch.chdir(tmp_path)
        _clear_config_caches()

        captured: dict = {}
        runner = CliRunner()
        result = runner.invoke(_make_app(captured), ["cmd"])
        assert result.exit_code == 0, result.output

        # No --language flag => language defaulted from default_map.
        # Built-in 'default' profile is empty and YAML is empty, so the
        # fallback constant supplies the value.
        assert captured["language_param_source"] == ParameterSource.DEFAULT_MAP
        assert captured["sources"]["language"].startswith("const:")
        assert "DEFAULT_LANGUAGE" in captured["sources"]["language"]

    def test_yaml_value_when_no_profile_no_flag(self, monkeypatch, tmp_path):
        """Top-level YAML supplies a value; no profile preset, no flag.
        Expectation: language comes from YAML, sidecar shows yaml: label,
        ParameterSource is DEFAULT_MAP.
        """
        yaml_path = tmp_path / "phentrieve.yaml"
        yaml_path.write_text(
            textwrap.dedent(
                """\
                default_language: it
                """
            )
        )
        monkeypatch.chdir(tmp_path)
        _clear_config_caches()

        captured: dict = {}
        runner = CliRunner()
        result = runner.invoke(_make_app(captured), ["cmd"])
        assert result.exit_code == 0, result.output

        assert captured["language"] == "it"
        assert captured["language_param_source"] == ParameterSource.DEFAULT_MAP
        assert captured["sources"]["language"] == "yaml:default_language"

    def test_profile_value_beats_yaml(self, monkeypatch, tmp_path):
        """A user profile sets language=de while YAML says it.
        Expectation: profile wins; sidecar carries profile:<name>.
        """
        yaml_path = tmp_path / "phentrieve.yaml"
        yaml_path.write_text(
            textwrap.dedent(
                """\
                default_language: it
                profiles:
                  german_workflow:
                    description: German extraction
                    language: de
                """
            )
        )
        monkeypatch.chdir(tmp_path)
        _clear_config_caches()

        captured: dict = {}
        runner = CliRunner()
        result = runner.invoke(
            _make_app(captured), ["--profile", "german_workflow", "cmd"]
        )
        assert result.exit_code == 0, result.output

        assert captured["language"] == "de"
        assert captured["language_param_source"] == ParameterSource.DEFAULT_MAP
        assert captured["sources"]["language"] == "profile:german_workflow"

    def test_explicit_flag_beats_profile(self, monkeypatch, tmp_path):
        """User passes --language fr while profile says de.
        Expectation: function sees fr; ParameterSource is COMMANDLINE
        (overrides any sidecar entry).
        """
        yaml_path = tmp_path / "phentrieve.yaml"
        yaml_path.write_text(
            textwrap.dedent(
                """\
                profiles:
                  german_workflow:
                    language: de
                """
            )
        )
        monkeypatch.chdir(tmp_path)
        _clear_config_caches()

        captured: dict = {}
        runner = CliRunner()
        result = runner.invoke(
            _make_app(captured),
            ["--profile", "german_workflow", "cmd", "--language", "fr"],
        )
        assert result.exit_code == 0, result.output

        assert captured["language"] == "fr"
        # COMMANDLINE always wins over any default_map injection - this is
        # the user-vs-injected distinction the spec calls out as the key
        # signal that the sidecar source map alone cannot provide.
        assert captured["language_param_source"] == ParameterSource.COMMANDLINE

    def test_builtin_interactive_profile_label(self, monkeypatch, tmp_path):
        """The interactive built-in pre-sets chunk_retrieval_threshold=0.3.
        Expectation: param source DEFAULT_MAP and the sidecar label is the
        profile-named label since the per-command --profile default is
        ``interactive``.
        """
        from phentrieve.cli._profile import apply_profile_callback

        monkeypatch.chdir(tmp_path)
        _clear_config_caches()

        captured: dict = {}

        app = typer.Typer()

        @app.callback()
        def root(
            ctx: typer.Context,
            profile: str = typer.Option(
                "interactive",
                "--profile",
                callback=apply_profile_callback,
                is_eager=True,
            ),
        ):
            ctx.ensure_object(dict)

        @app.command()
        def cmd(
            ctx: typer.Context,
            chunk_retrieval_threshold: float | None = typer.Option(
                None, "--chunk-retrieval-threshold"
            ),
        ):
            captured["chunk"] = chunk_retrieval_threshold
            captured["sources"] = dict(ctx.obj.get("resolved_sources", {}))
            captured["param_source"] = ctx.get_parameter_source(
                "chunk_retrieval_threshold"
            )

        runner = CliRunner()
        result = runner.invoke(app, ["cmd"])
        assert result.exit_code == 0, result.output

        assert captured["chunk"] == 0.3
        assert captured["param_source"] == ParameterSource.DEFAULT_MAP
        # Label is profile:interactive (the value Click passes through).
        assert captured["sources"]["chunk_retrieval_threshold"] == (
            "profile:interactive"
        )

    def test_idempotent_callback(self, monkeypatch, tmp_path):
        """Calling the callback twice with the same value must be a no-op
        (avoids double-population when wired on root and subcommand).
        """
        from phentrieve.cli._profile import apply_profile_callback

        monkeypatch.chdir(tmp_path)
        _clear_config_caches()

        # Build a minimal Click context manually so we can call the
        # callback twice and inspect ctx state.
        cmd = click.Command("dummy")
        ctx = click.Context(cmd)
        ctx.ensure_object(dict)

        # First call: populates resolved_sources with the const labels.
        apply_profile_callback(ctx, None, "default")  # type: ignore[arg-type]
        first_sources = dict(ctx.obj["resolved_sources"])
        first_default_map = dict(ctx.default_map or {})

        # Second call with the same value: no-op (the _profile_applied
        # guard short-circuits).
        apply_profile_callback(ctx, None, "default")  # type: ignore[arg-type]
        second_sources = dict(ctx.obj["resolved_sources"])
        second_default_map = dict(ctx.default_map or {})

        assert first_sources == second_sources
        assert first_default_map == second_default_map
