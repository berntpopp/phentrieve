"""Tests for the eager --profile callback (Spec A Task 7)."""

from __future__ import annotations

import click
import typer
from typer.testing import CliRunner


def make_test_app():
    """A throwaway Typer app with --profile wired via the eager callback."""
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
        ctx.obj.setdefault("seen_root", True)

    @app.command()
    def echo(
        ctx: typer.Context,
        language: str | None = None,
        num_results: int | None = None,
    ):
        # Resolve fallbacks.
        lang = language if language is not None else "en"
        n = num_results if num_results is not None else 10
        click.echo(f"language={lang} num_results={n}")

    return app


class TestApplyProfileCallback:
    def test_default_profile_no_explicit_flag(self, monkeypatch, tmp_path):
        from phentrieve.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)  # No phentrieve.yaml present.
        _load_yaml_config.cache_clear()

        runner = CliRunner()
        result = runner.invoke(make_test_app(), ["echo"])
        assert result.exit_code == 0, result.output
        # No profile, no YAML -> falls through to function-body fallbacks.
        assert "language=en" in result.stdout
        assert "num_results=10" in result.stdout

    def test_unknown_profile_errors(self, monkeypatch, tmp_path):
        from phentrieve.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()

        runner = CliRunner()
        result = runner.invoke(make_test_app(), ["--profile", "ghost", "echo"])
        assert result.exit_code != 0
        assert "ghost" in result.output


class TestSidecarSourceMap:
    def test_sidecar_populated_with_profile_label(self, monkeypatch, tmp_path):
        """The built-in `interactive` profile sets chunk_retrieval_threshold=0.3
        and the sidecar map records that the value came from the profile.
        """
        from phentrieve.cli._profile import apply_profile_callback
        from phentrieve.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()

        captured: dict = {}

        app = typer.Typer()

        @app.callback()
        def root(
            ctx: typer.Context,
            profile: str = typer.Option(
                "interactive",  # built-in default for this test app
                "--profile",
                callback=apply_profile_callback,
                is_eager=True,
            ),
        ):
            ctx.ensure_object(dict)

        @app.command()
        def cmd(
            ctx: typer.Context,
            chunk_retrieval_threshold: float | None = None,
        ):
            captured["sources"] = dict(ctx.obj.get("resolved_sources", {}))
            captured["threshold"] = (
                chunk_retrieval_threshold
                if chunk_retrieval_threshold is not None
                else 0.7  # the constant fallback
            )

        runner = CliRunner()
        result = runner.invoke(app, ["cmd"])
        assert result.exit_code == 0, result.output
        # Built-in interactive sets chunk_retrieval_threshold=0.3.
        assert captured["threshold"] == 0.3
        assert "chunk_retrieval_threshold" in captured["sources"]
        label = captured["sources"]["chunk_retrieval_threshold"]
        assert "profile:builtin:interactive" in label or "profile:interactive" in label
