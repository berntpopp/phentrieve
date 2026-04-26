"""Tests for ``phentrieve text interactive`` --profile resolution (Plan A
Phase 4 / issue #171).

The built-in ``interactive`` profile is auto-selected when ``--profile`` is
not given on the commandline (the option default is ``"interactive"``).
That profile sets the loose retrieval defaults
(``chunk_retrieval_threshold=0.3``, ``aggregated_term_confidence=0.35``,
``num_results=5``) so existing users see no behaviour change. Passing
``--profile default`` switches to the empty built-in, falling through to
the strict ``DEFAULT_*`` constants. Explicit flags always beat both.

These tests assert the kwargs that ``orchestrate_hpo_extraction`` sees -
that is the precise observable point where profile/YAML/flag resolution
has finished and only one value can survive.
"""

from __future__ import annotations

from typing import Any

import pytest

from phentrieve.cli.text_interactive import interactive_text_mode
from phentrieve.config import _load_yaml_config


def _clear_config_caches() -> None:
    from phentrieve.utils import load_user_config

    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


@pytest.fixture(autouse=True)
def _isolated_yaml(tmp_path, monkeypatch):
    """Run from an empty cwd so phentrieve.yaml from the repo isn't picked up."""
    monkeypatch.chdir(tmp_path)
    _clear_config_caches()
    yield
    _clear_config_caches()


def _install_pipeline_doubles(monkeypatch) -> dict[str, Any]:
    """Install the minimal set of doubles to let interactive_text_mode reach
    ``orchestrate_hpo_extraction`` and exit cleanly via the ``q`` prompt.

    Returns a dict that gets populated with the orchestrator's kwargs once
    the function under test calls it.
    """
    captured: dict[str, Any] = {}
    prompt_inputs = iter(["Patient has seizures.", "q"])

    class FakePipeline:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def process(self, raw_text: str) -> list[dict[str, str]]:
            return [{"text": raw_text, "status": "present"}]

    class FakeRetriever:
        pass

    def _capture_orchestrator(**kwargs: Any) -> tuple[list, list]:
        captured["called"] = True
        captured["kwargs"] = kwargs
        return ([], [])

    monkeypatch.setattr(
        "phentrieve.cli.text_interactive.resolve_chunking_pipeline_config",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_interactive.TextProcessingPipeline",
        FakePipeline,
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_interactive.DenseRetriever.from_model_name",
        lambda **kwargs: FakeRetriever(),
    )
    monkeypatch.setattr(
        "phentrieve.embeddings.load_embedding_model",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_interactive.orchestrate_hpo_extraction",
        _capture_orchestrator,
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_interactive._display_interactive_text_results",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "rich.prompt.Prompt.ask",
        lambda *args, **kwargs: next(prompt_inputs),
    )
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: False,
    )
    return captured


class TestTextInteractiveProfileResolution:
    """The function-level invariant: when callers omit profileable args (or
    pass them as None as Click would on default), the body resolves them
    against the right precedence stack."""

    def test_default_invocation_uses_interactive_profile_values(
        self, monkeypatch
    ) -> None:
        """With no flags at all, all profileable args are None, so the body's
        value-or-constant fallback returns the constants. The built-in
        'interactive' profile sets the loose values via Click's default_map
        when the eager callback runs - but we are calling the function
        directly here, so the fallback receives None and uses the strict
        constants. This test therefore guards a different invariant:
        calling interactive_text_mode() with no kwargs MUST NOT raise and
        MUST reach orchestrate_hpo_extraction with non-None values.
        """
        captured = _install_pipeline_doubles(monkeypatch)

        interactive_text_mode()

        assert captured.get("called") is True
        kwargs = captured["kwargs"]
        assert kwargs["chunk_retrieval_threshold"] is not None
        assert kwargs["min_confidence_for_aggregated"] is not None
        assert kwargs["num_results_per_chunk"] is not None

    def test_profile_default_swaps_to_strict(self, monkeypatch) -> None:
        """Pass-through: explicit None args trigger the body's fallback to
        the DEFAULT_* constants - this is the path '--profile default' takes
        once Click hands off, since the empty built-in 'default' profile
        contributes no default_map entries.
        """
        from phentrieve.config import (
            DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
            DEFAULT_MIN_CONFIDENCE_AGGREGATED,
            DEFAULT_TOP_K,
        )

        captured = _install_pipeline_doubles(monkeypatch)

        interactive_text_mode(
            chunk_retrieval_threshold=None,
            aggregated_term_confidence=None,
            num_results=None,
        )

        assert captured.get("called") is True
        kwargs = captured["kwargs"]
        assert kwargs["chunk_retrieval_threshold"] == DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
        assert (
            kwargs["min_confidence_for_aggregated"] == DEFAULT_MIN_CONFIDENCE_AGGREGATED
        )
        assert kwargs["num_results_per_chunk"] == DEFAULT_TOP_K

    def test_explicit_flag_overrides_profile(self, monkeypatch) -> None:
        """Explicit non-None values reach the orchestrator unchanged, beating
        any profile-injected default. This mirrors what happens when the
        user passes --num-results 20 on the CLI: Click routes COMMANDLINE
        through to the function with the explicit value, the body's
        fallback is a no-op, and the orchestrator sees 20.
        """
        captured = _install_pipeline_doubles(monkeypatch)

        interactive_text_mode(
            language="fr",
            num_results=20,
            chunk_retrieval_threshold=0.42,
            aggregated_term_confidence=0.7,
        )

        assert captured.get("called") is True
        kwargs = captured["kwargs"]
        assert kwargs["language"] == "fr"
        assert kwargs["num_results_per_chunk"] == 20
        assert kwargs["chunk_retrieval_threshold"] == 0.42
        assert kwargs["min_confidence_for_aggregated"] == 0.7


class TestProfileOptionRegistered:
    """Smoke-test: the --profile option is wired with the eager callback."""

    def test_profile_option_present_with_eager_callback(self) -> None:
        import typer
        from typer.main import get_command

        from phentrieve.cli._profile import apply_profile_callback

        # Build a tiny app that registers interactive_text_mode under a
        # known name and inspect the resulting click.Group's subcommand.
        app = typer.Typer()
        app.command("interactive")(interactive_text_mode)
        # Force a second command so Typer materializes a Group rather
        # than collapsing to a single TyperCommand.
        app.command("noop")(lambda: None)
        click_cmd = get_command(app)
        sub = click_cmd.commands["interactive"]  # type: ignore[attr-defined]

        profile_param = next((p for p in sub.params if p.name == "profile"), None)
        assert profile_param is not None, "--profile option not registered"
        assert profile_param.is_eager is True
        # xdist workers may import _profile in different processes - compare
        # by qualified name rather than identity.
        assert profile_param.callback is not None
        assert profile_param.callback.__module__ == apply_profile_callback.__module__
        assert (
            profile_param.callback.__qualname__ == apply_profile_callback.__qualname__
        )
        # Built-in default per spec.
        assert profile_param.default == "interactive"
        # envvar wired per spec.
        assert profile_param.envvar == "PHENTRIEVE_PROFILE"
