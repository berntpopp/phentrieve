"""Deprecated no-op options on ``phentrieve text process`` (#414)."""

import pytest
from typer.testing import CliRunner

from phentrieve.cli import app


@pytest.fixture
def stub_llm_backend(monkeypatch):
    monkeypatch.setenv("PHENTRIEVE_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )


BASE_ARGS = ["text", "process", "Patient had seizures.", "--extraction-backend", "llm"]


@pytest.mark.parametrize(
    ("extra_args", "flag"),
    [
        (["--xlhpo"], "--cross-language-hpo-retrieval"),
        (["--cross-language-hpo-retrieval"], "--cross-language-hpo-retrieval"),
        (["--use-cached-index"], "--use-cached-index"),
        (["-cc", "0.4"], "--chunk-confidence"),
    ],
)
def test_deprecated_noop_options_still_parse_and_warn(
    stub_llm_backend, extra_args, flag
):
    result = CliRunner().invoke(app, [*BASE_ARGS, *extra_args])
    assert result.exit_code == 0, result.output
    assert f"{flag} is deprecated and has no effect" in result.stderr


def test_no_deprecation_warning_without_the_options(stub_llm_backend):
    result = CliRunner().invoke(app, BASE_ARGS)
    assert result.exit_code == 0, result.output
    assert "is deprecated and has no effect" not in result.stderr


def test_help_hides_deprecated_noop_options():
    result = CliRunner().invoke(app, ["text", "process", "--help"])
    assert result.exit_code == 0
    for flag in ("--xlhpo", "--use-cached-index", "--chunk-confidence"):
        assert flag not in result.output
