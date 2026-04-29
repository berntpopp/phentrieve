"""CLI tests for research-use notices and text privacy."""

from typer.testing import CliRunner

from phentrieve.cli import app


def test_query_command_shows_research_notice_without_echoing_raw_text(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.retrieval.query_orchestrator.orchestrate_query",
        lambda **_kwargs: [],
    )

    result = runner.invoke(
        app,
        ["query", "Research note mentions recurrent seizures."],
    )

    assert result.exit_code == 0
    assert "Research use only" in result.stderr
    assert "Research note mentions recurrent seizures" not in result.stdout


def test_text_process_command_shows_research_notice(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **_kwargs: {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Research note mentions recurrent seizures.",
            "--extraction-backend",
            "llm",
            "--llm-model",
            "gpt-5.4-mini",
        ],
    )

    assert result.exit_code == 0
    assert "Research use only" in result.stderr
