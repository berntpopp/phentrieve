from typer.testing import CliRunner

from phentrieve.cli import app


def test_text_process_accepts_llm_backend(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
        raising=False,
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Patient had recurrent seizures.",
            "--extraction-backend",
            "llm",
        ],
    )

    assert result.exit_code == 0
    assert "llm" in result.stdout.lower()


def test_text_process_rejects_invalid_backend():
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Patient had recurrent seizures.",
            "--extraction-backend",
            "bogus",
        ],
    )

    assert result.exit_code != 0
    assert "invalid value" in result.stderr.lower()
