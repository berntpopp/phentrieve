import json

from typer.testing import CliRunner

from phentrieve.cli import app


def test_text_process_accepts_llm_backend(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": "gpt-4o-mini",
                "llm_mode": "two_phase",
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
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
    assert "LLM metadata:" in result.stderr
    assert "model=gpt-4o-mini" in result.stderr
    assert "mode=two_phase" in result.stderr


def test_text_process_suppresses_llm_note_without_metadata(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
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
            "Patient had recurrent seizures.",
            "--extraction-backend",
            "llm",
        ],
    )

    assert result.exit_code == 0
    assert "Extraction backend: llm" not in result.stdout
    assert "LLM metadata:" not in result.stdout
    assert "LLM metadata:" not in result.stderr


def test_text_process_writes_llm_note_to_stderr_for_json_lines(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": "gpt-4o-mini",
                "llm_mode": "two_phase",
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Patient had recurrent seizures.",
            "--extraction-backend",
            "llm",
            "--output-format",
            "json_lines",
        ],
    )

    assert result.exit_code == 0
    assert "LLM metadata:" not in result.stdout
    assert "LLM metadata:" in result.stderr
    for line in result.stdout.splitlines():
        if line.strip():
            json.loads(line)


def test_text_process_handles_standard_chunks(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {"extraction_backend": "standard"},
            "processed_chunks": [
                {
                    "chunk_id": 1,
                    "text": "First chunk",
                    "status": "affirmed",
                    "hpo_matches": [
                        {
                            "id": "HP:0001250",
                            "name": "Seizure",
                            "score": 0.93,
                            "assertion_status": "affirmed",
                        }
                    ],
                    "start_char": 0,
                    "end_char": 11,
                },
                {
                    "chunk_id": 2,
                    "text": "Second chunk",
                    "status": "negated",
                    "hpo_matches": [],
                    "start_char": 12,
                    "end_char": 24,
                },
            ],
            "aggregated_hpo_terms": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Patient had recurrent seizures.",
            "--output-format",
            "json_lines",
        ],
    )

    assert result.exit_code == 0
    parsed_lines = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert any("aggregated_hpo_terms" in line for line in parsed_lines)
    assert parsed_lines[-1]["aggregated_hpo_terms"][0]["chunk_idx"] == 0
    assert parsed_lines[-1]["aggregated_hpo_terms"][0]["chunk_text"] == "First chunk"


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
