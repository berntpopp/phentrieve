import json

import pytest
from typer.testing import CliRunner

from phentrieve.cli import app
from phentrieve.cli.text_commands import _run_llm_backend


def test_text_process_accepts_llm_backend(monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("PHENTRIEVE_LLM_MODEL", "gpt-4o-mini")
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
    monkeypatch.setenv("PHENTRIEVE_LLM_MODEL", "gpt-4o-mini")
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
    monkeypatch.setenv("PHENTRIEVE_LLM_MODEL", "gpt-4o-mini")
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


def test_text_process_rich_json_summary_handles_llm_terms(monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("PHENTRIEVE_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": "gpt-4o-mini",
                "llm_mode": "two_phase",
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [
                {
                    "id": "HP:0001250",
                    "name": "Seizure",
                    "evidence": "Patient had recurrent seizures.",
                    "status": "present",
                }
            ],
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
            "rich_json_summary",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["document"]["total_hpo_terms"] == 1
    assert payload["document"]["hpo_terms"][0]["hpo_id"] == "HP:0001250"
    assert payload["document"]["hpo_terms"][0]["confidence"] == 0.0
    assert payload["document"]["hpo_terms"][0]["status"] == "present"
    assert payload["document"]["hpo_terms"][0]["evidence_count"] == 1


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


def test_run_llm_backend_uses_pipeline_and_provider(monkeypatch):
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, *, provider):
            calls["provider"] = provider

        def run(self, *, text, config):
            calls["text"] = text
            calls["config"] = config
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001250",
                        label="Seizure",
                        evidence="Patient had recurrent seizures.",
                        assertion="present",
                    )
                ],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                ),
            )

    fake_provider = object()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.get_llm_provider",
        lambda **kwargs: fake_provider,
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.TwoPhaseLLMPipeline",
        FakePipeline,
    )

    result = _run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
    )

    assert calls["provider"] is fake_provider
    assert calls["text"] == "Patient had recurrent seizures."
    assert result["meta"]["extraction_backend"] == "llm"
    assert result["meta"]["llm_model"] == "gemini-2.5-flash"
    assert result["meta"]["llm_mode"] == "two_phase"
    assert result["aggregated_hpo_terms"] == [
        {
            "id": "HP:0001250",
            "name": "Seizure",
            "evidence": "Patient had recurrent seizures.",
            "status": "present",
        }
    ]


def test_text_process_passes_llm_options_to_service(monkeypatch):
    runner = CliRunner()
    calls: dict[str, object] = {}

    def fake_run_full_text_service(**kwargs):
        calls.update(kwargs)
        return {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": kwargs["llm_model"],
                "llm_mode": kwargs["llm_mode"],
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }

    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        fake_run_full_text_service,
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Patient had recurrent seizures.",
            "--extraction-backend",
            "llm",
            "--llm-model",
            "gemini-2.5-flash",
            "--llm-mode",
            "two_phase",
        ],
    )

    assert result.exit_code == 0
    assert calls["llm_model"] == "gemini-2.5-flash"
    assert calls["llm_mode"] == "two_phase"


def test_text_process_rejects_llm_backend_without_model(monkeypatch):
    runner = CliRunner()
    monkeypatch.delenv("PHENTRIEVE_LLM_MODEL", raising=False)

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

    assert result.exit_code == 1
    assert "Provide --llm-model or set PHENTRIEVE_LLM_MODEL" in result.stderr


def test_run_llm_backend_requires_explicit_model(monkeypatch):
    monkeypatch.delenv("PHENTRIEVE_LLM_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="No LLM model configured"):
        _run_llm_backend(text="Patient had recurrent seizures.")
