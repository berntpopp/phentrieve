import json
import logging
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from phentrieve.cli import app
from phentrieve.cli.text_commands import _run_llm_backend


@pytest.fixture(autouse=True)
def stub_grounded_chunks(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.text_processing.full_text_service.preprocess_grounded_document",
        lambda **kwargs: SimpleNamespace(
            grounded_chunks=[{"chunk_id": 1, "text": kwargs["text"]}],
            extraction_groups=[],
        ),
    )


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


def test_text_process_logs_llm_token_usage(monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("PHENTRIEVE_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": "gpt-4o-mini",
                "llm_mode": "two_phase",
                "token_input": 12,
                "token_output": 34,
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
    assert "LLM token usage:" in result.stderr
    assert "input=12" in result.stderr
    assert "output=34" in result.stderr


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


def test_text_process_phenopacket_output_can_request_sidecar(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.phenopackets.utils.export_phenopacket_bundle",
        lambda **kwargs: {
            "phenopacket_json": '{"id": "packet-1"}',
            "annotation_sidecar": {"schema_version": "1.0.0", "annotations": []},
        },
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {"extraction_backend": "standard"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "clinical note",
            "--output-format",
            "phenopacket_v2_json",
            "--phenopacket-sidecar",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["id"] == "packet-1"
    assert "schema_version" in result.stderr


def test_text_process_phenopacket_output_defaults_to_no_sidecar(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.phenopackets.utils.export_phenopacket_bundle",
        lambda **kwargs: {
            "phenopacket_json": '{"id": "packet-1"}',
            "annotation_sidecar": None,
        },
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {"extraction_backend": "standard"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "clinical note",
            "--output-format",
            "phenopacket_v2_json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["id"] == "packet-1"


def test_run_llm_backend_uses_pipeline_and_provider(monkeypatch):
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, *, provider):
            calls["provider"] = provider

        def run(self, *, text, grounded_chunks, config, extraction_groups=None):
            calls["text"] = text
            calls["grounded_chunks"] = grounded_chunks
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
    assert isinstance(calls["grounded_chunks"], list)
    assert result["meta"]["extraction_backend"] == "llm"
    assert result["meta"]["llm_model"] == "gemini-2.5-flash"
    assert result["meta"]["llm_mode"] == "two_phase"
    assert result["aggregated_hpo_terms"] == [
        {
            "id": "HP:0001250",
            "name": "Seizure",
            "evidence": "Patient had recurrent seizures.",
            "status": "present",
            "evidence_records": [],
        }
    ]


def test_run_llm_backend_supports_injected_factories_for_supported_mode():
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, *, provider):
            calls["provider"] = provider

        def run(self, *, text, grounded_chunks, config, extraction_groups=None):
            calls["text"] = text
            calls["grounded_chunks"] = grounded_chunks
            calls["config"] = config
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                ),
            )

    fake_provider = object()
    result = _run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        provider_factory=lambda **kwargs: fake_provider,
        pipeline_factory=FakePipeline,
    )

    assert calls["provider"] is fake_provider
    assert calls["text"] == "Patient had recurrent seizures."
    assert isinstance(calls["grounded_chunks"], list)
    assert calls["config"].mode == "two_phase"
    assert result["meta"]["llm_mode"] == "two_phase"


def test_run_llm_backend_rejects_invalid_mode_before_provider(monkeypatch):
    calls = {"provider_called": False}

    def provider_factory(**kwargs):
        calls["provider_called"] = True
        raise AssertionError("provider_factory should not be called for invalid mode")

    with pytest.raises(ValueError, match="Unsupported LLM mode: 'bogus'"):
        _run_llm_backend(
            text="Patient had recurrent seizures.",
            llm_model="gemini-2.5-flash",
            llm_mode="bogus",
            provider_factory=provider_factory,
            pipeline_factory=lambda **kwargs: None,
        )

    assert calls["provider_called"] is False


def test_run_llm_backend_logs_completion_once(caplog):
    caplog.set_level(
        logging.INFO, logger="phentrieve.text_processing.full_text_service"
    )

    class FakePipeline:
        def __init__(self, *, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config, extraction_groups=None):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                ),
            )

    _run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        provider_factory=lambda **kwargs: object(),
        pipeline_factory=FakePipeline,
    )

    completion_logs = [
        record.message
        for record in caplog.records
        if record.message.startswith("LLM backend completed:")
    ]
    assert len(completion_logs) == 1


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


def test_text_process_passes_provider_and_base_url_to_service(monkeypatch) -> None:
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
            "--extraction-backend",
            "llm",
            "--llm-provider",
            "ollama",
            "--llm-model",
            "qwen3.5:35b",
            "--llm-base-url",
            "http://localhost:11434",
            "Patient has seizures.",
        ],
    )

    assert result.exit_code == 0
    assert calls["llm_provider"] == "ollama"
    assert calls["llm_base_url"] == "http://localhost:11434"


def test_text_process_keeps_bare_gemini_model_compatible(monkeypatch) -> None:
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
            "--extraction-backend",
            "llm",
            "--llm-model",
            "gemini-2.5-flash",
            "Patient has seizures.",
        ],
    )

    assert result.exit_code == 0
    assert calls["llm_model"] == "gemini-2.5-flash"


def test_text_process_passes_llm_internal_mode_to_service(monkeypatch):
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
            "--llm-internal-mode",
            "whole_document_grounded",
        ],
    )

    assert result.exit_code == 0
    assert calls["llm_internal_mode"] == "whole_document_grounded"


def test_text_process_honors_assertion_preference_key(monkeypatch):
    runner = CliRunner()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
        lambda **kwargs: [{"type": "paragraph"}],
    )

    def fake_run_full_text_service(**kwargs):
        calls.update(kwargs)
        return {
            "meta": {"extraction_backend": "standard"},
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
            "--assertion-preference",
            "keyword",
        ],
    )

    assert result.exit_code == 0
    assert calls["assertion_config"] == {
        "disable": False,
        "preference": "keyword",
    }


def test_text_process_logs_llm_backend_configuration(monkeypatch, caplog):
    runner = CliRunner()
    caplog.set_level(logging.DEBUG, logger="phentrieve.cli.text_commands")

    monkeypatch.setattr(
        "phentrieve.utils.setup_logging_cli",
        lambda debug=False: None,
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": kwargs["llm_model"],
                "llm_mode": kwargs["llm_mode"],
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
            "--llm-model",
            "gemini-2.5-flash",
            "--llm-mode",
            "two_phase",
        ],
    )

    assert result.exit_code == 0
    assert any(
        record.message == "Using full-text extraction backend: llm"
        for record in caplog.records
    )
    assert any(
        "LLM backend configuration: model=gemini-2.5-flash, mode=two_phase"
        in record.message
        for record in caplog.records
    )


def test_text_process_uses_default_llm_model_when_not_explicitly_configured(
    monkeypatch,
):
    runner = CliRunner()
    monkeypatch.delenv("PHENTRIEVE_LLM_MODEL", raising=False)
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "llm_model": kwargs["llm_model"],
                "llm_mode": kwargs["llm_mode"],
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


def test_run_llm_backend_uses_default_model_when_not_explicitly_configured(
    monkeypatch,
):
    monkeypatch.delenv("PHENTRIEVE_LLM_MODEL", raising=False)

    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, *, provider):
            calls["provider"] = provider

        def warmup(self, language: str) -> None:
            calls["warmup_language"] = language

        def run(self, *, text, grounded_chunks, config, extraction_groups=None):
            calls["config"] = config
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(
        "phentrieve.cli.text_commands.get_llm_provider",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.TwoPhaseLLMPipeline",
        FakePipeline,
    )

    result = _run_llm_backend(text="Patient had recurrent seizures.")

    assert calls["config"].model == "gemini-3.1-flash-lite-preview"
    assert result["meta"]["llm_model"] == "gemini-3.1-flash-lite-preview"
