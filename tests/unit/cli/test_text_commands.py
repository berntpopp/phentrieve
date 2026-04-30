import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

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


def test_text_process_phenopacket_output_can_request_sidecar(monkeypatch, tmp_path):
    runner = CliRunner()
    sidecar_path = tmp_path / "packet.annotations.json"
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
            "--phenopacket-sidecar-output-file",
            str(sidecar_path),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["id"] == "packet-1"
    assert json.loads(sidecar_path.read_text())["schema_version"] == "1.0.0"


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


def test_text_process_phenopacket_sidecar_requires_output_file(monkeypatch):
    runner = CliRunner()
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

    assert result.exit_code == 1
    assert "--phenopacket-sidecar-output-file" in result.stderr


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
            "confidence": 0.0,
            "evidence_count": 0,
            "source_chunk_ids": [],
            "max_score_from_evidence": 0.0,
            "top_evidence_chunk_id": None,
            "text_attributions": [],
            "invalid_chunk_reference_count": 0,
            "score": 0.0,
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


def _clear_yaml_cache() -> None:
    from phentrieve.config import _load_yaml_config
    from phentrieve.utils import load_user_config

    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


class TestProcessProfileResolution:
    """Plan A Phase 5: ``phentrieve text process`` --profile resolution.

    The four hardcoded literals (``language="en"``, ``num_results=10``,
    ``chunk_confidence=0.2``, ``assertion_preference="dependency"``) get
    replaced with ``None`` Typer defaults plus a value-or-constant body
    fallback. Together with the eager ``--profile`` callback this lets the
    precedence stack (explicit flag > profile > top-level YAML > constants)
    work without a hardcoded literal short-circuiting it.
    """

    def setup_method(self) -> None:
        _clear_yaml_cache()

    def teardown_method(self) -> None:
        _clear_yaml_cache()

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_default_profile_falls_through_to_config(
        self, mock_run, monkeypatch, tmp_path
    ):
        from phentrieve.config import (
            DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
            DEFAULT_LANGUAGE,
        )

        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("Patient with seizures.")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["text", "process", "--input-file", str(tmp_path / "in.txt")],
        )
        assert result.exit_code == 0, result.output

        kwargs = mock_run.call_args.kwargs
        # No profile, no YAML override -> falls through to config constants.
        assert kwargs["chunk_retrieval_threshold"] == DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
        # language: not hardcoded to "en"; either auto-detected or DEFAULT_LANGUAGE.
        assert kwargs.get("language") in {None, DEFAULT_LANGUAGE}

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_profile_provides_defaults(self, mock_run, monkeypatch, tmp_path):
        # User profile with overrides.
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  high_recall_german:\n"
            "    command: text process\n"
            "    language: de\n"
            "    num_results: 5\n"
            "    chunk_retrieval_threshold: 0.5\n"
        )
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("Der Patient hat Anfaelle.")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "text",
                "process",
                "--input-file",
                str(tmp_path / "in.txt"),
                "--profile",
                "high_recall_german",
            ],
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_run.call_args.kwargs
        assert kwargs["language"] == "de"
        assert kwargs["chunk_retrieval_threshold"] == 0.5
        assert kwargs["num_results_per_chunk"] == 5

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_explicit_flag_overrides_profile(self, mock_run, monkeypatch, tmp_path):
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  custom:\n"
            "    command: text process\n"
            "    language: de\n"
            "    num_results: 5\n"
        )
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("Patient.")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "text",
                "process",
                "--input-file",
                str(tmp_path / "in.txt"),
                "--profile",
                "custom",
                "--language",
                "fr",
                "--num-results",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_run.call_args.kwargs
        # Explicit flags beat profile-supplied values.
        assert kwargs["language"] == "fr"
        assert kwargs["num_results_per_chunk"] == 3

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_language_auto_detect_when_not_in_profile(
        self, mock_run, monkeypatch, tmp_path
    ):
        """When neither flag nor profile/YAML supply a language, the body
        falls through to DEFAULT_LANGUAGE (or auto-detect). This guards that
        ``language="en"`` is no longer hardcoded as a Typer default literal.
        """
        from phentrieve.config import DEFAULT_LANGUAGE

        # Profile that DOES NOT set language.
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  recall_only:\n"
            "    command: text process\n"
            "    chunk_retrieval_threshold: 0.4\n"
        )
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("Patient.")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "text",
                "process",
                "--input-file",
                str(tmp_path / "in.txt"),
                "--profile",
                "recall_only",
            ],
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_run.call_args.kwargs
        # Language is whatever DEFAULT_LANGUAGE resolves to (matches API
        # behaviour). Critically NOT hardcoded as "en" inside Typer signature.
        assert kwargs["language"] == DEFAULT_LANGUAGE
        assert kwargs["chunk_retrieval_threshold"] == 0.4


class TestProcessAdaptiveRechunkingFlags:
    """Plan B Phase 8: --adaptive-rechunking* flags on ``text process``.

    Four flags map onto an ``AdaptiveRechunkingConfig`` resolved with the
    precedence stack CLI > profile > YAML > defaults. The boolean
    ``--adaptive-rechunking / --no-adaptive-rechunking`` only enters the
    CLI tier when the user explicitly supplied it; otherwise profile and
    YAML values pass through.
    """

    def setup_method(self) -> None:
        _clear_yaml_cache()

    def teardown_method(self) -> None:
        _clear_yaml_cache()

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_disabled_by_default(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(
            app, ["text", "process", "--input-file", str(tmp_path / "in.txt")]
        )
        assert result.exit_code == 0, result.output
        cfg = mock_run.call_args.kwargs.get("adaptive_rechunking")
        # When the flag isn't passed, the config is built with enabled=False
        # (or absent - both acceptable).
        if cfg is not None:
            assert getattr(cfg, "enabled", False) is False

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_enabled_via_flag(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "text",
                "process",
                "--input-file",
                str(tmp_path / "in.txt"),
                "--adaptive-rechunking",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
        assert cfg.enabled is True

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_threshold_flags(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "text",
                "process",
                "--input-file",
                str(tmp_path / "in.txt"),
                "--adaptive-rechunking",
                "--adaptive-rechunking-quality-threshold",
                "0.5",
                "--adaptive-rechunking-margin-threshold",
                "0.05",
                "--adaptive-rechunking-max-depth",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
        assert cfg.enabled is True
        assert cfg.quality_threshold == 0.5
        assert cfg.margin_threshold == 0.05
        assert cfg.max_depth == 1

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_yaml_extraction_block_applies(self, mock_run, tmp_path, monkeypatch):
        """YAML extraction.adaptive_rechunking propagates when no CLI flag."""
        (tmp_path / "phentrieve.yaml").write_text(
            "extraction:\n"
            "  adaptive_rechunking:\n"
            "    enabled: true\n"
            "    quality_threshold: 0.42\n"
        )
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(
            app, ["text", "process", "--input-file", str(tmp_path / "in.txt")]
        )
        assert result.exit_code == 0, result.output
        cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
        assert cfg.enabled is True
        assert cfg.quality_threshold == 0.42

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_profile_block_applies(self, mock_run, tmp_path, monkeypatch):
        """``Profile.adaptive_rechunking`` block flows through when no CLI flag."""
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  adaptive_demo:\n"
            "    command: text process\n"
            "    adaptive_rechunking:\n"
            "      enabled: true\n"
            "      quality_threshold: 0.6\n"
            "      max_depth: 3\n"
        )
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "text",
                "process",
                "--input-file",
                str(tmp_path / "in.txt"),
                "--profile",
                "adaptive_demo",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
        assert cfg.enabled is True
        assert cfg.quality_threshold == 0.6
        assert cfg.max_depth == 3

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_explicit_flag_beats_profile(self, mock_run, tmp_path, monkeypatch):
        """An explicit ``--adaptive-rechunking-*`` flag wins over profile values."""
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  adaptive_demo:\n"
            "    command: text process\n"
            "    adaptive_rechunking:\n"
            "      enabled: true\n"
            "      quality_threshold: 0.6\n"
            "      max_depth: 3\n"
        )
        monkeypatch.chdir(tmp_path)
        _clear_yaml_cache()
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        monkeypatch.setattr(
            "phentrieve.cli.text_commands.resolve_chunking_pipeline_config",
            lambda **kwargs: [{"type": "paragraph"}],
        )

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "text",
                "process",
                "--input-file",
                str(tmp_path / "in.txt"),
                "--profile",
                "adaptive_demo",
                "--adaptive-rechunking-quality-threshold",
                "0.5",
                "--adaptive-rechunking-max-depth",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
        # Profile still supplies enabled=True (untouched).
        assert cfg.enabled is True
        # Explicit flags beat profile-supplied values.
        assert cfg.quality_threshold == 0.5
        assert cfg.max_depth == 1
