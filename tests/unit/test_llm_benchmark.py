from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from phentrieve.benchmark import llm_benchmark, llm_cli

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def stub_grounded_chunks(monkeypatch):
    monkeypatch.setattr(
        llm_benchmark,
        "_build_grounded_chunks",
        lambda **kwargs: [{"chunk_id": 1, "text": kwargs["text"]}],
    )


def test_run_llm_benchmark_defaults_to_genereviews_dataset(monkeypatch):
    captured: dict[str, object] = {}

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        captured["test_path"] = test_path
        captured["dataset"] = dataset
        return {
            "metadata": {
                "dataset_name": f"phenobert_{dataset}",
                "source": "phenobert",
                "total_documents": 1,
                "total_annotations": 0,
            },
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
    )

    assert captured["dataset"] == "GeneReviews"
    assert result["dataset"] == "GeneReviews"
    assert result["cases"] == 1
    assert result["dataset_metadata"]["dataset_name"] == "phenobert_GeneReviews"


def test_run_llm_benchmark_includes_ontology_metrics_when_enabled(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    monkeypatch.setattr(llm_benchmark, "validate_hpo_graph_available", lambda: None)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        ontology_aware_metrics=True,
    )

    assert "ontology_metrics" in result["metrics"]["assertion_aware"]
    assert "ontology_metrics" in result["metrics"]["id_only"]
    assert result["ontology_aware_metrics"] is True
    assert result["ontology_semantic_floor"] == 0.30
    assert result["ontology_similarity_formula"] == "hybrid"


def test_run_llm_benchmark_requires_hpo_graph_for_ontology_metrics(monkeypatch):
    def fail_load_benchmark_data(*_args, **_kwargs):
        raise AssertionError("benchmark data should not load without ontology graph")

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fail_load_benchmark_data)
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: ({}, {}),
    )

    with pytest.raises(RuntimeError, match="HPO graph data is required"):
        llm_benchmark.run_llm_benchmark(
            test_file="tests/data/en/phenobert",
            llm_model="gemini-2.5-flash",
            ontology_aware_metrics=True,
        )


def test_run_llm_benchmark_filters_to_requested_doc_ids(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "GeneReviews_NBK1277",
                    "text": "Clinical text 1",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
                {
                    "id": "GeneReviews_NBK148668",
                    "text": "Clinical text 2",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        doc_ids=["GeneReviews_NBK148668"],
    )

    assert result["cases"] == 1
    assert result["results"][0]["doc_id"] == "GeneReviews_NBK148668"


def test_run_llm_benchmark_preserves_requested_doc_id_order(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "GeneReviews_NBK1277",
                    "text": "Clinical text 1",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
                {
                    "id": "GeneReviews_NBK148668",
                    "text": "Clinical text 2",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        doc_ids=["GeneReviews_NBK148668", "GeneReviews_NBK1277"],
    )

    assert [record["doc_id"] for record in result["results"]] == [
        "GeneReviews_NBK148668",
        "GeneReviews_NBK1277",
    ]


def test_run_llm_benchmark_cli_sets_up_logging(tmp_path, mocker, monkeypatch):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "result.json"

    mock_setup_logging = mocker.patch("phentrieve.benchmark.llm_cli.setup_logging_cli")
    monkeypatch.setattr(
        llm_cli.llm_benchmark,
        "run_llm_benchmark",
        lambda **kwargs: {
            "cases": 0,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
            "output_path": str(output_path),
        },
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_path=str(output_path),
    )

    mock_setup_logging.assert_called_once_with(debug=False)
    assert result["output_path"] == str(output_path)
    assert output_path.exists()


def test_run_llm_benchmark_passes_provider_to_factory(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakeProvider:
        provider_name = "ollama"
        model_name = "qwen3.5:35b"
        base_url = "http://localhost:11434"

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_provider=config.provider,
                    llm_model=config.model,
                    llm_mode=config.mode,
                ),
            )

    def fake_get_llm_provider(**kwargs):
        captured.update(kwargs)
        return _FakeProvider()

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_provider="ollama",
        llm_model="qwen3.5:35b",
        llm_timeout_seconds=900,
    )

    assert captured["llm_provider"] == "ollama"
    assert captured["timeout_seconds"] == 900


def test_run_llm_benchmark_rejects_invalid_ontology_formula_before_loading(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_benchmark,
        "load_benchmark_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("benchmark data should not be loaded")
        ),
    )

    with pytest.raises(ValueError, match="Invalid ontology similarity formula"):
        llm_benchmark.run_llm_benchmark(
            test_file="tests/data/en/phenobert",
            llm_model="gemini-2.5-flash",
            ontology_aware_metrics=True,
            ontology_similarity_formula="invalid_formula",
        )


@pytest.mark.parametrize("semantic_floor", [-0.1, 1.1])
def test_run_llm_benchmark_rejects_invalid_ontology_floor_before_loading(
    monkeypatch,
    semantic_floor,
) -> None:
    monkeypatch.setattr(
        llm_benchmark,
        "load_benchmark_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("benchmark data should not be loaded")
        ),
    )

    with pytest.raises(ValueError, match="ontology_semantic_floor"):
        llm_benchmark.run_llm_benchmark(
            test_file="tests/data/en/phenobert",
            llm_model="gemini-2.5-flash",
            ontology_aware_metrics=True,
            ontology_semantic_floor=semantic_floor,
        )


def test_run_llm_benchmark_records_resolved_provider_base_url(monkeypatch) -> None:
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakeProvider:
        provider_name = "ollama"
        model_name = "qwen3.5:35b"
        base_url = "http://localhost:11434"

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_provider=config.provider,
                    llm_model=config.model,
                    llm_mode=config.mode,
                ),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(
        llm_benchmark, "get_llm_provider", lambda **kwargs: _FakeProvider()
    )
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_provider="ollama",
        llm_model="qwen3.5:35b",
    )

    assert result["llm_base_url"] == "http://localhost:11434"


def test_run_llm_benchmark_records_openai_provider_metadata(monkeypatch) -> None:
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakeProvider:
        provider_name = "openai"
        model_name = "gpt-5.4-mini"
        base_url = "https://api.openai.com/v1"
        token_count_source = "estimated"  # noqa: S105

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_provider=config.provider,
                    llm_model=config.model,
                    llm_mode=config.mode,
                    token_count_source="estimated",  # noqa: S106
                ),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(
        llm_benchmark, "get_llm_provider", lambda **kwargs: _FakeProvider()
    )
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_provider="openai",
        llm_model="gpt-5.4-mini",
    )

    assert result["llm_provider"] == "openai"
    assert result["llm_model"] == "gpt-5.4-mini"
    assert result["llm_base_url"] == "https://api.openai.com/v1"
    assert result["prediction_records"][0]["metadata"]["llm_provider"] == "openai"
    assert result["prediction_records"][0]["metadata"]["model"] == "gpt-5.4-mini"
    assert (
        result["prediction_records"][0]["metadata"]["observability"][
            "token_count_source"
        ]
        == "estimated"  # noqa: S105
    )


def test_run_llm_benchmark_returns_benchmark_grade_metadata(monkeypatch):
    captured: dict[str, object] = {}

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {
                "dataset_name": f"phenobert_{dataset}",
                "source": "phenobert",
                "total_documents": 1,
                "total_annotations": 1,
            },
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [{"id": "HP:0001250", "assertion": "PRESENT"}],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            captured["language"] = config.language
            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001250",
                        label="Seizure",
                        evidence="has seizures",
                        assertion="present",
                    )
                ],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    prompt_version="test-v1",
                    token_usage={
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "total_tokens": 150,
                        "api_calls": 4,
                        "thoughts_tokens": 9,
                        "cached_content_tokens": 12,
                    },
                    token_input=120,
                    token_output=30,
                    request_count=4,
                    phase_timings={
                        "phase1_seconds": 0.4,
                        "phase2a_seconds": 0.2,
                        "phase2b_local_seconds": 0.1,
                        "phase2b_llm_seconds": 0.3,
                    },
                    phase_counts={
                        "extracted_phrases": 1,
                        "actionable_phrases": 1,
                        "candidate_sets": 1,
                        "unresolved_phrases": 0,
                        "local_matches": 1,
                        "llm_mapped_phrases": 0,
                        "local_fallbacks": 0,
                    },
                ),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        language="de",
        input_cost_per_1m_tokens=0.1,
        output_cost_per_1m_tokens=0.4,
    )

    assert captured["language"] == "de"
    assert result["token_usage"]["prompt_tokens"] == 120
    assert result["token_usage"]["completion_tokens"] == 30
    assert result["token_usage"]["total_tokens"] == 150
    assert result["token_usage"]["api_calls"] == 4
    assert result["token_usage"]["thoughts_tokens"] == 9
    assert result["token_usage"]["cached_content_tokens"] == 12
    assert result["timing_breakdown"]["wall_clock_seconds"] >= 0.0
    assert result["estimated_cost"]["input_cost"] > 0.0
    assert result["estimated_cost"]["output_cost"] > 0.0
    assert result["prediction_records"][0]["doc_id"] == "doc-1"
    assert (
        result["prediction_records"][0]["metadata"]["token_usage"]["total_tokens"]
        == 150
    )
    assert (
        result["prediction_records"][0]["metadata"]["token_usage"]["thoughts_tokens"]
        == 9
    )
    assert (
        result["prediction_records"][0]["metadata"]["timing_breakdown"]["total_seconds"]
        >= 0.0
    )
    assert (
        result["prediction_records"][0]["metadata"]["timing_breakdown"][
            "phase2b_llm_seconds"
        ]
        == 0.3
    )
    assert (
        result["prediction_records"][0]["metadata"]["observability"]["request_count"]
        == 4
    )


def test_estimate_cost_uses_cached_and_thought_tokens() -> None:
    estimated = llm_benchmark._estimate_cost(
        token_usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "thoughts_tokens": 30,
            "cached_content_tokens": 40,
        },
        pricing=llm_benchmark.TokenPricingConfig(
            input_cost_per_1m_tokens=1.0,
            output_cost_per_1m_tokens=2.0,
            cached_input_cost_per_1m_tokens=0.5,
        ),
    )

    assert estimated == {
        "input_cost": 0.00006,
        "cached_input_cost": 0.00002,
        "output_cost": 0.0001,
        "total_cost": 0.00018,
        "billable_input_tokens": 60,
        "billable_cached_tokens": 40,
        "billable_output_tokens": 50,
    }


def test_run_llm_benchmark_emits_token_cost_blocks_and_compat_alias(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    token_usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "thoughts_tokens": 30,
                        "cached_content_tokens": 40,
                        "total_tokens": 150,
                    },
                ),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        input_cost_per_1m_tokens=0.1,
        output_cost_per_1m_tokens=0.4,
    )

    assert result["estimated_token_cost"]["total_cost"] > 0.0
    assert result["estimated_cost"] == result["estimated_token_cost"]


def test_benchmark_accounting_config_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        llm_benchmark.BenchmarkAccountingConfig(
            token_pricing=llm_benchmark.TokenPricingConfig(
                input_cost_per_1m_tokens=-1.0
            )
        )


def test_run_llm_benchmark_cli_prefers_cli_pricing_over_file(
    tmp_path, monkeypatch
) -> None:
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "result.json"
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {
                "token_pricing": {
                    "input_cost_per_1m_tokens": 0.2,
                    "output_cost_per_1m_tokens": 0.3,
                }
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_llm_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "cases": 0,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
        }

    monkeypatch.setattr(
        llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark
    )

    llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        pricing_config=str(pricing_path),
        input_cost_per_1m_tokens=0.9,
        output_path=str(output_path),
    )

    accounting_config = captured["accounting_config"]
    assert accounting_config.token_pricing.input_cost_per_1m_tokens == 0.9
    assert accounting_config.token_pricing.output_cost_per_1m_tokens == 0.3


def test_load_accounting_config_fetches_openrouter_pricing(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": {
                    "pricing": {
                        "prompt": "0.0000008",
                        "completion": "0.0000024",
                        "input_cache_read": "0.0000002",
                    }
                }
            }

    captured: dict[str, object] = {}

    def fake_get(url: str, *, timeout: float):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llm_cli.httpx, "get", fake_get)

    config = llm_cli._load_accounting_config(
        pricing_config_path=None,
        llm_provider="openrouter",
        llm_model="meta-llama/llama-3.1-70b-instruct",
        input_cost_per_1m_tokens=None,
        output_cost_per_1m_tokens=None,
        cached_input_cost_per_1m_tokens=None,
        measure_energy=False,
        per_document_energy=False,
        electricity_cost_per_kwh=None,
        carbon_kg_per_kwh=None,
        currency=None,
    )

    assert (
        captured["url"]
        == "https://openrouter.ai/api/v1/model/meta-llama/llama-3.1-70b-instruct"
    )
    assert config.pricing_source == "openrouter_models_api"
    assert config.token_pricing.input_cost_per_1m_tokens == 0.8
    assert config.token_pricing.output_cost_per_1m_tokens == 2.4
    assert config.token_pricing.cached_input_cost_per_1m_tokens == 0.2


def test_load_accounting_config_keeps_manual_pricing_over_openrouter_fetch(
    monkeypatch,
) -> None:
    def fail_get(*args, **kwargs):
        raise AssertionError("OpenRouter pricing should not be fetched")

    monkeypatch.setattr(llm_cli.httpx, "get", fail_get)

    config = llm_cli._load_accounting_config(
        pricing_config_path=None,
        llm_provider="openrouter",
        llm_model="meta-llama/llama-3.1-70b-instruct",
        input_cost_per_1m_tokens=1.25,
        output_cost_per_1m_tokens=2.5,
        cached_input_cost_per_1m_tokens=None,
        measure_energy=False,
        per_document_energy=False,
        electricity_cost_per_kwh=None,
        carbon_kg_per_kwh=None,
        currency=None,
    )

    assert config.pricing_source == "cli"
    assert config.token_pricing.input_cost_per_1m_tokens == 1.25
    assert config.token_pricing.output_cost_per_1m_tokens == 2.5


def test_run_llm_benchmark_marks_energy_unavailable_when_tracker_missing(
    monkeypatch,
) -> None:
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    monkeypatch.setattr(
        llm_benchmark.energy,
        "create_energy_tracker",
        lambda config: llm_benchmark.energy.UnavailableEnergyTracker(
            reason="codecarbon_not_installed"
        ),
    )

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        accounting_config=llm_benchmark.BenchmarkAccountingConfig(
            energy_accounting=llm_benchmark.EnergyAccountingConfig(measure_energy=True)
        ),
    )

    assert result["estimated_energy_cost"]["measurement_source"] == "unavailable"
    assert result["estimated_energy_cost"]["reason"] == "codecarbon_not_installed"


def test_run_llm_benchmark_uses_energy_rate_for_run_level_cost(monkeypatch) -> None:
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    class _FakeTracker:
        measurement_source = "measured"

        def start_run(self) -> None:
            return None

        def stop_run(self) -> dict[str, float | str]:
            return {
                "measurement_source": "measured",
                "energy_kwh": 0.5,
                "carbon_kg": 0.2,
            }

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    monkeypatch.setattr(
        llm_benchmark.energy,
        "create_energy_tracker",
        lambda config: _FakeTracker(),
    )

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        accounting_config=llm_benchmark.BenchmarkAccountingConfig(
            energy_accounting=llm_benchmark.EnergyAccountingConfig(
                measure_energy=True,
                electricity_cost_per_kwh=0.4,
                currency="EUR",
            )
        ),
    )

    assert result["estimated_energy_cost"]["energy_kwh"] == 0.5
    assert result["estimated_energy_cost"]["electricity_cost"] == 0.2
    assert result["estimated_energy_cost"]["currency"] == "EUR"


def test_prediction_records_only_include_energy_when_enabled(monkeypatch) -> None:
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    class _FakeTracker:
        def start_run(self) -> None:
            return None

        def stop_run(self) -> dict[str, float | str]:
            return {
                "measurement_source": "measured",
                "energy_kwh": 0.5,
                "carbon_kg": 0.2,
            }

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    monkeypatch.setattr(
        llm_benchmark.energy,
        "create_energy_tracker",
        lambda config: _FakeTracker(),
    )

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        accounting_config=llm_benchmark.BenchmarkAccountingConfig(
            energy_accounting=llm_benchmark.EnergyAccountingConfig(
                measure_energy=True,
                per_document_energy=False,
                electricity_cost_per_kwh=0.4,
                currency="EUR",
            )
        ),
    )

    assert "estimated_energy_cost" not in result["prediction_records"][0]["metadata"]


def test_prediction_records_include_energy_when_per_document_enabled(
    monkeypatch,
) -> None:
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    class _FakeTracker:
        def start_run(self) -> None:
            return None

        def stop_run(self) -> dict[str, float | str]:
            return {
                "measurement_source": "measured",
                "energy_kwh": 0.5,
                "carbon_kg": 0.2,
            }

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    monkeypatch.setattr(
        llm_benchmark.energy,
        "create_energy_tracker",
        lambda config: _FakeTracker(),
    )

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        accounting_config=llm_benchmark.BenchmarkAccountingConfig(
            energy_accounting=llm_benchmark.EnergyAccountingConfig(
                measure_energy=True,
                per_document_energy=True,
                electricity_cost_per_kwh=0.4,
                currency="EUR",
            )
        ),
    )

    assert result["prediction_records"][0]["metadata"]["estimated_energy_cost"] == {
        "measurement_source": "measured",
        "energy_kwh": 0.5,
        "carbon_kg": 0.2,
        "electricity_cost": 0.2,
        "currency": "EUR",
    }


def test_run_llm_benchmark_passes_seed_to_provider(monkeypatch):
    captured: dict[str, object] = {}

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    def fake_get_llm_provider(*, llm_model: str, api_key=None, seed=None):
        captured["llm_model"] = llm_model
        captured["api_key"] = api_key
        captured["seed"] = seed
        return object()

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        llm_seed=123,
    )

    assert captured["seed"] == 123


def test_run_llm_benchmark_uses_filtered_kwargs_and_default_provider_name(
    monkeypatch,
):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakeProvider:
        model_name = "gemini-2.5-flash"
        base_url = None

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    def fake_get_llm_provider(*, llm_model: str):
        assert llm_model == "gemini-2.5-flash"
        return _FakeProvider()

    def fail_if_called(_provider_factory):
        raise AssertionError("_provider_factory_supports_seed should not be called")

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(
        llm_benchmark,
        "_provider_factory_supports_seed",
        fail_if_called,
        raising=False,
    )
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    monkeypatch.setattr(llm_benchmark, "DEFAULT_PROVIDER_NAME", "ollama")

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
    )

    assert result["llm_provider"] == "ollama"


def test_run_llm_benchmark_surfaces_phase2_routing_counts(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {
                "dataset_name": f"phenobert_{dataset}",
                "source": "phenobert",
                "total_documents": 1,
                "total_annotations": 0,
            },
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has scoliosis.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    phase_counts={
                        "phase2b_local_accept_count": 3,
                        "phase2b_deferred_count": 2,
                        "phase2b_no_candidate_skip_count": 1,
                    },
                ),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
    )

    observability = result["prediction_records"][0]["metadata"]["observability"]
    assert observability["phase2b_local_accept_count"] == 3
    assert observability["phase2b_deferred_count"] == 2
    assert observability["phase2b_no_candidate_skip_count"] == 1


def test_run_llm_benchmark_excludes_other_category_from_scored_predictions(
    monkeypatch,
):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {
                "dataset_name": f"phenobert_{dataset}",
                "source": "phenobert",
                "total_documents": 1,
                "total_annotations": 1,
            },
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures. Symptoms began in infancy.",
                    "gold_hpo_terms": [{"id": "HP:0001250", "assertion": "PRESENT"}],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def warmup(self, *, language: str) -> None:
            return None

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001250",
                        label="Seizure",
                        evidence="seizures",
                        assertion="present",
                        category="abnormal",
                    ),
                    LLMPhenotype(
                        term_id="HP:0003593",
                        label="Infantile onset",
                        evidence="in infancy",
                        assertion="other",
                        category="other",
                    ),
                ],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
    )

    assert result["results"][0]["predicted_hpo_ids"] == ["HP:0001250"]
    assert [term["term_id"] for term in result["results"][0]["predicted_terms"]] == [
        "HP:0001250"
    ]
    assert len(result["prediction_records"][0]["annotations"]) == 2


def test_run_llm_benchmark_projects_assertions_to_dataset_granularity(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {
                "dataset_name": f"phenobert_{dataset}",
                "source": "phenobert",
                "total_documents": 1,
                "total_annotations": 2,
            },
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [
                        {"id": "HP:0001250", "assertion": "PRESENT"},
                        {"id": "HP:0000639", "assertion": "PRESENT"},
                    ],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def warmup(self, *, language: str) -> None:
            return None

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001250",
                        label="Seizure",
                        evidence="seizures",
                        assertion="present",
                        category="abnormal",
                    ),
                    LLMPhenotype(
                        term_id="HP:0000639",
                        label="Nystagmus",
                        evidence="nystagmus",
                        assertion="uncertain",
                        category="suspected",
                    ),
                    LLMPhenotype(
                        term_id="HP:0001249",
                        label="Intellectual disability",
                        evidence="normal intelligence",
                        assertion="negated",
                        category="normal",
                    ),
                    LLMPhenotype(
                        term_id="HP:0000365",
                        label="Hearing impairment",
                        evidence="mother has hearing loss",
                        assertion="family_history",
                        category="family_history",
                    ),
                    LLMPhenotype(
                        term_id="HP:0003593",
                        label="Infantile onset",
                        evidence="in infancy",
                        assertion="other",
                        category="other",
                    ),
                ],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        dataset="GeneReviews",
    )

    assert result["results"][0]["predicted_hpo_ids"] == ["HP:0000639", "HP:0001250"]
    assert result["results"][0]["predicted_terms"] == [
        {
            "term_id": "HP:0001250",
            "label": "Seizure",
            "assertion": "PRESENT",
            "evidence": "seizures",
            "category": "abnormal",
        },
        {
            "term_id": "HP:0000639",
            "label": "Nystagmus",
            "assertion": "PRESENT",
            "evidence": "nystagmus",
            "category": "suspected",
        },
    ]
    assert len(result["prediction_records"][0]["annotations"]) == 5


def test_run_llm_benchmark_includes_trace_with_projected_scoring(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {
                "dataset_name": f"phenobert_{dataset}",
                "source": "phenobert",
                "total_documents": 1,
                "total_annotations": 1,
            },
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [{"id": "HP:0001250", "assertion": "PRESENT"}],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def warmup(self, *, language: str) -> None:
            return None

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001250",
                        label="Seizure",
                        evidence="seizures",
                        assertion="present",
                        category="abnormal",
                    ),
                    LLMPhenotype(
                        term_id="HP:0001249",
                        label="Intellectual disability",
                        evidence="normal intelligence",
                        assertion="negated",
                        category="normal",
                    ),
                ],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    trace={
                        "phase1": {
                            "extracted": [
                                {
                                    "phrase": "seizures",
                                    "category": "abnormal",
                                    "actionable": True,
                                },
                                {
                                    "phrase": "normal intelligence",
                                    "category": "normal",
                                    "actionable": True,
                                },
                            ]
                        }
                    },
                ),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        dataset="GeneReviews",
    )

    trace = result["prediction_records"][0]["trace"]
    assert trace["phase1"]["extracted"][0]["phrase"] == "seizures"
    assert trace["phase1"]["extracted"][1]["category"] == "normal"
    assert trace["final_annotations"] == [
        {
            "term_id": "HP:0001250",
            "label": "Seizure",
            "assertion": "present",
            "evidence": "seizures",
            "category": "abnormal",
        },
        {
            "term_id": "HP:0001249",
            "label": "Intellectual disability",
            "assertion": "negated",
            "evidence": "normal intelligence",
            "category": "normal",
        },
    ]
    assert trace["projected_predictions"] == [
        {
            "term_id": "HP:0001250",
            "label": "Seizure",
            "assertion": "PRESENT",
            "evidence": "seizures",
            "category": "abnormal",
        }
    ]


def test_run_llm_benchmark_projects_csc_to_present_only(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        assert dataset == "CSC"
        return {
            "metadata": {
                "dataset_name": "phenobert_CSC",
                "source": "rag_hpo_paper",
                "dataset_namespace": "rag_hpo_paper",
                "total_documents": 1,
                "total_annotations": 1,
            },
            "documents": [
                {
                    "id": "CSC_1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [{"id": "HP:0001250", "assertion": "PRESENT"}],
                    "source_dataset": "CSC",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def warmup(self, *, language: str) -> None:
            return None

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001250",
                        label="Seizure",
                        evidence="seizures",
                        assertion="present",
                        category="abnormal",
                    ),
                    LLMPhenotype(
                        term_id="HP:0001249",
                        label="Intellectual disability",
                        evidence="normal intelligence",
                        assertion="negated",
                        category="normal",
                    ),
                    LLMPhenotype(
                        term_id="HP:0000365",
                        label="Hearing impairment",
                        evidence="mother has hearing loss",
                        assertion="family_history",
                        category="family_history",
                    ),
                    LLMPhenotype(
                        term_id="HP:0000639",
                        label="Nystagmus",
                        evidence="possible nystagmus",
                        assertion="uncertain",
                        category="suspected",
                    ),
                ],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        dataset="CSC",
    )

    assert result["results"][0]["predicted_hpo_ids"] == ["HP:0001250"]
    assert result["results"][0]["predicted_terms"] == [
        {
            "term_id": "HP:0001250",
            "label": "Seizure",
            "assertion": "PRESENT",
            "evidence": "seizures",
            "category": "abnormal",
        }
    ]
    assert (
        result["prediction_records"][0]["trace"]["projection"]["assertion_projection"]
        == llm_benchmark.DATASET_ASSERTION_PROJECTION["CSC"]
    )


def test_run_llm_benchmark_records_failed_documents(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FailingPipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            raise llm_benchmark.LLMPipelinePhaseError(
                "phase1", "Structured extraction failed"
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FailingPipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
    )

    assert result["results"][0]["status"] == "failed"
    assert result["results"][0]["error_phase"] == "phase1"
    assert result["results"][0]["error_message"] == "Structured extraction failed"


def test_run_llm_benchmark_treats_grounded_preprocessing_failures_as_document_failures(
    monkeypatch,
):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
                {
                    "id": "doc-2",
                    "text": "Patient has ataxia.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
            ],
        }

    class _Pipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    def fake_build_grounded_chunks(**kwargs):
        if "seizures" in kwargs["text"]:
            raise RuntimeError("chunking failed")
        return [{"chunk_id": 1, "text": kwargs["text"]}]

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _Pipeline)
    monkeypatch.setattr(
        llm_benchmark,
        "_build_grounded_chunks",
        fake_build_grounded_chunks,
    )

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        llm_internal_mode="whole_document_grounded",
    )

    assert [record["doc_id"] for record in result["results"]] == ["doc-1", "doc-2"]
    assert result["results"][0]["status"] == "failed"
    assert result["results"][0]["error_phase"] == "phase1"
    assert result["results"][0]["error_message"] == "Grounded preprocessing failed"
    assert "error_phase" not in result["results"][1]
    assert "error_message" not in result["results"][1]


def test_run_llm_benchmark_counts_failed_documents_as_metric_misses(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [{"id": "HP:0001250", "assertion": "PRESENT"}],
                    "source_dataset": "GeneReviews",
                },
                {
                    "id": "doc-2",
                    "text": "Patient has ataxia.",
                    "gold_hpo_terms": [{"id": "HP:0001251", "assertion": "PRESENT"}],
                    "source_dataset": "GeneReviews",
                },
            ],
        }

    class _MixedPipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            if "ataxia" in text:
                raise llm_benchmark.LLMPipelinePhaseError(
                    "phase1", "Structured extraction failed"
                )
            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001250",
                        label="Seizure",
                        evidence="seizures",
                        assertion="present",
                        category="abnormal",
                    )
                ],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _MixedPipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
    )

    assertion_micro = result["metrics"]["assertion_aware"]["micro"]
    id_only_micro = result["metrics"]["id_only"]["micro"]

    assert result["cases"] == 2
    assert [record["doc_id"] for record in result["results"]] == ["doc-1", "doc-2"]
    assert result["results"][1]["status"] == "failed"
    assert assertion_micro == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3}
    assert id_only_micro == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3}


def test_run_llm_benchmark_skips_singleton_extraction_groups(monkeypatch):
    captured: dict[str, object] = {}

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _Provider:
        def count_tokens(self, *, system_prompt: str, user_prompt: str):
            return {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12}

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config, extraction_groups=None):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            captured["extraction_groups"] = extraction_groups
            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(
        llm_benchmark,
        "get_llm_provider",
        lambda llm_model: _Provider(),
    )
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    monkeypatch.setattr(
        llm_benchmark,
        "build_extraction_groups",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("build_extraction_groups should not run for fitting notes")
        ),
    )

    llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        llm_internal_mode="whole_document_grounded",
    )

    assert captured["extraction_groups"] is None


def test_run_llm_benchmark_logs_case_progress_at_info(monkeypatch, caplog):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    request_count=2,
                    phase_timings={
                        "phase1_seconds": 0.5,
                        "phase2a_seconds": 0.2,
                        "phase2b_local_seconds": 0.1,
                        "phase2b_llm_seconds": 0.3,
                    },
                    phase_counts={
                        "extracted_phrases": 4,
                        "actionable_phrases": 3,
                        "candidate_sets": 3,
                        "unresolved_phrases": 1,
                        "local_matches": 2,
                        "llm_mapped_phrases": 1,
                        "local_fallbacks": 0,
                    },
                ),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    caplog.set_level("INFO", logger="phentrieve.benchmark.llm_benchmark")

    llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
    )

    assert any(
        "Benchmark document start" in record.message
        and "doc-1" in record.message
        and "1/1" in record.message
        for record in caplog.records
    )
    assert any(
        "Benchmark document complete" in record.message
        and "doc-1" in record.message
        and "requests=2" in record.message
        and "phase1=0.500s" in record.message
        and "phase2b_llm=0.300s" in record.message
        for record in caplog.records
    )


def test_run_llm_benchmark_warms_pipeline_once_per_run(monkeypatch):
    warmup_calls: list[str] = []
    seen_texts: list[str] = []

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
                {
                    "id": "doc-2",
                    "text": "Patient has ataxia.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def warmup(self, *, language: str) -> None:
            warmup_calls.append(language)

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            seen_texts.append(text)
            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        language="en",
    )

    assert warmup_calls == ["en"]
    assert seen_texts == ["Patient has seizures.", "Patient has ataxia."]


def test_run_llm_benchmark_logs_warmup_start_and_end(monkeypatch, caplog):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def warmup(self, *, language: str) -> None:
            assert language == "en"

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)
    caplog.set_level("INFO", logger="phentrieve.benchmark.llm_benchmark")

    llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
    )

    assert any("Benchmark warmup start" in record.message for record in caplog.records)
    assert any(
        "Benchmark warmup complete" in record.message and "elapsed=" in record.message
        for record in caplog.records
    )


def test_run_llm_benchmark_resumes_completed_cases_from_checkpoint(monkeypatch):
    seen_texts: list[str] = []

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": f"phenobert_{dataset}"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [{"id": "HP:0001250", "assertion": "PRESENT"}],
                    "source_dataset": "GeneReviews",
                },
                {
                    "id": "doc-2",
                    "text": "Patient has ataxia.",
                    "gold_hpo_terms": [{"id": "HP:0001251", "assertion": "PRESENT"}],
                    "source_dataset": "GeneReviews",
                },
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            seen_texts.append(text)
            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001251",
                        label="Ataxia",
                        evidence="ataxia",
                        assertion="present",
                    )
                ],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    request_count=2,
                ),
            )

    checkpoint_payload = {
        "status": "running",
        "cases": 2,
        "timing_breakdown": {"wall_clock_seconds": 12.0},
        "results": [
            {
                "case_index": 1,
                "doc_id": "doc-1",
                "source_dataset": "GeneReviews",
                "expected_hpo_ids": ["HP:0001250"],
                "predicted_hpo_ids": ["HP:0001250"],
                "predicted_terms": [
                    {
                        "term_id": "HP:0001250",
                        "label": "Seizure",
                        "assertion": "present",
                        "evidence": "seizures",
                    }
                ],
                "timing_seconds": 1.0,
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "api_calls": 1,
                },
                "estimated_cost": None,
            }
        ],
        "prediction_records": [
            {
                "doc_id": "doc-1",
                "annotations": [],
                "metadata": {
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "api_calls": 1,
                    }
                },
            }
        ],
        "token_usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "api_calls": 1,
        },
    }

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        checkpoint_state=checkpoint_payload,
    )

    assert seen_texts == ["Patient has ataxia."]
    assert [record["doc_id"] for record in result["results"]] == ["doc-1", "doc-2"]
    assert result["token_usage"]["api_calls"] == 3
    assert result["timing_breakdown"]["wall_clock_seconds"] >= 12.0


def test_run_llm_benchmark_temporarily_overrides_prompt_templates_dir(
    tmp_path, monkeypatch
):
    custom_templates_dir = tmp_path / "prompts"
    custom_templates_dir.mkdir()
    original_templates_dir = llm_benchmark.prompt_loader.USER_TEMPLATES_DIR
    captured: dict[str, object] = {}

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {
                "dataset_name": "phenobert_GeneReviews",
                "source": "phenobert",
            },
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            captured["templates_dir_during_run"] = (
                llm_benchmark.prompt_loader.USER_TEMPLATES_DIR
            )
            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(llm_model=config.model, llm_mode=config.mode),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        prompt_templates_dir=str(custom_templates_dir),
    )

    assert captured["templates_dir_during_run"] == custom_templates_dir
    assert llm_benchmark.prompt_loader.USER_TEMPLATES_DIR == original_templates_dir


def test_run_llm_benchmark_retries_failed_checkpoint_cases(monkeypatch):
    seen_texts: list[str] = []

    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
                {
                    "id": "doc-2",
                    "text": "Patient has ataxia.",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                },
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            seen_texts.append(text)
            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    request_count=1,
                ),
            )

    checkpoint_payload = {
        "status": "running",
        "cases": 2,
        "timing_breakdown": {"wall_clock_seconds": 4.0},
        "results": [
            {
                "case_index": 1,
                "doc_id": "doc-1",
                "source_dataset": "GeneReviews",
                "status": "failed",
                "error_phase": "phase1",
                "error_message": "Structured extraction failed",
            }
        ],
        "prediction_records": [
            {
                "case_index": 1,
                "doc_id": "doc-1",
                "status": "failed",
                "error_phase": "phase1",
                "error_message": "Structured extraction failed",
            }
        ],
    }

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        checkpoint_state=checkpoint_payload,
    )

    assert seen_texts == ["Patient has seizures.", "Patient has ataxia."]
    assert [record["doc_id"] for record in result["results"]] == ["doc-1", "doc-2"]
    assert all(record.get("status") != "failed" for record in result["results"])


def test_run_llm_benchmark_cli_writes_prediction_and_metrics_artifacts(
    tmp_path, monkeypatch
):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "summary.json"
    artifacts_dir = tmp_path / "artifacts"

    monkeypatch.setattr(
        llm_cli.llm_benchmark,
        "run_llm_benchmark",
        lambda **kwargs: {
            "cases": 1,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
            "dataset_metadata": {"dataset_name": "phenobert_GeneReviews"},
            "metrics": {
                "assertion_aware": {"micro": {"f1": 1.0}},
                "id_only": {"micro": {"f1": 1.0}},
            },
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "timing_breakdown": {"wall_clock_seconds": 1.0},
            "estimated_cost": {
                "input_cost": 0.1,
                "output_cost": 0.2,
                "total_cost": 0.3,
            },
            "prediction_records": [
                {
                    "doc_id": "doc-1",
                    "annotations": [],
                    "trace": {
                        "phase1": {"extracted": [{"phrase": "seizures"}]},
                        "projected_predictions": [
                            {
                                "term_id": "HP:0001250",
                                "assertion": "PRESENT",
                            }
                        ],
                    },
                    "metadata": {
                        "model": kwargs["llm_model"],
                        "mode": kwargs["llm_mode"],
                        "token_usage": {"total_tokens": 15},
                    },
                }
            ],
            "results": [{"doc_id": "doc-1"}],
        },
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_path=str(output_path),
        artifacts_dir=str(artifacts_dir),
    )

    assert result["output_path"] == str(output_path)
    assert (artifacts_dir / "predictions" / "two_phase" / "case_doc-1.json").exists()
    assert (artifacts_dir / "traces" / "two_phase" / "case_doc-1.json").exists()
    assert (artifacts_dir / "metrics" / "benchmark_two_phase.json").exists()


def test_run_llm_benchmark_cli_writes_ontology_metrics_artifact(tmp_path, monkeypatch):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "summary.json"
    artifacts_dir = tmp_path / "artifacts"

    monkeypatch.setattr(
        llm_cli.llm_benchmark,
        "run_llm_benchmark",
        lambda **kwargs: {
            "cases": 1,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
            "dataset_metadata": {"dataset_name": "phenobert_GeneReviews"},
            "metrics": {
                "assertion_aware": {
                    "micro": {"f1": 1.0},
                    "ontology_metrics": {
                        "soft": {"micro": {"f1": 0.75}},
                        "partial": {"micro": {"f1": 0.5}},
                    },
                },
                "id_only": {"micro": {"f1": 1.0}},
            },
            "prediction_records": [],
            "results": [{"doc_id": "doc-1"}],
        },
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_path=str(output_path),
        artifacts_dir=str(artifacts_dir),
    )

    metrics_path = Path(result["metrics_path"])
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "ontology_metrics" in metrics_payload["assertion_aware_metrics"]
    assert (
        metrics_payload["assertion_aware_metrics"]["ontology_metrics"]["soft"]["micro"][
            "f1"
        ]
        == 0.75
    )


def test_run_llm_benchmark_cli_sanitizes_artifact_filenames(tmp_path, monkeypatch):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "summary.json"
    artifacts_dir = tmp_path / "artifacts"

    monkeypatch.setattr(
        llm_cli.llm_benchmark,
        "run_llm_benchmark",
        lambda **kwargs: {
            "cases": 1,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
            "dataset_metadata": {"dataset_name": "phenobert_GeneReviews"},
            "metrics": {
                "assertion_aware": {"micro": {"f1": 1.0}},
                "id_only": {"micro": {"f1": 1.0}},
            },
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "timing_breakdown": {"wall_clock_seconds": 1.0},
            "estimated_cost": None,
            "prediction_records": [
                {
                    "case_index": 7,
                    "doc_id": "folder/unsafe doc:id",
                    "annotations": [],
                    "trace": {"phase1": {"extracted": []}},
                }
            ],
            "results": [{"doc_id": "folder/unsafe doc:id"}],
        },
    )

    llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_path=str(output_path),
        artifacts_dir=str(artifacts_dir),
    )

    prediction_path = (
        artifacts_dir / "predictions" / "two_phase" / "case_7_folder_unsafe_doc_id.json"
    )
    trace_path = (
        artifacts_dir / "traces" / "two_phase" / "case_7_folder_unsafe_doc_id.json"
    )

    assert prediction_path.exists()
    assert trace_path.exists()
    assert not (artifacts_dir / "predictions" / "two_phase" / "folder").exists()


def test_run_llm_benchmark_cli_writes_checkpoint_snapshot(tmp_path, monkeypatch):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "summary.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    artifacts_dir = tmp_path / "artifacts"
    progress_events: list[dict[str, object]] = []

    def fake_run_llm_benchmark(**kwargs):
        progress_callback = kwargs["progress_callback"]
        assert kwargs["checkpoint_state"] is None
        partial_payload = {
            "status": "running",
            "cases": 2,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
            "dataset_metadata": {"dataset_name": "phenobert_GeneReviews"},
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "api_calls": 2,
            },
            "timing_breakdown": {"wall_clock_seconds": 1.0},
            "prediction_records": [
                {
                    "doc_id": "doc-1",
                    "annotations": [],
                    "metadata": {
                        "token_usage": {"total_tokens": 15, "api_calls": 2},
                    },
                }
            ],
            "results": [{"doc_id": "doc-1"}],
        }
        progress_callback(partial_payload)
        progress_events.append(partial_payload)
        return {
            **partial_payload,
            "status": "completed",
            "metrics": {
                "assertion_aware": {"micro": {"f1": 1.0}},
                "id_only": {"micro": {"f1": 1.0}},
            },
            "estimated_cost": None,
        }

    monkeypatch.setattr(
        llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        artifacts_dir=str(artifacts_dir),
    )

    assert progress_events
    assert checkpoint_path.exists()
    checkpoint_payload = checkpoint_path.read_text(encoding="utf-8")
    assert '"status": "completed"' in checkpoint_payload
    assert result["checkpoint_path"] == str(checkpoint_path)


def test_run_llm_benchmark_cli_rejects_mismatched_checkpoint(tmp_path):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "test_file": str(test_file),
                "dataset": "GeneReviews",
                "llm_model": "gemini-2.5-pro",
                "llm_mode": "two_phase",
                "language": "en",
                "prompt_templates_dir": None,
                "requested_doc_ids": None,
                "status": "running",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Checkpoint does not match"):
        llm_cli.run_llm_benchmark_cli(
            test_file=str(test_file),
            llm_model="gemini-2.5-flash",
            checkpoint_path=str(checkpoint_path),
            output_path=str(tmp_path / "summary.json"),
        )


def test_run_llm_benchmark_cli_resumes_checkpoint_without_ontology_keys(
    tmp_path, monkeypatch
):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.json"
    output_path = tmp_path / "summary.json"
    artifacts_dir = tmp_path / "artifacts"
    checkpoint_path.write_text(
        json.dumps(
            {
                "test_file": str(test_file),
                "dataset": "GeneReviews",
                "llm_provider": None,
                "llm_model": "gemini-2.5-flash",
                "llm_base_url": None,
                "llm_timeout_seconds": None,
                "llm_seed": None,
                "llm_mode": "two_phase",
                "llm_internal_mode": "whole_document_grounded",
                "language": "en",
                "capture_phase1_debug": False,
                "prompt_templates_dir": None,
                "requested_doc_ids": None,
                "status": "running",
                "prediction_records": [],
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    captured_checkpoint_state: dict[str, object] = {}

    def fake_run_llm_benchmark(**kwargs):
        captured_checkpoint_state.update(kwargs["checkpoint_state"])
        return {
            "status": "completed",
            "cases": 0,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
            "dataset_metadata": {"dataset_name": "phenobert_GeneReviews"},
            "metrics": {
                "assertion_aware": {"micro": {"f1": 0.0}},
                "id_only": {"micro": {"f1": 0.0}},
            },
            "prediction_records": [],
            "results": [],
        }

    monkeypatch.setattr(
        llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark
    )

    llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        artifacts_dir=str(artifacts_dir),
    )

    assert captured_checkpoint_state["status"] == "running"
    assert "ontology_aware_metrics" not in captured_checkpoint_state


def test_run_llm_benchmark_cli_overwrites_existing_output_without_checkpoint(
    tmp_path, monkeypatch
):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "summary.json"
    output_path.write_text(
        json.dumps(
            {
                "test_file": str(test_file),
                "dataset": "GeneReviews",
                "llm_model": "gemini-2.5-pro",
                "llm_mode": "two_phase",
                "llm_internal_mode": "whole_document_grounded",
                "language": "en",
                "prompt_templates_dir": None,
                "requested_doc_ids": None,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    def fake_run_llm_benchmark(**kwargs):
        assert kwargs["checkpoint_state"] is None
        return {
            "status": "completed",
            "cases": 0,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "llm_internal_mode": kwargs["llm_internal_mode"],
            "dataset": kwargs["dataset"],
            "language": kwargs["language"],
            "prompt_templates_dir": kwargs["prompt_templates_dir"],
            "requested_doc_ids": None,
            "dataset_metadata": {"dataset_name": "phenobert_GeneReviews"},
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0,
            },
            "timing_breakdown": {
                "wall_clock_seconds": 0.0,
                "avg_seconds_per_case": 0.0,
            },
            "prediction_records": [],
            "results": [],
            "metrics": {
                "assertion_aware": {"micro": {"f1": 0.0}},
                "id_only": {"micro": {"f1": 0.0}},
            },
            "estimated_cost": None,
        }

    monkeypatch.setattr(
        llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_path=str(output_path),
    )

    assert result["llm_model"] == "gemini-2.5-flash"
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["llm_model"]
        == "gemini-2.5-flash"
    )


def test_assertion_distribution_surfaces_dropped_deliverable():
    """The present-only projection drops negated/family findings from scoring;
    the distribution report must count them so the deliverable stays visible."""
    import types

    def term(assertion: str):
        return types.SimpleNamespace(
            term_id="HP:0000001", label="x", assertion=assertion, evidence="e"
        )

    pipeline_result = types.SimpleNamespace(
        terms=[term("present"), term("present"), term("negated"), term("uncertain")],
        family_history_findings=[term("present")],
    )

    dist = llm_benchmark._assertion_distribution(pipeline_result, dataset="GeneReviews")

    assert dist["proband_by_assertion"] == {"present": 2, "negated": 1, "uncertain": 1}
    # GeneReviews keeps present+uncertain, drops negated.
    assert dist["proband_scored"] == 3
    assert dist["proband_dropped_by_projection"] == 1
    assert dist["family_history_findings"] == 1


def test_aggregate_assertion_distribution_sums_records():
    records = [
        {
            "assertion_distribution": {
                "proband_by_assertion": {"present": 2, "negated": 1},
                "proband_scored": 2,
                "proband_dropped_by_projection": 1,
                "family_history_findings": 1,
            }
        },
        {
            "assertion_distribution": {
                "proband_by_assertion": {"present": 3},
                "proband_scored": 3,
                "proband_dropped_by_projection": 0,
                "family_history_findings": 0,
            }
        },
        # A checkpoint-restored record from before the field existed: no block.
        {"doc_id": "legacy-doc"},
    ]

    agg = llm_benchmark._aggregate_assertion_distribution(records)

    assert agg["proband_by_assertion"] == {"present": 5, "negated": 1}
    assert agg["proband_scored"] == 5
    assert agg["proband_dropped_by_projection"] == 1
    assert agg["family_history_findings"] == 1
    # Undercount is visible: 2 of 3 records contributed.
    assert agg["documents_counted"] == 2
    assert agg["documents_total"] == 3
