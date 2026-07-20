from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from phentrieve.benchmark import llm_benchmark, llm_cli
from phentrieve.benchmark.result_store import create_run_layout
from phentrieve.benchmark.run_identity import (
    RetrievalAssetIdentity,
    build_dataset_identity,
    build_run_fingerprints,
)
from phentrieve.llm.prompts.identity import build_prompt_bundle_identity

pytestmark = pytest.mark.unit


def _test_fingerprints(test_file: Path) -> dict[str, str]:
    fingerprints = build_run_fingerprints(
        build_dataset_identity(test_file, "GeneReviews"),
        build_prompt_bundle_identity("two_phase", "en"),
        {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "base_url": None,
            "base_url_behavior_sha256": None,
            "seed": None,
            "timeout_seconds": None,
            "internal_mode": "whole_document_grounded",
        },
        RetrievalAssetIdentity(
            asset_type="single_vector",
            embedding_model="BAAI/bge-m3",
            hpo_version="v2026-06-23",
            manifest_sha256="a" * 64,
        ),
    )
    return {
        "execution_fingerprint": fingerprints.execution_sha256,
        "scoring_fingerprint": fingerprints.scoring_sha256,
    }


@pytest.fixture(autouse=True)
def stub_grounded_chunks(monkeypatch):
    monkeypatch.setattr(
        llm_benchmark,
        "_build_grounded_chunks",
        lambda **kwargs: [{"chunk_id": 1, "text": kwargs["text"]}],
    )
    monkeypatch.setattr(
        llm_cli,
        "load_retrieval_asset_identity",
        lambda _data_dir=None: RetrievalAssetIdentity(
            asset_type="single_vector",
            embedding_model="BAAI/bge-m3",
            hpo_version="v2026-06-23",
            manifest_sha256="a" * 64,
        ),
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
    output_dir = tmp_path / "results"

    mock_setup_logging = mocker.patch("phentrieve.benchmark.llm_cli.setup_logging_cli")
    monkeypatch.setattr(
        llm_cli.llm_benchmark,
        "run_llm_benchmark",
        lambda **kwargs: {
            "status": "completed",
            "cases": 0,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
        },
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_dir=str(output_dir),
    )

    mock_setup_logging.assert_called_once_with(debug=False)
    run_dir = Path(result["run_dir"])
    assert run_dir.is_relative_to(output_dir)
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "manifest.json").exists()


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
    output_dir = tmp_path / "results"
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
            "status": "completed",
            "cases": 0,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
        }

    monkeypatch.setattr(
        llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark
    )

    llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        pricing_config=str(pricing_path),
        input_cost_per_1m_tokens=0.9,
        output_dir=str(output_dir),
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
    assert [record["doc_id"] for record in result["case_records"]] == [
        "doc-1",
        "doc-2",
    ]
    assert {
        record["doc_id"]: record["status"] for record in result["case_records"]
    } == {"doc-1": "complete", "doc-2": "complete"}
    term_ids_by_doc = {
        doc_id: sorted(
            term["hpo_id"]
            for term in result["term_records"]
            if term["doc_id"] == doc_id
        )
        for doc_id in ("doc-1", "doc-2")
    }
    assert term_ids_by_doc == {"doc-1": ["HP:0001250"], "doc-2": ["HP:0001251"]}


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
    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        llm_cli.llm_benchmark,
        "run_llm_benchmark",
        lambda **kwargs: {
            "status": "completed",
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
                            {"term_id": "HP:0001250", "assertion": "PRESENT"}
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
            "term_records": [],
            "case_records": [{"doc_id": "doc-1", "status": "complete"}],
        },
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_dir=str(output_dir),
    )

    run_dir = Path(result["run_dir"])
    assert (run_dir / "predictions" / "two_phase" / "case_doc-1.json").exists()
    assert (run_dir / "traces" / "two_phase" / "case_doc-1.json").exists()
    assert (run_dir / "metrics" / "benchmark_two_phase.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["llm_predictions"]["path"] == "predictions/two_phase"
    assert manifest["artifacts"]["llm_traces"]["path"] == "traces/two_phase"
    assert (
        manifest["artifacts"]["metrics"]["path"] == "metrics/benchmark_two_phase.json"
    )
    checkpoint = run_dir / "checkpoint.json"
    checkpoint_entry = manifest["artifacts"]["checkpoint:checkpoint.json"]
    assert (
        checkpoint_entry["sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "completed"


def test_run_llm_benchmark_cli_writes_ontology_metrics_artifact(tmp_path, monkeypatch):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        llm_cli.llm_benchmark,
        "run_llm_benchmark",
        lambda **kwargs: {
            "status": "completed",
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
            "term_records": [],
            "case_records": [{"doc_id": "doc-1", "status": "complete"}],
        },
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_dir=str(output_dir),
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
    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        llm_cli.llm_benchmark,
        "run_llm_benchmark",
        lambda **kwargs: {
            "status": "completed",
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
            "term_records": [],
            "case_records": [{"doc_id": "folder/unsafe doc:id", "status": "complete"}],
        },
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_dir=str(output_dir),
    )

    run_dir = Path(result["run_dir"])
    prediction_path = (
        run_dir / "predictions" / "two_phase" / "case_7_folder_unsafe_doc_id.json"
    )
    trace_path = run_dir / "traces" / "two_phase" / "case_7_folder_unsafe_doc_id.json"

    assert prediction_path.exists()
    assert trace_path.exists()
    assert not (run_dir / "predictions" / "two_phase" / "folder").exists()


def test_run_llm_benchmark_cli_writes_partial_manifest_during_checkpoints(
    tmp_path, monkeypatch
):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "results"
    captured_manifests: list[dict[str, object]] = []

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
                    "metadata": {"token_usage": {"total_tokens": 15, "api_calls": 2}},
                }
            ],
            "results": [{"doc_id": "doc-1"}],
        }
        progress_callback(partial_payload)
        manifest_path = next(output_dir.rglob("manifest.json"))
        captured_manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        return {
            **partial_payload,
            "status": "completed",
            "metrics": {
                "assertion_aware": {"micro": {"f1": 1.0}},
                "id_only": {"micro": {"f1": 1.0}},
            },
            "estimated_cost": None,
            "term_records": [],
            "case_records": [{"doc_id": "doc-1", "status": "complete"}],
        }

    monkeypatch.setattr(
        llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark
    )

    result = llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_dir=str(output_dir),
    )

    assert captured_manifests[0]["status"] == "partial"
    assert captured_manifests[0]["schema_version"] == 2
    assert (
        captured_manifests[0]["execution_fingerprint"]
        == result["execution_fingerprint"]
    )
    assert "terms" not in captured_manifests[0]["counts"]
    assert "cases" not in captured_manifests[0]["counts"]
    run_dir = Path(result["run_dir"])
    checkpoint_path = run_dir / "checkpoint.json"
    assert checkpoint_path.exists()
    assert '"status": "completed"' in checkpoint_path.read_text(encoding="utf-8")
    final_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["status"] == "complete"
    assert final_manifest["counts"]["terms"] == 0
    assert final_manifest["counts"]["cases"] == 1


def test_run_llm_benchmark_cli_rejects_mismatched_checkpoint(tmp_path) -> None:
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "results"

    run_layout = create_run_layout(
        output_dir, "llm", "cases", "gemini-2.5-flash", run_id="fixed-run"
    )
    (run_layout.run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "test_file": str(test_file),
                "dataset_sha256": llm_cli.sha256_path(test_file),
                "accounting_config": llm_benchmark.BenchmarkAccountingConfig().model_dump(
                    mode="json"
                ),
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

    with pytest.raises(ValueError, match="execution fingerprint mismatch"):
        llm_cli.run_llm_benchmark_cli(
            test_file=str(test_file),
            llm_model="gemini-2.5-flash",
            output_dir=str(output_dir),
            run_id="fixed-run",
            overwrite=True,
        )


def test_checkpoint_requires_matching_execution_and_scoring_fingerprints(
    tmp_path,
) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "status": "running",
                "execution_fingerprint": "old",
                "scoring_fingerprint": "score",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="execution fingerprint mismatch"):
        llm_cli._load_checkpoint_payload(
            path=path,
            execution_fingerprint="new",
            scoring_fingerprint="score",
            allow_completed=True,
        )
    path.write_text(
        json.dumps(
            {
                "status": "running",
                "execution_fingerprint": "exec",
                "scoring_fingerprint": "old",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scoring fingerprint mismatch"):
        llm_cli._load_checkpoint_payload(
            path=path,
            execution_fingerprint="exec",
            scoring_fingerprint="new",
            allow_completed=True,
        )


def test_persisted_payload_sanitizes_nested_base_url_credentials() -> None:
    sanitized = llm_cli._sanitize_persisted_base_urls(
        {"llm_base_url": "https://user:secret@example.test:8443/api?token=x#frag"}
    )
    assert sanitized == {"llm_base_url": "https://example.test:8443/api"}


def test_producer_identity_is_anchored_to_package_repository(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    identity = llm_cli._build_producer_identity()
    repository = Path(llm_cli.__file__).resolve().parents[2]
    git_executable = shutil.which("git")
    assert git_executable is not None
    expected = subprocess.run(  # noqa: S603 - executable resolved by shutil.which
        [git_executable, "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    assert identity["commit"] == expected
    assert identity["provenance_status"] == "resolved"


def test_runner_rejects_provider_runtime_identity_mismatch(monkeypatch) -> None:
    from phentrieve.llm.providers.base import ResolvedLLMProviderRequest

    monkeypatch.setattr(
        llm_benchmark,
        "load_benchmark_data",
        lambda *args, **kwargs: {"metadata": {}, "documents": []},
    )
    provider = type(
        "Provider",
        (),
        {
            "provider_name": "openai",
            "model_name": "different",
            "base_url": "https://example.test/v1",
        },
    )()
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda **kwargs: provider)
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        llm_benchmark.run_llm_benchmark(
            test_file="unused.json",
            llm_provider="openai",
            llm_model="expected",
            llm_base_url="https://example.test/v1",
            _resolved_provider_request=ResolvedLLMProviderRequest(
                provider="openai", model="expected", base_url="https://example.test/v1"
            ),
        )


def test_run_llm_benchmark_cli_resumes_checkpoint_without_ontology_keys(
    tmp_path, monkeypatch
):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "results"

    run_layout = create_run_layout(
        output_dir, "llm", "cases", "gemini-2.5-flash", run_id="fixed-run"
    )
    (run_layout.run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "test_file": str(test_file),
                "dataset_sha256": llm_cli.sha256_path(test_file),
                "accounting_config": llm_benchmark.BenchmarkAccountingConfig().model_dump(
                    mode="json"
                ),
                "dataset": "GeneReviews",
                "llm_provider": "gemini",
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
                **_test_fingerprints(test_file),
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
            "term_records": [],
            "case_records": [],
        }

    monkeypatch.setattr(
        llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark
    )

    llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_dir=str(output_dir),
        run_id="fixed-run",
        overwrite=True,
    )

    assert captured_checkpoint_state["status"] == "running"
    assert "ontology_aware_metrics" not in captured_checkpoint_state


def test_checkpoint_identity_includes_dataset_and_accounting_fingerprints(tmp_path):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    accounting = llm_benchmark.BenchmarkAccountingConfig(
        token_pricing=llm_benchmark.TokenPricingConfig(
            input_cost_per_1m_tokens=1.25,
        )
    )

    identity = llm_cli._build_checkpoint_identity(
        test_file_path=test_file,
        dataset_sha256="dataset-hash",
        accounting_config=accounting,
        dataset="CSC",
        resolved_provider="openrouter",
        resolved_model="google/model",
        resolved_base_url="https://openrouter.ai/api/v1",
        llm_timeout_seconds=None,
        llm_seed=None,
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        language="en",
        capture_phase1_debug=False,
        ontology_aware_metrics=False,
        ontology_semantic_floor=0.3,
        ontology_similarity_formula="hybrid",
        prompt_templates_dir=None,
        doc_ids=None,
    )

    assert identity["dataset_sha256"] == "dataset-hash"
    assert (
        identity["accounting_config"]["token_pricing"]["input_cost_per_1m_tokens"]
        == 1.25
    )


def test_write_legacy_llm_artifacts_preserves_deprecated_paths(tmp_path):
    payload = {
        "llm_mode": "two_phase",
        "llm_model": "google/model",
        "dataset": "CSC",
        "cases": 0,
        "prediction_records": [],
        "metrics": {"micro": {"f1": 1.0}},
    }
    output_path = tmp_path / "legacy" / "summary.json"
    checkpoint_path = tmp_path / "legacy" / "checkpoint.json"
    artifacts_dir = tmp_path / "legacy-artifacts"

    llm_cli._write_legacy_artifacts(
        payload=payload,
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        artifacts_dir=str(artifacts_dir),
    )

    assert output_path.is_file()
    assert checkpoint_path.is_file()
    assert (artifacts_dir / "metrics" / "benchmark_two_phase.json").is_file()


def test_run_llm_benchmark_cli_resumes_checkpoint_missing_capture_phase1_debug_key(
    tmp_path, monkeypatch
):
    """Checkpoints written before capture_phase1_debug was persisted must still
    match a same-config resume via CHECKPOINT_DEFAULTS, not just checkpoints
    written by the current code (which always includes the key)."""
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "results"

    run_layout = create_run_layout(
        output_dir, "llm", "cases", "gemini-2.5-flash", run_id="fixed-run"
    )
    (run_layout.run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "test_file": str(test_file),
                "dataset_sha256": llm_cli.sha256_path(test_file),
                "accounting_config": llm_benchmark.BenchmarkAccountingConfig().model_dump(
                    mode="json"
                ),
                "dataset": "GeneReviews",
                "llm_provider": "gemini",
                "llm_model": "gemini-2.5-flash",
                "llm_base_url": None,
                "llm_timeout_seconds": None,
                "llm_seed": None,
                "llm_mode": "two_phase",
                "llm_internal_mode": "whole_document_grounded",
                "language": "en",
                "prompt_templates_dir": None,
                "requested_doc_ids": None,
                "status": "running",
                "prediction_records": [],
                "results": [],
                **_test_fingerprints(test_file),
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
            "term_records": [],
            "case_records": [],
        }

    monkeypatch.setattr(
        llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark
    )

    llm_cli.run_llm_benchmark_cli(
        test_file=str(test_file),
        llm_model="gemini-2.5-flash",
        output_dir=str(output_dir),
        run_id="fixed-run",
        overwrite=True,
    )

    assert captured_checkpoint_state["status"] == "running"
    assert "capture_phase1_debug" not in captured_checkpoint_state


def test_run_llm_benchmark_cli_reuses_completed_checkpoint_when_overwriting_existing_run(
    tmp_path, monkeypatch
):
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "results"

    run_layout = create_run_layout(
        output_dir, "llm", "cases", "gemini-2.5-flash", run_id="fixed-run"
    )
    (run_layout.run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "test_file": str(test_file),
                "dataset_sha256": llm_cli.sha256_path(test_file),
                "accounting_config": llm_benchmark.BenchmarkAccountingConfig().model_dump(
                    mode="json"
                ),
                "dataset": "GeneReviews",
                "llm_provider": "gemini",
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
                "status": "completed",
                "prediction_records": [],
                "results": [],
                **_test_fingerprints(test_file),
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_llm_benchmark(**kwargs):
        captured.update(kwargs)
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
            "term_records": [],
            "case_records": [],
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
        output_dir=str(output_dir),
        run_id="fixed-run",
        overwrite=True,
    )

    assert result["llm_model"] == "gemini-2.5-flash"
    assert captured["checkpoint_state"]["status"] == "completed"
    assert (
        json.loads(run_layout.summary_path.read_text(encoding="utf-8"))["llm_model"]
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


def test_build_terms_and_cases_classifies_tp_fp_fn_and_prefers_gold_labels():
    documents = [
        {
            "id": "doc-1",
            "gold_hpo_terms": [
                {"id": "HP:0001250", "label": "Seizure", "assertion": "PRESENT"},
                {"id": "HP:0001251", "label": "Ataxia", "assertion": "PRESENT"},
            ],
        }
    ]
    results = [
        {
            "case_index": 1,
            "doc_id": "doc-1",
            "expected_hpo_ids": ["HP:0001250", "HP:0001251"],
            "predicted_hpo_ids": ["HP:0001250", "HP:0001252"],
            "predicted_terms": [
                {
                    "term_id": "HP:0001250",
                    "label": "Seizure disorder",
                    "assertion": "PRESENT",
                    "evidence": "seizures",
                    "category": "neurological",
                },
                {
                    "term_id": "HP:0001252",
                    "label": "Hypotonia",
                    "assertion": "PRESENT",
                    "evidence": "low tone",
                    "category": "neurological",
                },
            ],
            "timing_seconds": 1.5,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "estimated_cost": {"total_cost": 0.01},
            "partial_failure_counts": {
                "phase1_completed_groups": 1,
                "phase1_failed_groups": 0,
                "phase1_partial_failures": 0,
            },
        }
    ]

    terms, cases = llm_benchmark._build_terms_and_cases(documents, results)

    terms_by_id = {term["hpo_id"]: term for term in terms}
    assert terms_by_id["HP:0001250"]["outcome"] == "tp"
    assert terms_by_id["HP:0001250"]["label"] == "Seizure"
    assert terms_by_id["HP:0001251"]["outcome"] == "fn"
    assert terms_by_id["HP:0001251"]["is_predicted"] is False
    assert terms_by_id["HP:0001252"]["outcome"] == "fp"
    assert terms_by_id["HP:0001252"]["label"] == "Hypotonia"
    assert terms_by_id["HP:0001252"]["evidence"] == "low tone"
    assert terms_by_id["HP:0001252"]["category"] == "neurological"

    assert len(cases) == 1
    assert cases[0]["doc_id"] == "doc-1"
    assert cases[0]["metrics"] == {"tp": 1, "fp": 1, "fn": 1}
    assert cases[0]["timing_seconds"] == 1.5
    assert cases[0]["status"] == "complete"


def test_build_terms_and_cases_marks_failed_documents():
    documents = [
        {
            "id": "doc-1",
            "gold_hpo_terms": [
                {"id": "HP:0001250", "label": "Seizure", "assertion": "PRESENT"}
            ],
        }
    ]
    results = [
        {
            "case_index": 1,
            "doc_id": "doc-1",
            "status": "failed",
            "error_phase": "phase1",
            "error_message": "Structured extraction failed",
        }
    ]

    terms, cases = llm_benchmark._build_terms_and_cases(documents, results)

    assert len(terms) == 1
    assert terms[0]["hpo_id"] == "HP:0001250"
    assert terms[0]["outcome"] == "fn"
    assert cases[0]["status"] == "failed"
    assert cases[0]["expected_hpo_ids"] == ["HP:0001250"]
    assert cases[0]["predicted_hpo_ids"] == []
    assert cases[0]["metrics"] == {"tp": 0, "fp": 0, "fn": 1}


def test_run_llm_benchmark_returns_term_and_case_records(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [
                        {"id": "HP:0001250", "label": "Seizure", "assertion": "PRESENT"}
                    ],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            return LLMExtractionResult(
                terms=[
                    LLMPhenotype(
                        term_id="HP:0001250",
                        label="Seizure",
                        evidence="seizures",
                        assertion="present",
                    )
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

    assert result["term_records"] == [
        {
            "doc_id": "doc-1",
            "hpo_id": "HP:0001250",
            "label": "Seizure",
            "is_gold": True,
            "is_predicted": True,
            "gold_assertion": "PRESENT",
            "predicted_assertion": "PRESENT",
            "outcome": "tp",
            "evidence": "seizures",
            "category": None,
        }
    ]
    assert result["case_records"][0]["doc_id"] == "doc-1"
    assert result["case_records"][0]["metrics"] == {"tp": 1, "fp": 0, "fn": 0}
    assert result["case_records"][0]["status"] == "complete"


def test_derive_run_status_reflects_failed_documents() -> None:
    """``run_llm_benchmark`` reports ``completed`` even when documents failed."""
    assert llm_cli._derive_run_status([]) == "complete"
    assert llm_cli._derive_run_status([{"status": "complete"}]) == "complete"
    assert (
        llm_cli._derive_run_status([{"status": "complete"}, {"status": "failed"}])
        == "partial"
    )
    assert (
        llm_cli._derive_run_status([{"status": "failed"}, {"status": "failed"}])
        == "failed"
    )


def test_prompt_template_edits_change_the_checkpoint_identity(tmp_path) -> None:
    """A checkpoint must not be resumable under different prompt templates.

    Only the templates *directory* used to be part of the checkpoint identity,
    so editing a template in place left the identity unchanged and let a resume
    merge old-prompt and new-prompt document outputs into one set of metrics.
    """
    templates = tmp_path / "prompts"
    templates.mkdir()
    template_file = templates / "extract.yaml"
    template_file.write_text("prompt: original\n", encoding="utf-8")

    before = llm_cli._prompt_templates_sha256(str(templates))
    template_file.write_text("prompt: edited\n", encoding="utf-8")
    after = llm_cli._prompt_templates_sha256(str(templates))

    assert before["user"] is not None
    assert before["user"] != after["user"]

    assert not llm_cli._checkpoint_matches_run(
        payload={"prompt_templates_sha256": before},
        current_run={"prompt_templates_sha256": after},
    )
    assert llm_cli._checkpoint_matches_run(
        payload={"prompt_templates_sha256": before},
        current_run={"prompt_templates_sha256": before},
    )


def test_checkpoint_without_prompt_hash_stays_resumable(tmp_path) -> None:
    """Checkpoints predating prompt hashing must not be invalidated on upgrade.

    They carry no evidence about the templates they ran under, so they keep the
    behaviour they were written with. A checkpoint that *does* record a hash is
    still compared strictly.
    """
    templates = tmp_path / "prompts"
    templates.mkdir()
    (templates / "extract.yaml").write_text("prompt: original\n", encoding="utf-8")
    current = llm_cli._prompt_templates_sha256(str(templates))

    # Legacy payload: key absent entirely -> unverifiable -> still resumable.
    assert llm_cli._checkpoint_matches_run(
        payload={"llm_model": "m"},
        current_run={"llm_model": "m", "prompt_templates_sha256": current},
    )
    # Recorded payload: key present and different -> rejected.
    assert not llm_cli._checkpoint_matches_run(
        payload={"llm_model": "m", "prompt_templates_sha256": {"user": "stale"}},
        current_run={"llm_model": "m", "prompt_templates_sha256": current},
    )
