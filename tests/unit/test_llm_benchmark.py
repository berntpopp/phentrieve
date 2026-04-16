from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert result["timing_breakdown"]["wall_clock_seconds"] >= 0.0
    assert result["estimated_cost"]["input_cost"] > 0.0
    assert result["estimated_cost"]["output_cost"] > 0.0
    assert result["prediction_records"][0]["doc_id"] == "doc-1"
    assert (
        result["prediction_records"][0]["metadata"]["token_usage"]["total_tokens"]
        == 150
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
    assert (artifacts_dir / "predictions" / "two_phase" / "doc-1.json").exists()
    assert (artifacts_dir / "traces" / "two_phase" / "doc-1.json").exists()
    assert (artifacts_dir / "metrics" / "benchmark_two_phase.json").exists()


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
