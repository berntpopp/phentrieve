from __future__ import annotations

from pathlib import Path

import pytest

from phentrieve.benchmark import llm_benchmark, llm_cli

pytestmark = pytest.mark.unit


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

        def run(self, *, text, config):
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

        def run(self, *, text, config):
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

        def run(self, *, text, config):
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

        def run(self, *, text, config):
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

        def run(self, *, text, config):
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
    assert (artifacts_dir / "metrics" / "benchmark_two_phase.json").exists()
