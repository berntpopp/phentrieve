"""
Integration tests for complete benchmark workflow with new metrics.

Tests the full evaluation pipeline:
- Running benchmarks with new MTEB-aligned metrics
- Bootstrap CI calculation for all metrics
- Model comparison with significance testing
- Result saving and loading
- Metric consistency across pipeline

Note: These tests require ChromaDB indexes to be built.
Tests will automatically skip if indexes are not available.
To build indexes: phentrieve index build --model sentence-transformers/LaBSE
"""

import json
import tempfile
from pathlib import Path

import pytest

from phentrieve.benchmark.extraction_benchmark import (
    ExtractionBenchmark,
    ExtractionConfig,
)
from phentrieve.benchmark.llm_cli import run_llm_benchmark_cli
from phentrieve.evaluation.runner import compare_models, run_evaluation
from phentrieve.evaluation.statistics import (
    compare_models_with_significance,
)
from phentrieve.llm.types import ExtractionGroup


@pytest.fixture
def temp_results_dir():
    """Create temporary directory for benchmark results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_test_data_tiny(benchmark_data_dir):
    """Path to tiny benchmark dataset."""
    return benchmark_data_dir / "german" / "tiny_v1.json"


@pytest.fixture
def available_model() -> str:
    """Get first available model from ChromaDB or skip."""
    import sqlite3
    from pathlib import Path

    db_path = Path("data/indexes/chroma.sqlite3")
    if not db_path.exists():
        pytest.skip("No ChromaDB database found")
        return ""  # Never reached, but satisfies type checker

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM collections LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if not row:
            pytest.skip("No collections in ChromaDB")
            return ""  # Never reached, but satisfies type checker

        # Convert collection name back to model name
        # phentrieve_biolord_2023_m -> FremyCompany/BioLORD-2023-M
        collection_name = row[0].replace("phentrieve_", "")
        if collection_name == "biolord_2023_m":
            return "FremyCompany/BioLORD-2023-M"
        elif collection_name == "labse":
            return "sentence-transformers/LaBSE"
        else:
            # Generic fallback
            return collection_name.replace("_", "-")
    except Exception as e:
        pytest.skip(f"Failed to query ChromaDB: {e}")
        return ""  # Never reached, but satisfies type checker


def check_results_or_skip(results):
    """Helper to skip tests if ChromaDB index is not available."""
    if results is None:
        pytest.skip("ChromaDB index not built - run 'phentrieve index build' first")
    return results


# ============================================================================
# Full Benchmark Workflow Tests
# ============================================================================


def test_run_evaluation_includes_new_metrics(
    mock_test_data_tiny, temp_results_dir, available_model
):
    """run_evaluation should calculate all new MTEB metrics."""
    results = run_evaluation(
        model_name=available_model,
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5),
        save_results=True,
        results_dir=temp_results_dir,
    )
    results = check_results_or_skip(results)

    # Check that new metrics are present
    assert "ndcg_dense@1" in results
    assert "ndcg_dense@3" in results
    assert "ndcg_dense@5" in results

    assert "recall_dense@1" in results
    assert "recall_dense@3" in results
    assert "recall_dense@5" in results

    assert "precision_dense@1" in results
    assert "precision_dense@3" in results
    assert "precision_dense@5" in results

    assert "map_dense@1" in results
    assert "map_dense@3" in results
    assert "map_dense@5" in results

    # Check that average metrics are calculated
    assert "avg_ndcg_dense@1" in results
    assert "avg_recall_dense@1" in results
    assert "avg_precision_dense@1" in results
    assert "avg_map_dense@1" in results


def test_run_evaluation_includes_confidence_intervals(
    mock_test_data_tiny, temp_results_dir
):
    """run_evaluation should calculate bootstrap CIs for all metrics."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        save_results=True,
        results_dir=temp_results_dir,
    )
    results = check_results_or_skip(results)

    # Check that confidence intervals are present
    assert "confidence_intervals" in results
    ci = results["confidence_intervals"]

    # Check MRR CI
    assert "mrr_dense" in ci
    assert "point_estimate" in ci["mrr_dense"]
    assert "ci_lower" in ci["mrr_dense"]
    assert "ci_upper" in ci["mrr_dense"]
    assert "ci_level" in ci["mrr_dense"]
    assert ci["mrr_dense"]["ci_level"] == 0.95

    # Check new metric CIs
    assert "ndcg_dense@1" in ci
    assert "recall_dense@1" in ci
    assert "precision_dense@1" in ci
    assert "map_dense@1" in ci


def test_compare_models_includes_new_metrics(mock_test_data_tiny, temp_results_dir):
    """compare_models should include new metrics in comparison table."""
    # Run evaluation for single model
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5),
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Compare with itself (for testing structure)
    comparison_df = compare_models([results, results])

    # Check that new metric columns exist
    expected_columns = [
        "NDCG@1 (Dense)",
        "NDCG@3 (Dense)",
        "NDCG@5 (Dense)",
        "Recall@1 (Dense)",
        "Recall@3 (Dense)",
        "Recall@5 (Dense)",
        "Precision@1 (Dense)",
        "Precision@3 (Dense)",
        "Precision@5 (Dense)",
        "MAP@1 (Dense)",
        "MAP@3 (Dense)",
        "MAP@5 (Dense)",
    ]

    for col in expected_columns:
        assert col in comparison_df.columns


def test_compare_models_with_significance_workflow(
    mock_test_data_tiny, temp_results_dir
):
    """Full workflow: run 2 models, compare with significance tests."""
    # Run evaluation for first model
    results_a = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        save_results=False,
    )
    results_a = check_results_or_skip(results_a)

    # Simulate second model results (slightly different)
    results_b = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        save_results=False,
    )
    results_b = check_results_or_skip(results_b)

    # Compare with significance testing
    comparison = compare_models_with_significance(
        results_a,
        results_b,
        model_a_name="Model A",
        model_b_name="Model B",
        k_values=(1, 3),
        n_bootstrap=100,  # Low for speed
    )

    # Check structure
    assert "model_a" in comparison
    assert "model_b" in comparison
    assert "comparisons" in comparison

    # Check that new metrics are compared
    comparisons = comparison["comparisons"]
    assert "ndcg_dense@1" in comparisons or len(comparisons) > 0
    assert "recall_dense@1" in comparisons or len(comparisons) > 0

    # Each comparison should have diff, p_value, significant
    for metric, result in comparisons.items():
        assert "diff" in result
        assert "p_value" in result
        assert "significant" in result


# ============================================================================
# Metric Consistency Tests
# ============================================================================


def test_metrics_values_are_bounded(mock_test_data_tiny):
    """All metrics should be in valid ranges [0, 1]."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5),
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Check all average metrics are in [0, 1]
    metric_keys = [
        "avg_mrr_dense",
        "avg_ndcg_dense@1",
        "avg_ndcg_dense@3",
        "avg_ndcg_dense@5",
        "avg_recall_dense@1",
        "avg_recall_dense@3",
        "avg_recall_dense@5",
        "avg_precision_dense@1",
        "avg_precision_dense@3",
        "avg_precision_dense@5",
        "avg_map_dense@1",
        "avg_map_dense@3",
        "avg_map_dense@5",
    ]

    for key in metric_keys:
        if key in results:
            value = results[key]
            assert 0.0 <= value <= 1.0, f"{key} out of bounds: {value}"


def test_confidence_intervals_are_valid(mock_test_data_tiny):
    """All confidence intervals should satisfy: lower <= point <= upper."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        save_results=False,
    )
    results = check_results_or_skip(results)

    ci = results["confidence_intervals"]

    for metric_name, ci_data in ci.items():
        point = ci_data["point_estimate"]
        lower = ci_data["ci_lower"]
        upper = ci_data["ci_upper"]

        # CI should contain point estimate
        assert lower <= point <= upper, (
            f"{metric_name}: CI [{lower}, {upper}] does not contain point {point}"
        )

        # All values should be in [0, 1]
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= point <= 1.0
        assert 0.0 <= upper <= 1.0


def test_recall_increases_with_k(mock_test_data_tiny):
    """Recall@K should monotonically increase as K increases."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5, 10),
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Average recall should increase with K
    recall_1 = results["avg_recall_dense@1"]
    recall_3 = results["avg_recall_dense@3"]
    recall_5 = results["avg_recall_dense@5"]
    recall_10 = results["avg_recall_dense@10"]

    # Monotonic increase (or equal)
    assert recall_1 <= recall_3 <= recall_5 <= recall_10


def test_ndcg_bounded_by_one(mock_test_data_tiny):
    """NDCG@K should never exceed 1.0 (perfect ranking)."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5),
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Check per-query NDCG values
    for k in [1, 3, 5]:
        ndcg_values = results[f"ndcg_dense@{k}"]
        for ndcg in ndcg_values:
            assert 0.0 <= ndcg <= 1.0, f"NDCG@{k} out of bounds: {ndcg}"


def test_precision_at_1_binary(mock_test_data_tiny):
    """Precision@1 should be either 0.0 or 1.0 (binary for single result)."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1,),
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Each precision@1 value should be 0 or 1
    precision_values = results["precision_dense@1"]
    for p in precision_values:
        assert p in [0.0, 1.0], f"Precision@1 should be binary, got {p}"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_single_query_benchmark(benchmark_data_dir, temp_results_dir):
    """Benchmark should work with single query (edge case for bootstrap)."""
    # Create single-query test file
    single_query_file = temp_results_dir / "single_query.json"
    with open(single_query_file, "w") as f:
        json.dump(
            [
                {
                    "input_text": "Krampfanfälle",
                    "expected_hpo_ids": ["HP:0001250"],
                    "language": "de",
                }
            ],
            f,
        )

    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(single_query_file),
        k_values=(1,),
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Should complete without errors
    assert "avg_ndcg_dense@1" in results
    assert "confidence_intervals" in results


def test_empty_benchmark_handles_gracefully(temp_results_dir):
    """Benchmark should handle empty test file gracefully."""
    empty_file = temp_results_dir / "empty.json"
    with open(empty_file, "w") as f:
        json.dump([], f)

    # Should return None (indicating failure to load test data)
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(empty_file),
        k_values=(1,),
        save_results=False,
    )

    # Empty test file should result in None return
    assert results is None


def test_llm_benchmark_smoke_writes_result_file(tmp_path, monkeypatch):
    output_path = tmp_path / "llm_benchmark_result.json"

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.run_llm_benchmark",
        lambda **kwargs: {
            "cases": 1,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "output_path": str(output_path),
        },
    )

    result = run_llm_benchmark_cli(
        test_file="tests/data/benchmarks/german/tiny_v1.json",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        output_path=str(output_path),
    )

    assert result["cases"] == 1
    assert Path(result["output_path"]).exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["llm_model"] == (
        "gemini-2.5-flash"
    )


def test_llm_benchmark_cli_rejects_missing_input_file(tmp_path):
    missing_file = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="Benchmark test file not found"):
        run_llm_benchmark_cli(
            test_file=str(missing_file),
            llm_model="gemini-2.5-flash",
        )


def test_llm_benchmark_cli_rejects_invalid_json_input(tmp_path):
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{not json}", encoding="utf-8")

    with pytest.raises(ValueError, match="must be valid JSON"):
        run_llm_benchmark_cli(
            test_file=str(invalid_file),
            llm_model="gemini-2.5-flash",
        )


def test_llm_benchmark_smoke_supports_phenobert_directory(
    tmp_path, monkeypatch, benchmark_data_dir
):
    output_path = tmp_path / "llm_benchmark_gene_reviews.json"

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            assert text
            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                ),
            )

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.get_llm_provider",
        lambda llm_model: object(),
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.TwoPhaseLLMPipeline",
        _FakePipeline,
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark._build_grounded_chunks",
        lambda **kwargs: [{"chunk_id": 1, "text": kwargs["text"]}],
    )

    result = run_llm_benchmark_cli(
        test_file=str(Path("tests/data/en/phenobert")),
        dataset="GeneReviews",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        output_path=str(output_path),
    )

    assert result["cases"] == 10
    assert result["dataset"] == "GeneReviews"
    assert result["test_file"] == str(Path("tests/data/en/phenobert"))
    assert Path(result["output_path"]).exists()


def test_llm_benchmark_smoke_persists_grounded_trace_fields(tmp_path, monkeypatch):
    output_path = tmp_path / "llm_benchmark_grounded_trace.json"

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            assert text
            assert grounded_chunks[0]["chunk_id"] == 1
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
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    trace={
                        "phase1": {
                            "extracted": [
                                {
                                    "phrase": "seizures",
                                    "category": "abnormal",
                                    "chunk_ids": [1],
                                    "evidence_text": "seizures",
                                    "actionable": True,
                                }
                            ]
                        }
                    },
                ),
            )

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.get_llm_provider",
        lambda llm_model: object(),
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.TwoPhaseLLMPipeline",
        _FakePipeline,
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark._build_grounded_chunks",
        lambda **kwargs: [{"chunk_id": 1, "text": "Patient has seizures."}],
    )

    result = run_llm_benchmark_cli(
        test_file=str(Path("tests/data/en/phenobert")),
        dataset="GeneReviews",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        output_path=str(output_path),
    )

    observability = result["prediction_records"][0]["metadata"]["observability"]
    assert observability["phase1_completed_groups"] == 0
    assert observability["phase1_failed_groups"] == 0
    assert observability["phase1_partial_failures"] == 0
    assert result["prediction_records"][0]["trace"]["phase1"]["extracted"][0][
        "chunk_ids"
    ] == [1]


def test_benchmark_trace_persists_group_failures(tmp_path, monkeypatch):
    output_path = tmp_path / "llm_benchmark_group_failures.json"
    captured_calls: list[dict[str, object]] = []

    class _FakeProvider:
        def count_tokens(self, *, system_prompt, user_prompt):
            return {"prompt_tokens": 30001, "total_tokens": 30001}

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, extraction_groups, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            captured_calls.append(
                {
                    "text": text,
                    "grounded_chunks": grounded_chunks,
                    "extraction_groups": extraction_groups,
                }
            )
            assert text
            assert grounded_chunks[0]["chunk_id"] == 1
            assert extraction_groups == [
                {
                    "group_id": 1,
                    "chunk_ids": [1],
                    "text": "chunk_id=1: Patient has seizures.",
                    "estimated_prompt_tokens": 12,
                },
                {
                    "group_id": 2,
                    "chunk_ids": [2],
                    "text": "chunk_id=2: Additional chunk.",
                    "estimated_prompt_tokens": 11,
                },
            ]
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
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    phase_counts={
                        "extracted_phrases": 1,
                        "actionable_phrases": 1,
                        "phase1_completed_groups": 1,
                        "phase1_failed_groups": 1,
                        "phase1_partial_failures": 1,
                    },
                    trace={
                        "phase1": {
                            "extracted": [
                                {
                                    "phrase": "seizures",
                                    "category": "abnormal",
                                    "chunk_ids": [1],
                                    "evidence_text": "seizures",
                                    "actionable": True,
                                }
                            ],
                            "groups": [
                                {
                                    "group_id": 1,
                                    "chunk_ids": [1],
                                    "status": "completed",
                                    "extracted_count": 1,
                                    "extracted": [
                                        {
                                            "phrase": "seizures",
                                            "category": "abnormal",
                                            "chunk_ids": [1],
                                            "evidence_text": "seizures",
                                            "actionable": True,
                                        }
                                    ],
                                },
                                {
                                    "group_id": 2,
                                    "chunk_ids": [2],
                                    "status": "failed",
                                    "error": "Structured extraction failed",
                                    "error_type": "LLMPipelinePhaseError",
                                    "extracted_count": 0,
                                    "extracted": [],
                                },
                            ],
                        }
                    },
                ),
            )

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.get_llm_provider",
        lambda llm_model: _FakeProvider(),
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.TwoPhaseLLMPipeline",
        _FakePipeline,
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark._build_grounded_chunks",
        lambda **kwargs: [
            {"chunk_id": 1, "text": "Patient has seizures."},
            {"chunk_id": 2, "text": "Additional chunk."},
        ],
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.build_extraction_groups",
        lambda **kwargs: [
            ExtractionGroup(
                group_id=1,
                chunk_ids=[1],
                text="chunk_id=1: Patient has seizures.",
                estimated_prompt_tokens=12,
            ),
            ExtractionGroup(
                group_id=2,
                chunk_ids=[2],
                text="chunk_id=2: Additional chunk.",
                estimated_prompt_tokens=11,
            ),
        ],
    )

    result = run_llm_benchmark_cli(
        test_file=str(Path("tests/data/en/phenobert")),
        dataset="GeneReviews",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        output_path=str(output_path),
    )

    assert captured_calls
    assert captured_calls[0]["grounded_chunks"] == [
        {"chunk_id": 1, "text": "Patient has seizures."},
        {"chunk_id": 2, "text": "Additional chunk."},
    ]
    prediction_record = result["prediction_records"][0]
    observability = prediction_record["metadata"]["observability"]
    phase1_trace = prediction_record["trace"]["phase1"]
    assert observability["request_count"] == 0
    assert observability["extracted_phrases"] == 1
    assert observability["actionable_phrases"] == 1
    assert observability["phase1_completed_groups"] == 1
    assert observability["phase1_failed_groups"] == 1
    assert observability["phase1_partial_failures"] == 1
    assert observability["grounded_chunks"] == 2
    assert observability["extraction_groups"] == 2
    assert observability["failed_groups"] == 1
    assert observability["deduplicated_phase1_mentions"] == 0
    assert observability["deduplicated_unresolved_mappings"] == 0
    assert observability["phase1_requests"] == 0
    assert observability["phase2b_llm_requests"] == 0
    assert phase1_trace["groups"] == [
        {
            "group_id": 1,
            "chunk_ids": [1],
            "status": "completed",
            "extracted_count": 1,
            "extracted": [
                {
                    "phrase": "seizures",
                    "category": "abnormal",
                    "chunk_ids": [1],
                    "evidence_text": "seizures",
                    "actionable": True,
                }
            ],
        },
        {
            "group_id": 2,
            "chunk_ids": [2],
            "status": "failed",
            "error": "Structured extraction failed",
            "error_type": "LLMPipelinePhaseError",
            "extracted_count": 0,
            "extracted": [],
        },
    ]
    assert phase1_trace["extracted"] == [
        {
            "phrase": "seizures",
            "category": "abnormal",
            "chunk_ids": [1],
            "evidence_text": "seizures",
            "actionable": True,
        }
    ]
    assert result["results"][0]["partial_failure_counts"] == {
        "phase1_completed_groups": 1,
        "phase1_failed_groups": 1,
        "phase1_partial_failures": 1,
    }


def test_benchmark_artifact_persists_group_counts_and_phase1_group_timings(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "llm_benchmark_group_observability.json"

    class _FakeProvider:
        def count_tokens(self, *, system_prompt, user_prompt):
            return {"prompt_tokens": 30001, "total_tokens": 30001}

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, extraction_groups, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            assert text
            assert len(grounded_chunks) == 2
            assert len(extraction_groups) == 2
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
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    phase_counts={
                        "extracted_phrases": 1,
                        "actionable_phrases": 2,
                        "candidate_sets": 2,
                        "unresolved_phrases": 2,
                        "local_matches": 0,
                        "llm_mapped_phrases": 2,
                        "local_fallbacks": 0,
                        "phase1_completed_groups": 2,
                        "phase1_failed_groups": 0,
                        "phase1_partial_failures": 0,
                    },
                    trace={
                        "phase1": {
                            "extracted": [
                                {
                                    "phrase": "seizures",
                                    "category": "abnormal",
                                    "chunk_ids": [1, 2],
                                    "evidence_text": "seizures",
                                    "actionable": True,
                                }
                            ],
                            "groups": [
                                {
                                    "group_id": 1,
                                    "chunk_ids": [1],
                                    "status": "completed",
                                    "extracted_count": 1,
                                    "prompt_tokens": 10,
                                    "completion_tokens": 3,
                                    "request_count": 1,
                                    "elapsed_seconds": 0.41,
                                    "extracted": [
                                        {
                                            "phrase": "seizures",
                                            "category": "abnormal",
                                            "chunk_ids": [1],
                                            "evidence_text": "seizures",
                                            "actionable": True,
                                        }
                                    ],
                                },
                                {
                                    "group_id": 2,
                                    "chunk_ids": [2],
                                    "status": "completed",
                                    "extracted_count": 1,
                                    "prompt_tokens": 9,
                                    "completion_tokens": 4,
                                    "request_count": 1,
                                    "elapsed_seconds": 0.63,
                                    "extracted": [
                                        {
                                            "phrase": "seizures",
                                            "category": "abnormal",
                                            "chunk_ids": [2],
                                            "evidence_text": "seizures",
                                            "actionable": True,
                                        }
                                    ],
                                },
                            ],
                        },
                        "phase2b_llm": {
                            "resolved": [
                                {
                                    "phrase": "seizures",
                                    "selected_id": "HP:0001250",
                                    "term_id": "HP:0001250",
                                    "label": "Seizure",
                                    "assertion": "present",
                                    "category": "abnormal",
                                    "match_method": "llm",
                                    "local_fallback": False,
                                    "chunk_ids": [1],
                                    "evidence_text": "seizures",
                                },
                                {
                                    "phrase": "seizures",
                                    "selected_id": "HP:0001250",
                                    "term_id": "HP:0001250",
                                    "label": "Seizure",
                                    "assertion": "present",
                                    "category": "abnormal",
                                    "match_method": "llm",
                                    "local_fallback": False,
                                    "chunk_ids": [2],
                                    "evidence_text": "seizures",
                                },
                            ]
                        },
                    },
                ),
            )

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.get_llm_provider",
        lambda llm_model: _FakeProvider(),
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.TwoPhaseLLMPipeline",
        _FakePipeline,
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark._build_grounded_chunks",
        lambda **kwargs: [
            {"chunk_id": 1, "text": "Patient has seizures."},
            {"chunk_id": 2, "text": "Repeated mention of seizures."},
        ],
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.build_extraction_groups",
        lambda **kwargs: [
            ExtractionGroup(
                group_id=1,
                chunk_ids=[1],
                text="chunk_id=1: Patient has seizures.",
                estimated_prompt_tokens=10,
            ),
            ExtractionGroup(
                group_id=2,
                chunk_ids=[2],
                text="chunk_id=2: Repeated mention of seizures.",
                estimated_prompt_tokens=9,
            ),
        ],
    )

    result = run_llm_benchmark_cli(
        test_file=str(Path("tests/data/en/phenobert")),
        dataset="GeneReviews",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        output_path=str(output_path),
    )

    prediction_record = result["prediction_records"][0]
    observability = prediction_record["metadata"]["observability"]
    phase1_groups = prediction_record["trace"]["phase1"]["groups"]
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    persisted_prediction = persisted["prediction_records"][0]
    persisted_observability = persisted_prediction["metadata"]["observability"]
    persisted_phase1_groups = persisted_prediction["trace"]["phase1"]["groups"]
    assert observability["grounded_chunks"] == 2
    assert observability["extraction_groups"] == 2
    assert observability["failed_groups"] == 0
    assert observability["deduplicated_phase1_mentions"] == 1
    assert observability["deduplicated_unresolved_mappings"] == 1
    assert phase1_groups[0]["elapsed_seconds"] == 0.41
    assert phase1_groups[1]["elapsed_seconds"] == 0.63
    assert phase1_groups[0]["prompt_tokens"] == 10
    assert phase1_groups[1]["completion_tokens"] == 4
    assert persisted_observability["grounded_chunks"] == 2
    assert persisted_observability["extraction_groups"] == 2
    assert persisted_observability["deduplicated_phase1_mentions"] == 1
    assert persisted_observability["deduplicated_unresolved_mappings"] == 1
    assert persisted_phase1_groups[0]["elapsed_seconds"] == 0.41
    assert persisted_phase1_groups[1]["elapsed_seconds"] == 0.63


def test_benchmark_record_includes_partial_failure_counts(tmp_path, monkeypatch):
    output_path = tmp_path / "llm_benchmark_partial_failure_counts.json"

    class _FakeProvider:
        def count_tokens(self, *, system_prompt, user_prompt):
            return {"prompt_tokens": 30001, "total_tokens": 30001}

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, extraction_groups, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            assert text
            assert grounded_chunks
            assert extraction_groups
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
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    phase_counts={
                        "extracted_phrases": 1,
                        "actionable_phrases": 1,
                        "candidate_sets": 1,
                        "unresolved_phrases": 0,
                        "local_matches": 1,
                        "llm_mapped_phrases": 0,
                        "local_fallbacks": 0,
                        "phase1_completed_groups": 1,
                        "phase1_failed_groups": 1,
                        "phase1_partial_failures": 1,
                    },
                    trace={
                        "phase1": {
                            "extracted": [
                                {
                                    "phrase": "seizures",
                                    "category": "abnormal",
                                    "chunk_ids": [1],
                                    "evidence_text": "seizures",
                                    "actionable": True,
                                }
                            ],
                            "groups": [
                                {
                                    "group_id": 1,
                                    "chunk_ids": [1],
                                    "status": "completed",
                                    "extracted_count": 1,
                                    "elapsed_seconds": 0.4,
                                    "extracted": [
                                        {
                                            "phrase": "seizures",
                                            "category": "abnormal",
                                            "chunk_ids": [1],
                                            "evidence_text": "seizures",
                                            "actionable": True,
                                        }
                                    ],
                                },
                                {
                                    "group_id": 2,
                                    "chunk_ids": [2],
                                    "status": "failed",
                                    "error": "Structured extraction failed",
                                    "error_type": "LLMPipelinePhaseError",
                                    "extracted_count": 0,
                                    "elapsed_seconds": 0.2,
                                    "extracted": [],
                                },
                            ],
                        }
                    },
                ),
            )

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.get_llm_provider",
        lambda llm_model: _FakeProvider(),
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.TwoPhaseLLMPipeline",
        _FakePipeline,
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark._build_grounded_chunks",
        lambda **kwargs: [
            {"chunk_id": 1, "text": "Patient has seizures."},
            {"chunk_id": 2, "text": "Additional chunk."},
        ],
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.build_extraction_groups",
        lambda **kwargs: [
            ExtractionGroup(
                group_id=1,
                chunk_ids=[1],
                text="chunk_id=1: Patient has seizures.",
                estimated_prompt_tokens=12,
            ),
            ExtractionGroup(
                group_id=2,
                chunk_ids=[2],
                text="chunk_id=2: Additional chunk.",
                estimated_prompt_tokens=11,
            ),
        ],
    )

    result = run_llm_benchmark_cli(
        test_file=str(Path("tests/data/en/phenobert")),
        dataset="GeneReviews",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        output_path=str(output_path),
    )

    prediction_record = result["prediction_records"][0]
    observability = prediction_record["metadata"]["observability"]
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    persisted_record = persisted["results"][0]
    persisted_observability = persisted["prediction_records"][0]["metadata"][
        "observability"
    ]
    assert observability["failed_groups"] == 1
    assert result["results"][0]["partial_failure_counts"] == {
        "phase1_completed_groups": 1,
        "phase1_failed_groups": 1,
        "phase1_partial_failures": 1,
    }
    assert persisted_observability["failed_groups"] == 1
    assert persisted_record["partial_failure_counts"] == {
        "phase1_completed_groups": 1,
        "phase1_failed_groups": 1,
        "phase1_partial_failures": 1,
    }


def test_benchmark_grouped_execution_keeps_legacy_pipeline_call_surface(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "llm_benchmark_legacy_pipeline.json"
    build_extraction_groups_calls: list[dict[str, object]] = []
    captured_calls: list[dict[str, object]] = []

    class _FakeProvider:
        def count_tokens(self, *, system_prompt, user_prompt):
            return {"prompt_tokens": 30001, "total_tokens": 30001}

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype

            captured_calls.append(
                {
                    "text": text,
                    "grounded_chunks": grounded_chunks,
                    "config": config,
                }
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

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.get_llm_provider",
        lambda llm_model: _FakeProvider(),
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.TwoPhaseLLMPipeline",
        _FakePipeline,
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark._build_grounded_chunks",
        lambda **kwargs: [
            {"chunk_id": 1, "text": "Patient has seizures."},
            {"chunk_id": 2, "text": "Additional chunk."},
        ],
    )

    def _fake_build_extraction_groups(**kwargs):
        build_extraction_groups_calls.append(kwargs)
        return [
            ExtractionGroup(
                group_id=1,
                chunk_ids=[1],
                text="chunk_id=1: Patient has seizures.",
                estimated_prompt_tokens=12,
            ),
            ExtractionGroup(
                group_id=2,
                chunk_ids=[2],
                text="chunk_id=2: Additional chunk.",
                estimated_prompt_tokens=11,
            ),
        ]

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.build_extraction_groups",
        _fake_build_extraction_groups,
    )

    result = run_llm_benchmark_cli(
        test_file=str(Path("tests/data/en/phenobert")),
        dataset="GeneReviews",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        output_path=str(output_path),
    )

    assert build_extraction_groups_calls
    assert captured_calls
    assert captured_calls[0]["grounded_chunks"] == [
        {"chunk_id": 1, "text": "Patient has seizures."},
        {"chunk_id": 2, "text": "Additional chunk."},
    ]
    assert result["results"][0]["predicted_hpo_ids"] == ["HP:0001250"]


def test_llm_benchmark_rejects_unknown_phenobert_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        run_llm_benchmark_cli(
            test_file=str(Path("tests/data/en/phenobert")),
            dataset="not_a_dataset",
            llm_model="gemini-2.5-flash",
            output_path=str(tmp_path / "result.json"),
        )


def test_llm_benchmark_rejects_invalid_phenobert_root(tmp_path):
    invalid_root = tmp_path / "GeneReviews"
    invalid_root.mkdir()

    with pytest.raises(ValueError, match="No benchmark documents found"):
        run_llm_benchmark_cli(
            test_file=str(invalid_root),
            dataset="GeneReviews",
            llm_model="gemini-2.5-flash",
            output_path=str(tmp_path / "result.json"),
        )


def test_extraction_benchmark_uses_effective_config_overrides(tmp_path, monkeypatch):
    benchmark = ExtractionBenchmark(
        model_name="sentence-transformers/LaBSE",
        config=ExtractionConfig(
            model_name="sentence-transformers/LaBSE",
            dataset="all",
            averaging="micro",
            bootstrap_ci=False,
        ),
    )

    captured: dict[str, object] = {}

    def _fake_load_benchmark_data(test_path, dataset):
        captured["dataset"] = dataset
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
                    "source_dataset": dataset,
                }
            ],
        }

    monkeypatch.setattr(
        "phentrieve.benchmark.extraction_benchmark.load_benchmark_data",
        _fake_load_benchmark_data,
    )
    monkeypatch.setattr(benchmark.extractor, "extract", lambda text: [])

    output_dir = tmp_path / "results"
    benchmark.run_benchmark(
        test_file=tmp_path,
        output_dir=output_dir,
        config_overrides={
            "dataset": "GeneReviews",
            "averaging": "macro",
        },
    )

    saved_payload = json.loads(
        (output_dir / "extraction_results.json").read_text(encoding="utf-8")
    )

    assert captured["dataset"] == "GeneReviews"
    assert saved_payload["metadata"]["config"]["averaging"] == "macro"
    assert saved_payload["metadata"]["dataset"]["dataset_name"] == (
        "phenobert_GeneReviews"
    )
