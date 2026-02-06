"""Tests for LLM benchmark data loader and metrics calculation."""

from __future__ import annotations

from pathlib import Path

import pytest

from phentrieve.benchmark.data_loader import (
    ASSERTION_STATUS_MAP,
    LLM_ASSERTION_TO_BENCHMARK,
    load_phenobert_data,
    parse_gold_terms,
)
from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    ExtractionResult,
)

PHENOBERT_DIR = Path(__file__).resolve().parents[1] / "data" / "en" / "phenobert"


class TestAssertionStatusMapping:
    """Test assertion status mapping constants."""

    def test_assertion_status_map_values(self):
        assert ASSERTION_STATUS_MAP["affirmed"] == "PRESENT"
        assert ASSERTION_STATUS_MAP["negated"] == "ABSENT"
        assert ASSERTION_STATUS_MAP["uncertain"] == "UNCERTAIN"
        assert ASSERTION_STATUS_MAP["normal"] == "PRESENT"

    def test_llm_assertion_to_benchmark_values(self):
        assert LLM_ASSERTION_TO_BENCHMARK["affirmed"] == "PRESENT"
        assert LLM_ASSERTION_TO_BENCHMARK["negated"] == "ABSENT"
        assert LLM_ASSERTION_TO_BENCHMARK["uncertain"] == "UNCERTAIN"

    def test_llm_mapping_covers_all_assertion_statuses(self):
        """LLM AssertionStatus enum values should all be mapped."""
        from phentrieve.llm.types import AssertionStatus

        for status in AssertionStatus:
            assert status.value in LLM_ASSERTION_TO_BENCHMARK


class TestParseGoldTerms:
    """Test parse_gold_terms with various input formats."""

    def test_dict_format_with_id_and_assertion(self):
        terms = [{"id": "HP:0001250", "assertion": "PRESENT"}]
        result = parse_gold_terms(terms)
        assert result == [("HP:0001250", "PRESENT")]

    def test_dict_format_with_hpo_id_key(self):
        terms = [{"hpo_id": "HP:0001250", "assertion": "ABSENT"}]
        result = parse_gold_terms(terms)
        assert result == [("HP:0001250", "ABSENT")]

    def test_dict_format_defaults_to_present(self):
        terms = [{"id": "HP:0001250"}]
        result = parse_gold_terms(terms)
        assert result == [("HP:0001250", "PRESENT")]

    def test_tuple_format(self):
        terms = [("HP:0001250", "ABSENT")]
        result = parse_gold_terms(terms)
        assert result == [("HP:0001250", "ABSENT")]

    def test_list_format(self):
        terms = [["HP:0001250", "UNCERTAIN"]]
        result = parse_gold_terms(terms)
        assert result == [("HP:0001250", "UNCERTAIN")]

    def test_string_format_defaults_to_present(self):
        terms = ["HP:0001250"]
        result = parse_gold_terms(terms)
        assert result == [("HP:0001250", "PRESENT")]

    def test_empty_list(self):
        assert parse_gold_terms([]) == []

    def test_mixed_formats(self):
        terms = [
            {"id": "HP:0001250", "assertion": "PRESENT"},
            ("HP:0002865", "ABSENT"),
            "HP:0000001",
        ]
        result = parse_gold_terms(terms)
        assert len(result) == 3
        assert result[0] == ("HP:0001250", "PRESENT")
        assert result[1] == ("HP:0002865", "ABSENT")
        assert result[2] == ("HP:0000001", "PRESENT")


class TestLoadPhenobertData:
    """Test PhenoBERT data loading from directory structure."""

    @pytest.mark.skipif(
        not PHENOBERT_DIR.exists(),
        reason="PhenoBERT test data not available",
    )
    def test_load_all_datasets(self):
        data = load_phenobert_data(PHENOBERT_DIR, dataset="all")
        assert "metadata" in data
        assert "documents" in data
        assert data["metadata"]["source"] == "phenobert"
        assert data["metadata"]["total_documents"] > 0
        assert len(data["documents"]) == data["metadata"]["total_documents"]

    @pytest.mark.skipif(
        not PHENOBERT_DIR.exists(),
        reason="PhenoBERT test data not available",
    )
    def test_load_single_dataset(self):
        data = load_phenobert_data(PHENOBERT_DIR, dataset="GeneReviews")
        assert data["metadata"]["dataset_name"] == "phenobert_GeneReviews"
        for doc in data["documents"]:
            assert doc["source_dataset"] == "GeneReviews"

    @pytest.mark.skipif(
        not PHENOBERT_DIR.exists(),
        reason="PhenoBERT test data not available",
    )
    def test_document_structure(self):
        data = load_phenobert_data(PHENOBERT_DIR, dataset="GeneReviews")
        assert len(data["documents"]) > 0
        doc = data["documents"][0]
        assert "id" in doc
        assert "text" in doc
        assert "gold_hpo_terms" in doc
        assert "source_dataset" in doc
        assert len(doc["text"]) > 0
        assert len(doc["gold_hpo_terms"]) > 0

    @pytest.mark.skipif(
        not PHENOBERT_DIR.exists(),
        reason="PhenoBERT test data not available",
    )
    def test_gold_terms_have_benchmark_format(self):
        """Gold terms should have assertion mapped to PRESENT/ABSENT/UNCERTAIN."""
        data = load_phenobert_data(PHENOBERT_DIR, dataset="GeneReviews")
        doc = data["documents"][0]
        for term in doc["gold_hpo_terms"]:
            assert "id" in term
            assert term["assertion"] in ("PRESENT", "ABSENT", "UNCERTAIN")

    def test_invalid_dataset_raises_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_phenobert_data(tmp_path, dataset="nonexistent")

    def test_missing_directory_logs_warning(self, tmp_path: Path, caplog):
        """Loading from empty directory should return 0 documents."""
        data = load_phenobert_data(tmp_path, dataset="all")
        assert len(data["documents"]) == 0


class TestMetricsCalculation:
    """Test metrics calculation with mock extraction results."""

    def test_perfect_predictions(self):
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001250", "PRESENT"), ("HP:0002865", "ABSENT")],
                gold=[("HP:0001250", "PRESENT"), ("HP:0002865", "ABSENT")],
            ),
        ]
        evaluator = CorpusExtractionMetrics(averaging="micro")
        metrics = evaluator.calculate_all_metrics(results)
        assert metrics.micro["precision"] == 1.0
        assert metrics.micro["recall"] == 1.0
        assert metrics.micro["f1"] == 1.0

    def test_no_overlap(self):
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001250", "PRESENT")],
                gold=[("HP:0002865", "PRESENT")],
            ),
        ]
        evaluator = CorpusExtractionMetrics(averaging="micro")
        metrics = evaluator.calculate_all_metrics(results)
        assert metrics.micro["precision"] == 0.0
        assert metrics.micro["recall"] == 0.0

    def test_assertion_mismatch_counts_as_wrong(self):
        """Same HPO ID but different assertion should not match."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001250", "PRESENT")],
                gold=[("HP:0001250", "ABSENT")],
            ),
        ]
        evaluator = CorpusExtractionMetrics(averaging="micro")
        metrics = evaluator.calculate_all_metrics(results)
        assert metrics.micro["f1"] == 0.0

    def test_id_only_ignores_assertion(self):
        """When both use PRESENT, assertion mismatch is eliminated."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001250", "PRESENT")],
                gold=[("HP:0001250", "PRESENT")],
            ),
        ]
        evaluator = CorpusExtractionMetrics(averaging="micro")
        metrics = evaluator.calculate_all_metrics(results)
        assert metrics.micro["f1"] == 1.0

    def test_multi_document_micro_averaging(self):
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001250", "PRESENT")],
                gold=[("HP:0001250", "PRESENT"), ("HP:0002865", "PRESENT")],
            ),
            ExtractionResult(
                doc_id="doc2",
                predicted=[("HP:0003000", "PRESENT")],
                gold=[("HP:0003000", "PRESENT")],
            ),
        ]
        evaluator = CorpusExtractionMetrics(averaging="micro")
        metrics = evaluator.calculate_all_metrics(results)
        # TP=2, FP=0, FN=1 -> P=1.0, R=2/3, F1=0.8
        assert metrics.micro["precision"] == 1.0
        assert abs(metrics.micro["recall"] - 2 / 3) < 1e-6


class TestLLMBenchmarkConfig:
    """Test LLMBenchmarkConfig defaults and construction."""

    def test_defaults(self):
        from phentrieve.benchmark.llm_benchmark import LLMBenchmarkConfig

        config = LLMBenchmarkConfig()
        assert config.model == "github/gpt-4o"
        assert config.modes == ["tool_text"]
        assert config.dataset == "all"
        assert config.include_assertions is True
        assert config.temperature == 0.0
        assert config.limit is None
        assert config.bootstrap_ci is False

    def test_custom_config(self):
        from phentrieve.benchmark.llm_benchmark import LLMBenchmarkConfig

        config = LLMBenchmarkConfig(
            model="gemini/gemini-3-flash-preview",
            modes=["direct", "tool_text"],
            dataset="GeneReviews",
            limit=10,
        )
        assert config.model == "gemini/gemini-3-flash-preview"
        assert len(config.modes) == 2
        assert config.dataset == "GeneReviews"
        assert config.limit == 10


class TestLLMBenchmarkLoadTestData:
    """Test LLMBenchmark._load_test_data for different formats."""

    def test_simple_json_format(self, tmp_path: Path):
        """Simple [{text, expected_hpo_ids}] format should be normalized."""
        import json

        from phentrieve.benchmark.llm_benchmark import (
            LLMBenchmark,
            LLMBenchmarkConfig,
        )

        test_data = [
            {
                "id": "case1",
                "text": "Patient has seizures.",
                "expected_hpo_ids": ["HP:0001250"],
            },
            {
                "id": "case2",
                "input_text": "No cardiac issues.",
                "expected_hpo_ids": ["HP:0001627"],
            },
        ]
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(test_data))

        benchmark = LLMBenchmark(LLMBenchmarkConfig())
        data = benchmark._load_test_data(test_file)

        assert len(data["documents"]) == 2
        assert data["documents"][0]["id"] == "case1"
        assert data["documents"][0]["text"] == "Patient has seizures."
        assert data["documents"][1]["text"] == "No cardiac issues."

        # Gold terms should be converted to dict format
        gold = data["documents"][0]["gold_hpo_terms"]
        assert len(gold) == 1
        assert gold[0]["id"] == "HP:0001250"
        assert gold[0]["assertion"] == "PRESENT"

    @pytest.mark.skipif(
        not PHENOBERT_DIR.exists(),
        reason="PhenoBERT test data not available",
    )
    def test_phenobert_directory_format(self):
        from phentrieve.benchmark.llm_benchmark import (
            LLMBenchmark,
            LLMBenchmarkConfig,
        )

        config = LLMBenchmarkConfig(dataset="GeneReviews")
        benchmark = LLMBenchmark(config)
        data = benchmark._load_test_data(PHENOBERT_DIR)

        assert len(data["documents"]) > 0
        assert data["metadata"]["source"] == "phenobert"


class TestLLMBenchmarkConvertToPhenobertFormat:
    """Test prediction output format conversion."""

    def test_convert_annotations_to_phenobert(self):
        from phentrieve.benchmark.llm_benchmark import (
            LLMBenchmark,
            LLMBenchmarkConfig,
        )
        from phentrieve.llm.types import (
            AnnotationMode,
            AnnotationResult,
            AssertionStatus,
            HPOAnnotation,
            TokenUsage,
        )

        benchmark = LLMBenchmark(LLMBenchmarkConfig())

        mock_result = AnnotationResult(
            annotations=[
                HPOAnnotation(
                    hpo_id="HP:0001250",
                    term_name="Seizure",
                    assertion=AssertionStatus.AFFIRMED,
                    confidence=0.95,
                    evidence_text="patient has seizures",
                    evidence_start=12,
                    evidence_end=32,
                ),
                HPOAnnotation(
                    hpo_id="HP:0001627",
                    term_name="Abnormal heart morphology",
                    assertion=AssertionStatus.NEGATED,
                    confidence=0.8,
                    evidence_text="no cardiac issues",
                ),
            ],
            input_text="The patient has seizures but no cardiac issues.",
            language="en",
            mode=AnnotationMode.TOOL_TEXT,
            model="test-model",
            token_usage=TokenUsage(
                prompt_tokens=100, completion_tokens=50, total_tokens=150, api_calls=1
            ),
        )

        result = benchmark._convert_to_phenobert_format(
            doc_id="test_doc",
            text="The patient has seizures but no cardiac issues.",
            result=mock_result,
            processing_time=1.5,
        )

        assert result["doc_id"] == "test_doc"
        assert result["source"] == "llm_annotation"
        assert result["metadata"]["model"] == "test-model"
        assert result["metadata"]["mode"] == "tool_text"
        assert len(result["annotations"]) == 2

        ann1 = result["annotations"][0]
        assert ann1["hpo_id"] == "HP:0001250"
        assert ann1["assertion_status"] == "affirmed"
        assert ann1["evidence_spans"][0]["text_snippet"] == "patient has seizures"
        assert ann1["evidence_spans"][0]["start_char"] == 12
        assert ann1["evidence_spans"][0]["end_char"] == 32

        ann2 = result["annotations"][1]
        assert ann2["assertion_status"] == "negated"
        # evidence_start/end not set -> no start_char/end_char in span
        assert "start_char" not in ann2["evidence_spans"][0]
