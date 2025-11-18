"""Integration tests for benchmark data loading and commands.

These tests verify benchmark datasets can be loaded and have valid structure.
Uses actual files from tests/data/benchmarks/ (not mocked).
"""

import re
from pathlib import Path

import pytest

from phentrieve.config import BENCHMARK_DATA_DIR, DEFAULT_BENCHMARK_FILE
from phentrieve.data_processing.test_data_loader import load_test_data

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]


class TestBenchmarkDataLoading:
    """Integration tests for benchmark data loading."""

    def test_all_datasets_loadable(self, benchmark_data_dir):
        """Verify all benchmark datasets can be loaded."""
        benchmark_files = list(benchmark_data_dir.rglob("*.json"))
        # Exclude non-dataset files
        benchmark_files = [
            f for f in benchmark_files if f.name not in ["datasets.json", "README.md"]
        ]

        assert len(benchmark_files) == 6, (
            f"Expected 6 datasets, found {len(benchmark_files)}"
        )

        for dataset_file in benchmark_files:
            dataset = load_test_data(str(dataset_file))

            # Basic validation
            assert dataset is not None, f"Failed to load {dataset_file.name}"
            assert len(dataset) > 0, f"Empty dataset: {dataset_file.name}"
            assert isinstance(dataset, list), f"Dataset not a list: {dataset_file.name}"

    def test_dataset_structure_valid(self, benchmark_data_dir):
        """Verify all datasets have required fields and correct structure."""
        hpo_pattern = re.compile(r"^HP:\d{7}$")

        for dataset_file in benchmark_data_dir.rglob("*.json"):
            if dataset_file.name in ["datasets.json", "README.md"]:
                continue

            dataset = load_test_data(str(dataset_file))
            assert dataset is not None, f"Failed to load {dataset_file.name}"

            for i, case in enumerate(dataset):
                # Required fields
                assert "text" in case, f"{dataset_file.name} case {i}: missing 'text'"
                assert "expected_hpo_ids" in case, (
                    f"{dataset_file.name} case {i}: missing 'expected_hpo_ids'"
                )

                # Field types
                assert isinstance(case["text"], str), (
                    f"{dataset_file.name} case {i}: text not string"
                )
                assert isinstance(case["expected_hpo_ids"], list), (
                    f"{dataset_file.name} case {i}: expected_hpo_ids not list"
                )

                # Field content
                assert len(case["text"]) > 0, (
                    f"{dataset_file.name} case {i}: empty text"
                )
                assert len(case["expected_hpo_ids"]) > 0, (
                    f"{dataset_file.name} case {i}: no expected HPO IDs"
                )

                # HPO ID format
                for hpo_id in case["expected_hpo_ids"]:
                    assert hpo_pattern.match(hpo_id), (
                        f"{dataset_file.name} case {i}: invalid HPO ID format: {hpo_id}"
                    )

    def test_expected_case_counts(self, benchmark_data_dir):
        """Verify datasets have expected number of cases."""
        expected_counts = {
            "tiny_v1.json": 9,
            "small_v1.json": 9,
            "70cases_gemini_v1.json": 70,
            "70cases_o3_v1.json": 70,
            "200cases_gemini_v1.json": 200,
            "200cases_o3_v1.json": 200,
        }

        for filename, expected_count in expected_counts.items():
            dataset_path = benchmark_data_dir / "german" / filename
            dataset = load_test_data(str(dataset_path))

            actual_count = len(dataset) if dataset else 0
            assert actual_count == expected_count, (
                f"{filename}: expected {expected_count} cases, got {actual_count}"
            )

    def test_default_dataset_loads(self):
        """Verify default dataset path works."""
        # Construct path like the orchestrator does
        project_root = Path(__file__).parent.parent.parent.parent
        test_file = str(project_root / BENCHMARK_DATA_DIR / DEFAULT_BENCHMARK_FILE)

        dataset = load_test_data(test_file)
        assert dataset is not None
        assert len(dataset) == 9  # tiny_v1.json has 9 cases

    def test_german_datasets_have_non_ascii(self, benchmark_data_dir):
        """Verify German datasets contain non-ASCII characters."""
        # Load a German dataset
        dataset_path = benchmark_data_dir / "german" / "tiny_v1.json"
        dataset = load_test_data(str(dataset_path))

        assert dataset is not None
        texts = [case["text"] for case in dataset]

        # Check for non-ASCII characters (German umlauts, etc.)
        has_non_ascii = any(any(ord(char) > 127 for char in text) for text in texts)
        assert has_non_ascii, "German dataset should contain non-ASCII characters"


class TestBenchmarkCommandsIntegration:
    """Integration tests for benchmark CLI commands with real data.

    These tests verify the benchmark pipeline works end-to-end.
    Marked as slow - skip in fast test runs with: pytest -m "not slow"
    """

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires HPO data prepared locally")
    def test_benchmark_run_tiny_dataset(self, benchmark_data_dir, tmp_path):
        """Test running benchmark with tiny dataset (end-to-end).

        NOTE: This test requires:
        - HPO data prepared: phentrieve data prepare
        - Vector index built: phentrieve index build

        Unskip and run manually when HPO data available.
        """
        from phentrieve.evaluation.benchmark_orchestrator import orchestrate_benchmark

        test_file = str(benchmark_data_dir / "german" / "tiny_v1.json")

        results = orchestrate_benchmark(
            test_file=test_file,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.1,
            cpu=True,
            debug=True,
            results_dir_override=str(tmp_path),
        )

        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)

        # Check expected metrics exist
        expected_metrics = [
            "mrr",
            "hit_rate_at_1",
            "hit_rate_at_5",
            "avg_semantic_similarity",
        ]
        for metric in expected_metrics:
            assert metric in results, f"Missing metric: {metric}"

        # Sanity check metric ranges
        assert 0 <= results["mrr"] <= 1, "MRR out of range"
        assert 0 <= results["hit_rate_at_1"] <= 1, "Hit rate out of range"

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires HPO data - run manually")
    def test_benchmark_comparison_pipeline(self, benchmark_data_dir, tmp_path):
        """Test benchmark comparison with multiple results (end-to-end).

        NOTE: Requires HPO data. Unskip for manual testing.
        """
        # This would test the full comparison pipeline
        # Implement when needed for regression testing
        pass

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires HPO data - run manually")
    def test_benchmark_visualization_pipeline(self, benchmark_data_dir, tmp_path):
        """Test benchmark visualization generation (end-to-end).

        NOTE: Requires HPO data. Unskip for manual testing.
        """
        # This would test visualization generation
        # Implement when needed for regression testing
        pass
