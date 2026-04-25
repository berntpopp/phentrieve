"""Tests that OrchestrationResult supports both legacy 2-tuple unpacking
and modern attribute access. Legacy unpacking is what keeps existing call
sites working."""

import pytest


class TestOrchestrationResult:
    def test_legacy_2_tuple_unpack_works(self):
        from phentrieve.text_processing.orchestration_result import (
            OrchestrationResult,
        )

        result = OrchestrationResult(
            aggregated_results=[{"id": "HP:0001"}],
            chunk_results=[{"chunk_idx": 0}],
            raw_query_results=[{"similarities": [[0.9]]}],
        )
        # The legacy call sites do this:
        agg, chunks = result
        assert agg == [{"id": "HP:0001"}]
        assert chunks == [{"chunk_idx": 0}]

    def test_attribute_access_works(self):
        from phentrieve.text_processing.orchestration_result import (
            OrchestrationResult,
        )

        result = OrchestrationResult(
            aggregated_results=[],
            chunk_results=[],
            raw_query_results=[{"similarities": [[]]}],
        )
        assert result.aggregated_results == []
        assert result.chunk_results == []
        assert result.raw_query_results == [{"similarities": [[]]}]

    def test_indexing_returns_legacy_2_tuple_elements(self):
        from phentrieve.text_processing.orchestration_result import (
            OrchestrationResult,
        )

        result = OrchestrationResult(
            aggregated_results=[1, 2],
            chunk_results=[3, 4],
            raw_query_results=[5, 6],
        )
        # Some call sites may index instead of unpack.
        assert result[0] == [1, 2]
        assert result[1] == [3, 4]

    def test_iteration_yields_2_tuple(self):
        from phentrieve.text_processing.orchestration_result import (
            OrchestrationResult,
        )

        result = OrchestrationResult([1], [2], [3])
        assert list(result) == [[1], [2]]

    def test_immutable(self):
        from phentrieve.text_processing.orchestration_result import (
            OrchestrationResult,
        )

        result = OrchestrationResult([], [], [])
        with pytest.raises(AttributeError):
            result.aggregated_results = []  # frozen=True
