"""Tests for assess_chunk_quality.

The trigger conjunction:
    is_poor = top_1 < quality_threshold AND (margin < margin_threshold OR top_2 is None)
"""

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    ChunkQualitySignals,
    assess_chunk_quality,
)


def make_raw_result(similarities: list[float]) -> dict:
    """Helper to build a raw query_batch output entry."""
    return {
        "ids": [[f"HP:{i:07d}" for i in range(len(similarities))]],
        "metadatas": [[{"id": f"HP:{i:07d}"} for i in range(len(similarities))]],
        "similarities": [similarities],
        "distances": [[1 - s for s in similarities]],
        "documents": [[""] * len(similarities)],
    }


@pytest.fixture
def config():
    return AdaptiveRechunkingConfig(quality_threshold=0.55, margin_threshold=0.03)


class TestAssessChunkQuality:
    def test_empty_raw_results_is_poor_no_matches(self, config):
        raw = make_raw_result([])
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        assert sig.is_poor is True
        assert sig.reason == "no_matches"
        assert sig.top_1 is None

    def test_single_match_above_threshold_is_ok(self, config):
        raw = make_raw_result([0.9])
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        # top_2 is None - but top_1 above quality_threshold means we trust it.
        assert sig.is_poor is False
        assert sig.reason == "ok"

    def test_single_match_below_threshold_is_poor_low_score(self, config):
        raw = make_raw_result([0.4])
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        # top_1 < 0.55 AND top_2 is None -> poor.
        assert sig.is_poor is True
        assert sig.reason == "low_score"

    def test_two_matches_above_threshold_with_low_margin_ok(self, config):
        # top_1=0.9, top_2=0.89, margin=0.01 < 0.03.
        # But top_1 >= quality_threshold so we trust the result regardless.
        raw = make_raw_result([0.9, 0.89])
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        assert sig.is_poor is False
        assert sig.reason == "ok"

    def test_two_matches_low_score_high_margin_ok(self, config):
        # top_1=0.5, top_2=0.1, margin=0.4 >= 0.03.
        # Conjunction: top_1 < 0.55 AND (margin < 0.03 OR top_2 is None).
        # margin is high -> second condition false -> not poor.
        raw = make_raw_result([0.5, 0.1])
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        assert sig.is_poor is False
        assert sig.reason == "ok"

    def test_two_matches_low_score_low_margin_is_poor_low_margin(self, config):
        # top_1=0.5, top_2=0.49, margin=0.01 < 0.03. Both bad -> poor.
        raw = make_raw_result([0.5, 0.49])
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        assert sig.is_poor is True
        assert sig.reason == "low_margin"

    def test_signals_record_top_1_top_2_margin(self, config):
        raw = make_raw_result([0.4, 0.38, 0.35])
        sig = assess_chunk_quality(
            raw, chunk_idx=2, chunk_retrieval_threshold=0.7, config=config
        )
        assert sig.chunk_idx == 2
        assert sig.top_1 == 0.4
        assert sig.top_2 == 0.38
        assert sig.margin == pytest.approx(0.02)

    def test_n_matches_above_threshold_informational(self, config):
        # chunk_retrieval_threshold filters chunk_results, but
        # assess_chunk_quality reads raw - n_matches_above_threshold is informational.
        raw = make_raw_result([0.9, 0.85, 0.5, 0.4])  # 2 above 0.7
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        assert sig.n_matches_above_threshold == 2

    def test_chunk_quality_signals_is_frozen(self):
        from dataclasses import FrozenInstanceError

        sig = ChunkQualitySignals(
            chunk_idx=0,
            top_1=0.5,
            top_2=0.4,
            margin=0.1,
            n_matches_above_threshold=0,
            is_poor=False,
            reason="ok",
        )
        with pytest.raises(FrozenInstanceError):
            sig.chunk_idx = 1  # type: ignore[misc]
