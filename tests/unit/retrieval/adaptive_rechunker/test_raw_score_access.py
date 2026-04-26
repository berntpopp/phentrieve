"""Critical invariant: assess_chunk_quality reads from raw query_batch
output, NOT from threshold-filtered chunk_results. Without this test, a
regression to filtered input could silently break the trigger for the
exact case adaptive rechunking is designed to handle."""

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    assess_chunk_quality,
)


class TestRawScoreAccess:
    def test_genuinely_poor_chunk_detectable(self):
        """Simulates a chunk where every retrieval result was below
        chunk_retrieval_threshold. orchestrate_hpo_extraction would put
        an empty `matches` list into chunk_results - but raw_query_results
        still has the (low) similarities."""
        config = AdaptiveRechunkingConfig(quality_threshold=0.55, margin_threshold=0.03)

        # All similarities below the standard chunk_retrieval_threshold of 0.7.
        # The corresponding chunk_results entry would have matches=[].
        raw = {
            "ids": [["HP:0001", "HP:0002"]],
            "metadatas": [[{"id": "HP:0001"}, {"id": "HP:0002"}]],
            "similarities": [[0.4, 0.39]],  # Both below 0.55, low margin.
            "distances": [[0.6, 0.61]],
            "documents": [["", ""]],
        }

        sig = assess_chunk_quality(
            raw_query_result=raw,
            chunk_idx=0,
            chunk_retrieval_threshold=0.7,  # all matches below this!
            config=config,
        )

        # The trigger MUST fire - both score and margin are bad.
        assert sig.is_poor is True
        assert sig.reason == "low_margin"
        # And n_matches_above_threshold reports the chunk_results filter result.
        assert sig.n_matches_above_threshold == 0  # nothing above 0.7

    def test_top_2_present_in_raw_even_if_filtered_out_of_chunk_results(self):
        """When chunk_retrieval_threshold = 0.7 and similarities = [0.6, 0.5],
        chunk_results[0]['matches'] is empty. But raw still has both scores.
        """
        config = AdaptiveRechunkingConfig(quality_threshold=0.55, margin_threshold=0.03)
        raw = {
            "ids": [["HP:0001", "HP:0002"]],
            "metadatas": [[{"id": "HP:0001"}, {"id": "HP:0002"}]],
            "similarities": [[0.6, 0.5]],
            "distances": [[0.4, 0.5]],
            "documents": [["", ""]],
        }
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        # top_1=0.6 >= quality_threshold 0.55, so we trust it.
        assert sig.is_poor is False
        assert sig.top_2 == 0.5
        assert sig.margin == pytest.approx(0.1)
