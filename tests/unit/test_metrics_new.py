"""
Unit tests for new MTEB-aligned metrics (NDCG, Recall, Precision, MAP).

Tests cover:
- NDCG@K calculation with proper DCG/IDCG discounting
- Recall@K for retrieval completeness
- Precision@K for retrieval efficiency
- MAP@K (Mean Average Precision)
- Edge cases (empty results, no matches, perfect ranking)
"""

import pytest

from phentrieve.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@pytest.fixture
def mock_perfect_results():
    """Perfect retrieval: all relevant items at top ranks."""
    return {
        "metadatas": [
            [
                {"hpo_id": "HP:0001250"},  # Rank 1 - relevant
                {"hpo_id": "HP:0002367"},  # Rank 2 - relevant
                {"hpo_id": "HP:0001251"},  # Rank 3 - relevant
                {"hpo_id": "HP:0000001"},  # Rank 4 - irrelevant
                {"hpo_id": "HP:0000002"},  # Rank 5 - irrelevant
            ]
        ],
        "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],  # Cosine distances (lower = better)
    }


@pytest.fixture
def mock_imperfect_results():
    """Imperfect retrieval: relevant items scattered across ranks."""
    return {
        "metadatas": [
            [
                {"hpo_id": "HP:0001250"},  # Rank 1 - relevant
                {"hpo_id": "HP:0000001"},  # Rank 2 - irrelevant
                {"hpo_id": "HP:0000002"},  # Rank 3 - irrelevant
                {"hpo_id": "HP:0002367"},  # Rank 4 - relevant
                {"hpo_id": "HP:0001251"},  # Rank 5 - relevant
            ]
        ],
        "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
    }


@pytest.fixture
def mock_no_match_results():
    """No relevant items retrieved."""
    return {
        "metadatas": [
            [
                {"hpo_id": "HP:0000001"},
                {"hpo_id": "HP:0000002"},
                {"hpo_id": "HP:0000003"},
            ]
        ],
        "distances": [[0.1, 0.2, 0.3]],
    }


@pytest.fixture
def expected_ids():
    """Expected relevant HPO IDs for testing."""
    return ["HP:0001250", "HP:0002367", "HP:0001251"]


# ============================================================================
# NDCG@K Tests
# ============================================================================


def test_ndcg_at_k_perfect_ranking(mock_perfect_results, expected_ids):
    """NDCG@3 should be 1.0 for perfect ranking (all relevant at top)."""
    ndcg = ndcg_at_k(mock_perfect_results, expected_ids, k=3)
    assert ndcg == pytest.approx(1.0, abs=1e-6)


def test_ndcg_at_k_imperfect_ranking(mock_imperfect_results, expected_ids):
    """NDCG@5 should be < 1.0 for imperfect ranking."""
    ndcg = ndcg_at_k(mock_imperfect_results, expected_ids, k=5)
    assert 0.0 < ndcg < 1.0  # Should be between 0 and 1
    # Expected: DCG penalizes items at lower ranks more heavily


def test_ndcg_at_k_small_k(mock_perfect_results, expected_ids):
    """NDCG@1 should only consider first result."""
    ndcg = ndcg_at_k(mock_perfect_results, expected_ids, k=1)
    # First result is relevant, so NDCG@1 = 1.0
    assert ndcg == pytest.approx(1.0, abs=1e-6)


def test_ndcg_at_k_no_relevant_found(mock_no_match_results, expected_ids):
    """NDCG@K should be 0.0 when no relevant items found."""
    ndcg = ndcg_at_k(mock_no_match_results, expected_ids, k=3)
    assert ndcg == 0.0


def test_ndcg_at_k_empty_results():
    """NDCG@K should be 0.0 for empty results."""
    empty_results = {"metadatas": [[]], "distances": [[]]}
    ndcg = ndcg_at_k(empty_results, ["HP:0001250"], k=3)
    assert ndcg == 0.0


def test_ndcg_at_k_empty_expected_ids(mock_perfect_results):
    """NDCG@K should be 0.0 when no expected IDs provided."""
    ndcg = ndcg_at_k(mock_perfect_results, [], k=3)
    assert ndcg == 0.0


def test_ndcg_at_k_with_graded_relevance(mock_perfect_results):
    """NDCG@K should support graded relevance scores."""
    expected_ids = ["HP:0001250", "HP:0002367", "HP:0001251"]
    relevance_scores = {
        "HP:0001250": 3.0,  # Highly relevant
        "HP:0002367": 2.0,  # Moderately relevant
        "HP:0001251": 1.0,  # Somewhat relevant
    }
    ndcg = ndcg_at_k(
        mock_perfect_results, expected_ids, k=3, relevance_scores=relevance_scores
    )
    # Should still be 1.0 for perfect ranking with graded relevance
    assert ndcg == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# Recall@K Tests
# ============================================================================


def test_recall_at_k_perfect(mock_perfect_results, expected_ids):
    """Recall@3 should be 1.0 when all 3 relevant items in top 3."""
    recall = recall_at_k(mock_perfect_results, expected_ids, k=3)
    assert recall == pytest.approx(1.0, abs=1e-6)


def test_recall_at_k_partial(mock_imperfect_results, expected_ids):
    """Recall@3 should be 1/3 when only 1 of 3 relevant items in top 3."""
    recall = recall_at_k(mock_imperfect_results, expected_ids, k=3)
    assert recall == pytest.approx(1 / 3, abs=1e-6)


def test_recall_at_k_full_retrieval(mock_imperfect_results, expected_ids):
    """Recall@5 should be 1.0 when all relevant items retrieved by rank 5."""
    recall = recall_at_k(mock_imperfect_results, expected_ids, k=5)
    assert recall == pytest.approx(1.0, abs=1e-6)


def test_recall_at_k_zero(mock_no_match_results, expected_ids):
    """Recall@K should be 0.0 when no relevant items found."""
    recall = recall_at_k(mock_no_match_results, expected_ids, k=3)
    assert recall == 0.0


def test_recall_at_k_empty_results():
    """Recall@K should be 0.0 for empty results."""
    empty_results = {"metadatas": [[]], "distances": [[]]}
    recall = recall_at_k(empty_results, ["HP:0001250"], k=3)
    assert recall == 0.0


def test_recall_at_k_k_larger_than_results(mock_perfect_results, expected_ids):
    """Recall@K should work when K > number of results."""
    recall = recall_at_k(mock_perfect_results, expected_ids, k=100)
    # Only 3 relevant items exist in 5 results
    assert recall == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# Precision@K Tests
# ============================================================================


def test_precision_at_k_perfect(mock_perfect_results, expected_ids):
    """Precision@3 should be 1.0 when all top 3 results are relevant."""
    precision = precision_at_k(mock_perfect_results, expected_ids, k=3)
    assert precision == pytest.approx(1.0, abs=1e-6)


def test_precision_at_k_imperfect(mock_imperfect_results, expected_ids):
    """Precision@3 should be 1/3 when only 1 of top 3 is relevant."""
    precision = precision_at_k(mock_imperfect_results, expected_ids, k=3)
    assert precision == pytest.approx(1 / 3, abs=1e-6)


def test_precision_at_k_zero(mock_no_match_results, expected_ids):
    """Precision@K should be 0.0 when no relevant items found."""
    precision = precision_at_k(mock_no_match_results, expected_ids, k=3)
    assert precision == 0.0


def test_precision_at_k_empty_results():
    """Precision@K should be 0.0 for empty results."""
    empty_results = {"metadatas": [[]], "distances": [[]]}
    precision = precision_at_k(empty_results, ["HP:0001250"], k=3)
    assert precision == 0.0


def test_precision_at_k_k_equals_one(mock_perfect_results, expected_ids):
    """Precision@1 should only consider first result."""
    precision = precision_at_k(mock_perfect_results, expected_ids, k=1)
    # First result is relevant
    assert precision == pytest.approx(1.0, abs=1e-6)


def test_precision_at_k_with_extra_results(mock_perfect_results, expected_ids):
    """Precision@5 should decrease when irrelevant items in top 5."""
    precision = precision_at_k(mock_perfect_results, expected_ids, k=5)
    # 3 relevant out of 5 results
    assert precision == pytest.approx(3 / 5, abs=1e-6)


# ============================================================================
# MAP@K (Average Precision) Tests
# ============================================================================


def test_map_at_k_perfect_ranking(mock_perfect_results, expected_ids):
    """MAP@3 should be 1.0 for perfect ranking."""
    map_score = average_precision_at_k(mock_perfect_results, expected_ids, k=3)
    # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
    assert map_score == pytest.approx(1.0, abs=1e-6)


def test_map_at_k_imperfect_ranking(mock_imperfect_results, expected_ids):
    """MAP@5 should reflect precision at each relevant item."""
    map_score = average_precision_at_k(mock_imperfect_results, expected_ids, k=5)
    # Relevant at ranks: 1, 4, 5
    # Precisions: 1/1, 2/4, 3/5
    # AP = (1.0 + 0.5 + 0.6) / 3 = 0.7
    assert map_score == pytest.approx(0.7, abs=1e-6)


def test_map_at_k_no_relevant_found(mock_no_match_results, expected_ids):
    """MAP@K should be 0.0 when no relevant items found."""
    map_score = average_precision_at_k(mock_no_match_results, expected_ids, k=3)
    assert map_score == 0.0


def test_map_at_k_empty_results():
    """MAP@K should be 0.0 for empty results."""
    empty_results = {"metadatas": [[]], "distances": [[]]}
    map_score = average_precision_at_k(empty_results, ["HP:0001250"], k=3)
    assert map_score == 0.0


def test_map_at_k_single_relevant(mock_perfect_results):
    """MAP@K should work with single expected ID."""
    single_expected = ["HP:0001250"]
    map_score = average_precision_at_k(mock_perfect_results, single_expected, k=3)
    # Only first result is relevant: AP = 1/1 / 1 = 1.0
    assert map_score == pytest.approx(1.0, abs=1e-6)


def test_map_at_k_k_smaller_than_relevant(mock_perfect_results):
    """MAP@K should only consider top K results."""
    expected_ids = ["HP:0001250", "HP:0002367", "HP:0001251"]
    map_score = average_precision_at_k(mock_perfect_results, expected_ids, k=2)
    # Only first 2 relevant items considered
    # AP = (1/1 + 2/2) / 2 = 1.0
    assert map_score == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# Integration Tests - Multiple Metrics
# ============================================================================


def test_metrics_consistency_perfect_ranking(mock_perfect_results, expected_ids):
    """All metrics should indicate perfect performance for perfect ranking."""
    ndcg = ndcg_at_k(mock_perfect_results, expected_ids, k=3)
    recall = recall_at_k(mock_perfect_results, expected_ids, k=3)
    precision = precision_at_k(mock_perfect_results, expected_ids, k=3)
    map_score = average_precision_at_k(mock_perfect_results, expected_ids, k=3)

    # All should be 1.0 for perfect ranking
    assert ndcg == pytest.approx(1.0, abs=1e-6)
    assert recall == pytest.approx(1.0, abs=1e-6)
    assert precision == pytest.approx(1.0, abs=1e-6)
    assert map_score == pytest.approx(1.0, abs=1e-6)


def test_metrics_consistency_no_match(mock_no_match_results, expected_ids):
    """All metrics should be 0.0 when no relevant items found."""
    ndcg = ndcg_at_k(mock_no_match_results, expected_ids, k=3)
    recall = recall_at_k(mock_no_match_results, expected_ids, k=3)
    precision = precision_at_k(mock_no_match_results, expected_ids, k=3)
    map_score = average_precision_at_k(mock_no_match_results, expected_ids, k=3)

    assert ndcg == 0.0
    assert recall == 0.0
    assert precision == 0.0
    assert map_score == 0.0


def test_metrics_increase_with_k(mock_imperfect_results, expected_ids):
    """Recall should increase as K increases (more chance to find relevant items)."""
    recall_1 = recall_at_k(mock_imperfect_results, expected_ids, k=1)
    recall_3 = recall_at_k(mock_imperfect_results, expected_ids, k=3)
    recall_5 = recall_at_k(mock_imperfect_results, expected_ids, k=5)

    # Recall should monotonically increase with K
    assert recall_1 <= recall_3 <= recall_5
