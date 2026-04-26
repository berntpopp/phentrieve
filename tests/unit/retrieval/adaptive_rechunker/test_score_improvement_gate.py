"""Tests for apply_score_improvement_gate."""

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    apply_score_improvement_gate,
)


def test_all_children_below_gate_revert():
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    parent_top_1 = {0: 0.5}  # parent_idx 0 had top_1 = 0.5
    child_top_1 = {1: 0.51, 2: 0.49}  # neither beats parent + 0.05 = 0.55

    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: [1, 2]},
        parent_top_1=parent_top_1,
        child_top_1=child_top_1,
        config=config,
    )
    assert revert == {0}
    assert keep == set()


def test_one_child_above_gate_keep():
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    parent_top_1 = {0: 0.5}
    child_top_1 = {1: 0.4, 2: 0.7}  # child 2 beats 0.5 + 0.05 = 0.55

    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: [1, 2]},
        parent_top_1=parent_top_1,
        child_top_1=child_top_1,
        config=config,
    )
    assert revert == set()
    assert keep == {0}


def test_multiple_parents_mixed_outcome():
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    parent_top_1 = {0: 0.5, 1: 0.6}
    child_top_1 = {2: 0.45, 3: 0.46, 4: 0.7, 5: 0.71}
    # Parent 0 (children 2,3): max(0.46) < 0.55 -> revert.
    # Parent 1 (children 4,5): max(0.71) > 0.65 -> keep.
    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: [2, 3], 1: [4, 5]},
        parent_top_1=parent_top_1,
        child_top_1=child_top_1,
        config=config,
    )
    assert revert == {0}
    assert keep == {1}


def test_parent_with_no_children_skipped():
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: []},
        parent_top_1={0: 0.5},
        child_top_1={},
        config=config,
    )
    # No children -> no decision, parent stays as-is.
    assert revert == set()
    assert keep == set()


def test_parent_missing_top_1_skipped():
    """Defensive: if a parent index has no recorded top_1, skip it."""
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: [1]},
        parent_top_1={},  # parent 0 has no entry
        child_top_1={1: 0.9},
        config=config,
    )
    assert revert == set()
    assert keep == set()


def test_children_missing_scores_revert():
    """If child top_1 entries are missing, treat as revert (no improvement)."""
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: [1, 2]},
        parent_top_1={0: 0.5},
        child_top_1={},  # no scores for any child
        config=config,
    )
    assert revert == {0}
    assert keep == set()
