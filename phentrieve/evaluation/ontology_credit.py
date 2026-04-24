from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

from phentrieve.evaluation.metrics import (
    SimilarityFormula,
    calculate_semantic_similarity,
    load_hpo_graph_data,
)


class MatchKind(str, Enum):  # noqa: UP042 - Python 3.10 compatible enum.
    EXACT = "exact"
    DESCENDANT = "descendant"
    ANCESTOR = "ancestor"
    SIBLING = "sibling"
    COUSIN = "cousin"
    SEMANTIC = "semantic"
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class OntologyCreditConfig:
    semantic_floor: float = 0.30
    descendant_base: float = 0.95
    descendant_step_penalty: float = 0.05
    descendant_min: float = 0.75
    ancestor_base: float = 0.85
    ancestor_step_penalty: float = 0.08
    ancestor_min: float = 0.50
    sibling_min: float = 0.65
    cousin_min: float = 0.45
    similarity_formula: SimilarityFormula = SimilarityFormula.HYBRID


@dataclass(frozen=True)
class PairCredit:
    predicted_id: str
    gold_id: str
    credit: float
    match_kind: MatchKind
    semantic_similarity: float
    distance: int | None


def calculate_pair_credit(
    predicted_id: str,
    gold_id: str,
    config: OntologyCreditConfig | None = None,
) -> PairCredit:
    config = config or OntologyCreditConfig()
    ancestors, depths = load_hpo_graph_data()
    if predicted_id == gold_id:
        return PairCredit(predicted_id, gold_id, 1.0, MatchKind.EXACT, 1.0, 0)

    distance = _ontology_distance(predicted_id, gold_id, ancestors, depths)
    if _is_descendant(predicted_id, gold_id, ancestors):
        credit = max(
            config.descendant_min,
            config.descendant_base
            - config.descendant_step_penalty * ((distance or 1) - 1),
        )
        return PairCredit(
            predicted_id, gold_id, credit, MatchKind.DESCENDANT, 0.0, distance
        )
    if _is_descendant(gold_id, predicted_id, ancestors):
        credit = max(
            config.ancestor_min,
            config.ancestor_base - config.ancestor_step_penalty * ((distance or 1) - 1),
        )
        return PairCredit(
            predicted_id, gold_id, credit, MatchKind.ANCESTOR, 0.0, distance
        )

    similarity = _safe_similarity(predicted_id, gold_id, config)
    relationship = _sibling_or_cousin(predicted_id, gold_id, ancestors, depths)
    if relationship == MatchKind.SIBLING:
        return PairCredit(
            predicted_id,
            gold_id,
            max(similarity, config.sibling_min),
            MatchKind.SIBLING,
            similarity,
            distance,
        )
    if relationship == MatchKind.COUSIN:
        return PairCredit(
            predicted_id,
            gold_id,
            max(similarity, config.cousin_min),
            MatchKind.COUSIN,
            similarity,
            distance,
        )
    if similarity >= config.semantic_floor:
        return PairCredit(
            predicted_id, gold_id, similarity, MatchKind.SEMANTIC, similarity, distance
        )
    return PairCredit(
        predicted_id, gold_id, 0.0, MatchKind.UNRELATED, similarity, distance
    )


def _is_descendant(
    child: str,
    parent: str,
    ancestors: dict[str, set[str]],
) -> bool:
    return child != parent and parent in ancestors.get(child, set())


def _ontology_distance(
    term_a: str,
    term_b: str,
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
) -> int | None:
    distances_a = _ancestor_distances(term_a, ancestors, depths)
    distances_b = _ancestor_distances(term_b, ancestors, depths)
    if not distances_a or not distances_b:
        return None

    common_ancestors = set(distances_a).intersection(distances_b)
    if not common_ancestors:
        return None

    return min(
        distances_a[ancestor] + distances_b[ancestor] for ancestor in common_ancestors
    )


def _ancestor_distances(
    term_id: str,
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
) -> dict[str, int]:
    if term_id not in ancestors:
        return {}

    distances = {term_id: 0}
    queue = deque([term_id])
    while queue:
        current = queue.popleft()
        for parent in _parents_of(current, ancestors, depths):
            if parent in distances:
                continue
            distances[parent] = distances[current] + 1
            queue.append(parent)

    return distances


def _parents_of(
    term_id: str,
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
) -> set[str]:
    proper_ancestors = ancestors.get(term_id, set()) - {term_id}
    if not proper_ancestors:
        return set()

    return {
        ancestor
        for ancestor in proper_ancestors
        if not any(
            ancestor in ancestors.get(other_ancestor, set())
            for other_ancestor in proper_ancestors
            if other_ancestor != ancestor
        )
    }


def _sibling_or_cousin(
    term_a: str,
    term_b: str,
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
) -> MatchKind | None:
    parents_a = _parents_of(term_a, ancestors, depths)
    parents_b = _parents_of(term_b, ancestors, depths)
    if not parents_a or not parents_b:
        return None

    if parents_a.intersection(parents_b):
        return MatchKind.SIBLING

    for parent_a in parents_a:
        grand_parents_a = _parents_of(parent_a, ancestors, depths)
        if not grand_parents_a:
            continue
        for parent_b in parents_b:
            if grand_parents_a.intersection(_parents_of(parent_b, ancestors, depths)):
                return MatchKind.COUSIN

    return None


def _safe_similarity(
    predicted_id: str,
    gold_id: str,
    config: OntologyCreditConfig,
) -> float:
    try:
        similarity = calculate_semantic_similarity(
            predicted_id,
            gold_id,
            config.similarity_formula,
        )
    except Exception:
        return 0.0

    if not math.isfinite(similarity):
        return 0.0
    return max(0.0, min(1.0, similarity))
