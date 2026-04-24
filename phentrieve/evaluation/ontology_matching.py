from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
from networkx.algorithms.matching import max_weight_matching

from phentrieve.evaluation.extraction_metrics import ExtractionResult
from phentrieve.evaluation.ontology_credit import (
    OntologyCreditConfig,
    PairCredit,
    calculate_pair_credit,
)

Annotation = tuple[str, str]


@dataclass(frozen=True)
class MatchedPair:
    predicted: Annotation
    gold: Annotation
    credit: PairCredit


@dataclass(frozen=True)
class DocumentOntologyMetrics:
    doc_id: str
    prediction_count: int
    gold_count: int
    strict_tp: int
    soft_tp: float
    soft_fp: float
    soft_fn: float
    soft_precision: float
    soft_recall: float
    soft_f1: float
    partial_precision: float
    partial_recall: float
    partial_f1: float
    matches: list[MatchedPair]
    unmatched_predictions: list[Annotation]
    unmatched_gold: list[Annotation]


def calculate_document_ontology_metrics(
    result: ExtractionResult,
    config: OntologyCreditConfig | None = None,
) -> DocumentOntologyMetrics:
    predictions = _unique_annotations(result.predicted)
    gold = _unique_annotations(result.gold)

    exact_matches = set(predictions).intersection(gold)
    matches = [
        MatchedPair(
            predicted=annotation,
            gold=annotation,
            credit=calculate_pair_credit(annotation[0], annotation[0], config),
        )
        for annotation in predictions
        if annotation in exact_matches
    ]

    unmatched_predictions = [
        annotation for annotation in predictions if annotation not in exact_matches
    ]
    unmatched_gold = [
        annotation for annotation in gold if annotation not in exact_matches
    ]

    matches.extend(
        _match_remaining_by_assertion(unmatched_predictions, unmatched_gold, config)
    )

    matched_predictions = {match.predicted for match in matches}
    matched_gold = {match.gold for match in matches}
    final_unmatched_predictions = [
        annotation
        for annotation in predictions
        if annotation not in matched_predictions
    ]
    final_unmatched_gold = [
        annotation for annotation in gold if annotation not in matched_gold
    ]

    prediction_count = len(predictions)
    gold_count = len(gold)
    soft_tp = _clean_float(sum(match.credit.credit for match in matches))
    soft_fp = _clean_float(prediction_count - soft_tp)
    soft_fn = _clean_float(gold_count - soft_tp)
    soft_precision, soft_recall, soft_f1 = _prf(soft_tp, soft_fp, soft_fn)
    partial_precision, partial_recall, partial_f1 = _partial_diagnostics(
        predictions,
        gold,
        config,
    )

    return DocumentOntologyMetrics(
        doc_id=result.doc_id,
        prediction_count=prediction_count,
        gold_count=gold_count,
        strict_tp=len(exact_matches),
        soft_tp=soft_tp,
        soft_fp=soft_fp,
        soft_fn=soft_fn,
        soft_precision=soft_precision,
        soft_recall=soft_recall,
        soft_f1=soft_f1,
        partial_precision=partial_precision,
        partial_recall=partial_recall,
        partial_f1=partial_f1,
        matches=matches,
        unmatched_predictions=final_unmatched_predictions,
        unmatched_gold=final_unmatched_gold,
    )


def _unique_annotations(annotations: list[Annotation]) -> list[Annotation]:
    return list(dict.fromkeys(annotations))


def _match_remaining_by_assertion(
    predictions: list[Annotation],
    gold: list[Annotation],
    config: OntologyCreditConfig | None,
) -> list[MatchedPair]:
    predictions_by_assertion = _group_by_assertion(predictions)
    gold_by_assertion = _group_by_assertion(gold)
    matches: list[MatchedPair] = []

    for assertion in predictions_by_assertion.keys() & gold_by_assertion.keys():
        matches.extend(
            _maximum_weight_matches(
                predictions_by_assertion[assertion],
                gold_by_assertion[assertion],
                config,
            )
        )

    return matches


def _maximum_weight_matches(
    predictions: list[Annotation],
    gold: list[Annotation],
    config: OntologyCreditConfig | None,
) -> list[MatchedPair]:
    graph = nx.Graph()
    for pred_index, prediction in enumerate(predictions):
        pred_node = ("pred", pred_index)
        graph.add_node(pred_node, bipartite=0)
        for gold_index, gold_annotation in enumerate(gold):
            gold_node = ("gold", gold_index)
            graph.add_node(gold_node, bipartite=1)
            credit = calculate_pair_credit(prediction[0], gold_annotation[0], config)
            if credit.credit <= 0.0:
                continue
            graph.add_edge(
                pred_node,
                gold_node,
                weight=credit.credit,
                predicted=prediction,
                gold=gold_annotation,
                credit=credit,
            )

    matched_pairs: list[MatchedPair] = []
    for node_a, node_b in max_weight_matching(graph, weight="weight"):
        edge_data = graph[node_a][node_b]
        matched_pairs.append(
            MatchedPair(
                predicted=edge_data["predicted"],
                gold=edge_data["gold"],
                credit=edge_data["credit"],
            )
        )

    return matched_pairs


def _group_by_assertion(annotations: list[Annotation]) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation[1]].append(annotation)
    return dict(grouped)


def _partial_diagnostics(
    predictions: list[Annotation],
    gold: list[Annotation],
    config: OntologyCreditConfig | None,
) -> tuple[float, float, float]:
    gold_by_assertion = _group_by_assertion(gold)
    predictions_by_assertion = _group_by_assertion(predictions)

    prediction_scores = [
        _best_credit(prediction, gold_by_assertion.get(prediction[1], []), config)
        for prediction in predictions
    ]
    gold_scores = [
        _best_recall_credit(
            predictions_by_assertion.get(gold_annotation[1], []),
            gold_annotation,
            config,
        )
        for gold_annotation in gold
    ]

    partial_precision = _mean(prediction_scores)
    partial_recall = _mean(gold_scores)
    partial_f1 = _harmonic_mean(partial_precision, partial_recall)
    return partial_precision, partial_recall, partial_f1


def _best_credit(
    source: Annotation,
    candidates: list[Annotation],
    config: OntologyCreditConfig | None,
) -> float:
    if not candidates:
        return 0.0
    return _clean_float(
        max(
            calculate_pair_credit(source[0], candidate[0], config).credit
            for candidate in candidates
        )
    )


def _best_recall_credit(
    prediction_candidates: list[Annotation],
    gold: Annotation,
    config: OntologyCreditConfig | None,
) -> float:
    if not prediction_candidates:
        return 0.0
    return _clean_float(
        max(
            calculate_pair_credit(prediction[0], gold[0], config).credit
            for prediction in prediction_candidates
        )
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _clean_float(sum(values) / len(values))


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return (
        _clean_float(precision),
        _clean_float(recall),
        _harmonic_mean(precision, recall),
    )


def _harmonic_mean(value_a: float, value_b: float) -> float:
    if value_a + value_b == 0:
        return 0.0
    return _clean_float(2 * value_a * value_b / (value_a + value_b))


def _clean_float(value: float) -> float:
    return round(value, 12)
