from phentrieve.evaluation.extraction_metrics import ExtractionResult
from phentrieve.evaluation.ontology_credit import MatchKind, PairCredit
from phentrieve.evaluation.ontology_matching import (
    calculate_document_ontology_metrics,
)


def _fake_credit(
    predicted_id: str,
    gold_id: str,
    credit: float,
    match_kind: str,
) -> PairCredit:
    return PairCredit(
        predicted_id=predicted_id,
        gold_id=gold_id,
        credit=credit,
        match_kind=MatchKind(match_kind),
        semantic_similarity=credit,
        distance=None,
    )


def test_exact_document_match_scores_one(monkeypatch):
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:1", "PRESENT")],
        gold=[("HP:1", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_precision == 1.0
    assert metrics.soft_recall == 1.0
    assert metrics.soft_f1 == 1.0
    assert metrics.soft_tp == 1.0


def test_partial_child_match_scores_fraction(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_matching.calculate_pair_credit",
        lambda pred, gold, config=None: _fake_credit(pred, gold, 0.95, "descendant"),
    )
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:child", "PRESENT")],
        gold=[("HP:parent", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_tp == 0.95
    assert metrics.soft_fp == 0.05
    assert metrics.soft_fn == 0.05
    assert metrics.soft_f1 == 0.95


def test_assertion_mismatch_gets_no_credit(monkeypatch):
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:1", "ABSENT")],
        gold=[("HP:1", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_tp == 0.0
    assert metrics.soft_precision == 0.0
    assert metrics.soft_recall == 0.0


def test_one_prediction_cannot_satisfy_two_gold_terms(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_matching.calculate_pair_credit",
        lambda pred, gold, config=None: _fake_credit(pred, gold, 0.9, "semantic"),
    )
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:p", "PRESENT")],
        gold=[("HP:g1", "PRESENT"), ("HP:g2", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_tp == 0.9
    assert metrics.soft_recall == 0.45
    assert metrics.partial_recall == 0.9


def test_partial_recall_uses_prediction_to_gold_credit_direction(monkeypatch):
    def directional_credit(pred, gold, config=None):
        if pred == "HP:child" and gold == "HP:parent":
            return _fake_credit(pred, gold, 0.95, "descendant")
        return _fake_credit(pred, gold, 0.50, "ancestor")

    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_matching.calculate_pair_credit",
        directional_credit,
    )
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:child", "PRESENT")],
        gold=[("HP:parent", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.partial_recall == 0.95
