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


def test_exact_document_match_does_not_calculate_pair_credit(monkeypatch):
    def fail_pair_credit(pred, gold, config=None):
        raise AssertionError("exact match should not calculate pair credit")

    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_matching.calculate_pair_credit",
        fail_pair_credit,
    )
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:1", "PRESENT")],
        gold=[("HP:1", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_tp == 1.0
    assert metrics.matches[0].credit.match_kind == MatchKind.EXACT
    assert metrics.matches[0].credit.semantic_similarity == 1.0
    assert metrics.matches[0].credit.distance == 0


def test_empty_document_scores_zero():
    result = ExtractionResult(doc_id="doc", predicted=[], gold=[])

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.prediction_count == 0
    assert metrics.gold_count == 0
    assert metrics.strict_tp == 0
    assert metrics.soft_tp == 0.0
    assert metrics.soft_fp == 0.0
    assert metrics.soft_fn == 0.0
    assert metrics.soft_precision == 0.0
    assert metrics.soft_recall == 0.0
    assert metrics.soft_f1 == 0.0
    assert metrics.partial_precision == 0.0
    assert metrics.partial_recall == 0.0
    assert metrics.partial_f1 == 0.0
    assert metrics.matches == []
    assert metrics.unmatched_predictions == []
    assert metrics.unmatched_gold == []


def test_duplicate_annotations_are_deduplicated_consistently(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_matching.calculate_pair_credit",
        lambda pred, gold, config=None: _fake_credit(pred, gold, 0.0, "unrelated"),
    )
    result = ExtractionResult(
        doc_id="doc",
        predicted=[
            ("HP:2", "PRESENT"),
            ("HP:1", "PRESENT"),
            ("HP:1", "PRESENT"),
        ],
        gold=[
            ("HP:3", "PRESENT"),
            ("HP:1", "PRESENT"),
            ("HP:1", "PRESENT"),
        ],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.prediction_count == 2
    assert metrics.gold_count == 2
    assert metrics.strict_tp == 1
    assert metrics.soft_tp == 1.0
    assert metrics.soft_fp == 1.0
    assert metrics.soft_fn == 1.0
    assert len(metrics.matches) == 1
    assert metrics.matches[0].predicted == ("HP:1", "PRESENT")
    assert metrics.unmatched_predictions == [("HP:2", "PRESENT")]
    assert metrics.unmatched_gold == [("HP:3", "PRESENT")]


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


def test_document_matching_memoizes_pair_credit(monkeypatch):
    calls = []

    def counting_credit(pred, gold, config=None):
        calls.append((pred, gold))
        return _fake_credit(pred, gold, 0.5, "semantic")

    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_matching.calculate_pair_credit",
        counting_credit,
    )
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:p1", "PRESENT"), ("HP:p2", "PRESENT")],
        gold=[("HP:g1", "PRESENT"), ("HP:g2", "PRESENT")],
    )

    calculate_document_ontology_metrics(result)

    assert sorted(calls) == [
        ("HP:p1", "HP:g1"),
        ("HP:p1", "HP:g2"),
        ("HP:p2", "HP:g1"),
        ("HP:p2", "HP:g2"),
    ]
