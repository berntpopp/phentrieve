"""H2 regression: negated findings must be emitted (not dropped) so that
``excluded: true`` phenopacket features can be built deterministically.

Root-cause note: the evaluation's "denies headache yields no Headache term"
symptom had two causes. (1) Before the C1 fix, prepositional-negation chunks
("no X", "does not have X") were mislabeled AFFIRMED, so no negated findings
existed at all. C1 fixes the polarity. (2) The aggregation layer already
preserves negated terms (see also TestAssertionStatuses in
test_hpo_extraction_orchestrator_char). These tests lock that contract so a
future change cannot silently drop negated findings from the aggregated list.
"""

from unittest.mock import MagicMock

import pytest

from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)

pytestmark = pytest.mark.unit


def _make_mock_retriever(batch_results):
    retriever = MagicMock()
    retriever.detect_index_type.return_value = "single_vector"
    retriever.query_batch.return_value = batch_results
    return retriever


def _chroma_batch_entry(items):
    metadatas = [[{"hpo_id": hpo_id, "label": label} for hpo_id, label, _ in items]]
    similarities = [[sim for _, _, sim in items]]
    return {"metadatas": metadatas, "similarities": similarities}


def test_negated_finding_is_emitted_as_excluded_capable():
    """A negated chunk with a retrieved match yields an aggregated term whose
    assertion_status is 'negated' -- the input an exporter needs for excluded=true."""
    retriever = _make_mock_retriever(
        [_chroma_batch_entry([("HP:0002315", "Headache", 0.98)])]
    )
    aggregated, chunk_results = orchestrate_hpo_extraction(
        text_chunks=["headache"],
        retriever=retriever,
        chunk_retrieval_threshold=0.5,
        assertion_statuses=["negated"],
    )
    # per-chunk matches carry the negated status
    assert chunk_results[0]["matches"][0]["assertion_status"] == "negated"
    # the aggregated term survives and is flagged negated (not dropped)
    headache = [t for t in aggregated if t["id"] == "HP:0002315"]
    assert headache, "negated finding was dropped from aggregated_hpo_terms"
    assert headache[0]["assertion_status"] == "negated"
    assert headache[0]["status"] == "negated"


def test_mixed_polarity_findings_both_emitted():
    """Affirmed and negated findings from different chunks both survive aggregation."""
    retriever = _make_mock_retriever(
        [
            _chroma_batch_entry([("HP:0001250", "Seizure", 0.95)]),
            _chroma_batch_entry([("HP:0002315", "Headache", 0.95)]),
        ]
    )
    aggregated, _ = orchestrate_hpo_extraction(
        text_chunks=["seizures", "headache"],
        retriever=retriever,
        chunk_retrieval_threshold=0.5,
        assertion_statuses=["affirmed", "negated"],
    )
    by_id = {t["id"]: t["status"] for t in aggregated}
    assert by_id.get("HP:0001250") == "affirmed"
    assert by_id.get("HP:0002315") == "negated"
