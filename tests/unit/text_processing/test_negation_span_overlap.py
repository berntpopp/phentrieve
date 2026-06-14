"""D1 -- span-level negation: a match is negated only when its matched-phrase
span overlaps a computed negated scope; co-located affirmed concepts in the
same chunk ("X without Y") must not be over-negated. The C1 cases (no X / not X
/ denies X) must stay negated.
"""

import pytest

from phentrieve.text_processing.assertion_detection import KeywordAssertionDetector
from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)


def test_detector_surfaces_negated_scope_text_for_without():
    det = KeywordAssertionDetector(language="en")
    _status, details = det.detect("severe intellectual disability without regression")
    texts = details.get("negated_scope_texts")
    assert texts is not None, "detector must surface negated_scope_texts"
    joined = " ".join(texts).lower()
    assert "regression" in joined
    assert "intellectual disability" not in joined


def _precomputed(hpo_id: str, label: str):
    return [
        {
            "metadatas": [[{"id": hpo_id, "label": label}]],
            "similarities": [[0.95]],
        }
    ]


def test_without_does_not_negate_the_head_concept():
    chunk = "severe intellectual disability without regression"
    result = orchestrate_hpo_extraction(
        text_chunks=[chunk],
        retriever=None,
        num_results_per_chunk=3,
        chunk_retrieval_threshold=0.0,
        assertion_statuses=["negated"],  # chunk-level says negated (the bug input)
        chunk_negated_scope_texts=[["regression"]],
        precomputed_query_results=_precomputed(
            "HP:0010864", "Severe intellectual disability"
        ),
    )
    term = next(t for t in result.aggregated_results if t["id"] == "HP:0010864")
    assert term["assertion_status"] == "affirmed"


def test_negated_concept_inside_scope_stays_negated():
    chunk = "no regression of milestones"
    result = orchestrate_hpo_extraction(
        text_chunks=[chunk],
        retriever=None,
        num_results_per_chunk=3,
        chunk_retrieval_threshold=0.0,
        assertion_statuses=["negated"],
        chunk_negated_scope_texts=[["regression of milestones"]],
        precomputed_query_results=_precomputed("HP:0001531", "Failure to thrive"),
    )
    # The single match's phrase ("regression ...") overlaps the negated scope.
    term = next(t for t in result.aggregated_results if t["id"] == "HP:0001531")
    # FTT label will not attribute into this chunk -> no attribution -> conservative
    # fallback keeps the chunk-level negated status (C1-safe).
    assert term["assertion_status"] == "negated"


def test_no_scope_texts_preserves_chunk_level_status():
    chunk = "the patient had seizures"
    result = orchestrate_hpo_extraction(
        text_chunks=[chunk],
        retriever=None,
        num_results_per_chunk=3,
        chunk_retrieval_threshold=0.0,
        assertion_statuses=["affirmed"],
        chunk_negated_scope_texts=None,  # legacy callers pass nothing
        precomputed_query_results=_precomputed("HP:0001250", "Seizure"),
    )
    term = next(t for t in result.aggregated_results if t["id"] == "HP:0001250")
    assert term["assertion_status"] == "affirmed"


@pytest.mark.parametrize(
    "text,kw",
    [
        ("There is no nystagmus.", "nystagmus"),
        ("She does not have ataxia.", "ataxia"),
        ("The patient denies headache.", "headache"),
    ],
)
def test_c1_negation_still_detected_end_to_end(text, kw):
    from phentrieve.text_processing.pipeline import TextProcessingPipeline

    pipe = TextProcessingPipeline(
        language="en",
        chunking_pipeline_config=[{"type": "paragraph"}, {"type": "sentence"}],
        assertion_config={
            "enable_keyword": True,
            "enable_dependency": True,
            "preference": "dependency",
        },
    )
    chunks = pipe.process(text, include_positions=True)
    neg = [c for c in chunks if kw in c["text"].lower()]
    assert neg, f"no chunk contained {kw}"
    assert all(c.get("status").value == "negated" for c in neg)
