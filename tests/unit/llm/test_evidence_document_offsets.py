"""Evidence offsets are document offsets computed in code, not by the model.

The phase-1 schema asks the model for ``start_char``/``end_char``, but the
model counts them itself and relative to a chunk, so they do not point into
the document. The pipeline locates ``evidence_text`` inside the cited chunks
instead and records how the span was found.
"""

from __future__ import annotations

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.evidence_offsets import locate_evidence
from phentrieve.llm.pipeline_phase2 import phenotype_from_candidate
from phentrieve.llm.types import LLMPhenotype

DOCUMENT = (
    "Es lagen weder Meläna noch Bauchschmerzen vor. Biopsieproben zeigten "
    "nekrotisches Material sowie Nekrosen der Schleimhaut\nund Submukosa."
)


def _span(text: str) -> tuple[int, int]:
    start = DOCUMENT.index(text)
    return start, start + len(text)


CHUNKS = {
    1: _span("Es lagen weder Meläna noch Bauchschmerzen vor"),
    2: _span(
        "Biopsieproben zeigten nekrotisches Material sowie Nekrosen der Schleimhaut"
    ),
    3: _span("und Submukosa"),
}


def test_evidence_inside_a_cited_chunk_gets_its_document_span():
    located = locate_evidence(DOCUMENT, "Bauchschmerzen", [1], CHUNKS)

    assert (located.start, located.end) == _span("Bauchschmerzen")
    assert located.basis == "evidence_in_chunk"


def test_evidence_across_cited_chunks_is_found_in_their_hull():
    located = locate_evidence(
        DOCUMENT, "Nekrosen der Schleimhaut und Submukosa", [2, 3], CHUNKS
    )

    assert DOCUMENT[located.start : located.end] == (
        "Nekrosen der Schleimhaut\nund Submukosa"
    )
    assert located.basis == "evidence_in_chunk_hull"


def test_typography_and_whitespace_do_not_prevent_a_match():
    located = locate_evidence(DOCUMENT, "weder  Meläna", [1], CHUNKS)

    assert DOCUMENT[located.start : located.end] == "weder Meläna"


def test_paraphrased_evidence_falls_back_to_the_cited_chunks():
    located = locate_evidence(DOCUMENT, "Nekrosen der... Submukosa", [2, 3], CHUNKS)

    assert (located.start, located.end) == (CHUNKS[2][0], CHUNKS[3][1])
    assert located.basis == "chunk_span"


def test_evidence_outside_the_cited_chunks_is_not_taken():
    located = locate_evidence(DOCUMENT, "Bauchschmerzen", [2], CHUNKS)

    assert (located.start, located.end) == CHUNKS[2]
    assert located.basis == "chunk_span"


def test_without_chunk_positions_the_span_stays_unknown():
    located = locate_evidence(DOCUMENT, "Bauchschmerzen", [9], CHUNKS)

    assert (located.start, located.end) == (None, None)
    assert located.basis == "unresolved"


def test_pipeline_replaces_model_counted_offsets_with_document_offsets():
    extracted = [
        {
            "phrase": "Bauchschmerzen",
            "evidence_text": "Bauchschmerzen",
            "chunk_ids": [1],
            "start_char": 27,  # model-counted, relative to the chunk
            "end_char": 41,
        }
    ]
    grounded_chunks = [
        {"chunk_id": cid, "text": DOCUMENT[s:e], "start_char": s, "end_char": e}
        for cid, (s, e) in CHUNKS.items()
    ]

    pipeline_module.TwoPhaseLLMPipeline._attach_document_offsets(
        extracted, text=DOCUMENT, grounded_chunks=grounded_chunks
    )

    item = extracted[0]
    assert DOCUMENT[item["start_char"] : item["end_char"]] == "Bauchschmerzen"
    assert item["offset_basis"] == "evidence_in_chunk"


def test_evidence_record_carries_the_offset_basis():
    item = {
        "phrase": "Bauchschmerzen",
        "category": "normal",
        "evidence_text": "Bauchschmerzen",
        "chunk_ids": [1],
        "start_char": 27,
        "end_char": 41,
        "offset_basis": "evidence_in_chunk",
        "experiencer": "proband",
        "assertion": "absent",
    }
    candidate = {"hpo_id": "HP:0002027", "term_name": "Abdominal pain", "score": 0.9}

    term = phenotype_from_candidate(item=item, candidate=candidate)

    record = term.evidence_records[0]
    assert (record.start_char, record.end_char) == (27, 41)
    assert record.offset_basis == "evidence_in_chunk"


def test_terms_have_a_temporality_defaulting_to_unknown():
    term = LLMPhenotype(term_id="HP:0002027", label="Abdominal pain")

    assert term.temporality == "unknown"
