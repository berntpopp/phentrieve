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


def test_letter_case_does_not_prevent_a_match():
    located = locate_evidence(DOCUMENT, "bauchschmerzen", [1], CHUNKS)

    assert (located.start, located.end) == _span("Bauchschmerzen")
    assert located.basis == "evidence_in_chunk"


def test_decomposed_document_text_matches_composed_evidence():
    document = "Es lagen weder Meläna noch Bauchschmerzen vor."
    chunks = {1: (0, len(document) - 1)}

    located = locate_evidence(document, "Meläna", [1], chunks)

    assert document[located.start : located.end] == "Meläna"
    assert located.basis == "evidence_in_chunk"


def test_hull_search_does_not_match_inside_an_uncited_chunk():
    # Chunks 1 and 3 are cited, chunk 2 lies between them but is not cited.
    located = locate_evidence(DOCUMENT, "Biopsieproben zeigten", [1, 3], CHUNKS)

    assert (located.start, located.end) == CHUNKS[1]
    assert located.basis == "chunk_span"


def test_chunks_without_a_located_position_are_ignored():
    chunks = {1: CHUNKS[1], 2: (-1, -1)}

    assert locate_evidence(DOCUMENT, "paraphrase", [2], chunks).basis == "unresolved"
    located = locate_evidence(DOCUMENT, "paraphrase", [1, 2], chunks)
    assert (located.start, located.end) == CHUNKS[1]


def test_paraphrased_evidence_falls_back_to_the_cited_chunks():
    located = locate_evidence(DOCUMENT, "Nekrosen der... Submukosa", [2, 3], CHUNKS)

    assert (located.start, located.end) == (CHUNKS[2][0], CHUNKS[3][1])
    assert located.basis == "chunk_span"


def test_fallback_covers_only_the_first_run_of_adjacent_cited_chunks():
    # Chunks 1 and 3 are not adjacent; a hull over both would also cover
    # chunk 2, which the model did not cite.
    located = locate_evidence(DOCUMENT, "Meläna ... Submukosa", [1, 3], CHUNKS)

    assert (located.start, located.end) == CHUNKS[1]
    assert located.basis == "chunk_span"


def test_fallback_uses_the_chunk_positions_not_the_search_region():
    search_spans = {1: CHUNKS[1], 2: (CHUNKS[1][1], CHUNKS[2][1])}

    located = locate_evidence(
        DOCUMENT, "paraphrase", [2], CHUNKS, search_spans=search_spans
    )

    assert (located.start, located.end) == CHUNKS[2]
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


def test_pipeline_skips_chunks_the_text_pipeline_could_not_locate():
    # TextProcessingPipeline marks unlocated chunks with -1 positions.
    extracted = [
        {"phrase": "x", "evidence_text": "paraphrase", "chunk_ids": [1, 2]},
        {"phrase": "y", "evidence_text": "paraphrase", "chunk_ids": [2]},
    ]
    grounded_chunks = [
        {"chunk_id": 1, "text": "a", "start_char": 0, "end_char": 38},
        {"chunk_id": 2, "text": "b", "start_char": -1, "end_char": -1},
    ]

    pipeline_module.TwoPhaseLLMPipeline._attach_document_offsets(
        extracted, text=DOCUMENT, grounded_chunks=grounded_chunks
    )

    assert (extracted[0]["start_char"], extracted[0]["end_char"]) == (0, 38)
    assert (extracted[1]["start_char"], extracted[1]["end_char"]) == (None, None)
    assert extracted[1]["offset_basis"] == "unresolved"


def test_evidence_may_quote_the_text_cleanup_dropped_before_a_chunk():
    # Chunk cleanup strips "Ein"; the phase-1 index shows it again (prompt_text),
    # so the model quotes it.
    document = "Vorstellung zur Routinekontrolle. Ein Asthma liegt nicht vor."
    first = document.index("Vorstellung zur Routinekontrolle")
    second = document.index("Asthma liegt nicht vor")
    grounded_chunks = [
        {
            "chunk_id": 1,
            "text": "Vorstellung zur Routinekontrolle",
            "start_char": first,
            "end_char": first + 32,
            "prompt_text": None,
        },
        {
            "chunk_id": 2,
            "text": "Asthma liegt nicht vor",
            "start_char": second,
            "end_char": second + 22,
            "prompt_text": "Ein Asthma liegt nicht vor",
        },
    ]
    extracted = [
        {
            "phrase": "Asthma",
            "evidence_text": "Ein Asthma liegt nicht vor",
            "chunk_ids": [2],
        }
    ]

    pipeline_module.TwoPhaseLLMPipeline._attach_document_offsets(
        extracted, text=document, grounded_chunks=grounded_chunks
    )

    item = extracted[0]
    assert (
        document[item["start_char"] : item["end_char"]] == "Ein Asthma liegt nicht vor"
    )
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


def test_phase1_schema_no_longer_asks_the_model_for_offsets():
    from phentrieve.llm.types import LLMGroundedExtractedPhenotype

    fields = LLMGroundedExtractedPhenotype.model_fields

    assert "start_char" not in fields
    assert "end_char" not in fields


def test_evidence_may_quote_the_punctuation_shown_after_a_chunk():
    document = "Kein Hinweis auf Asthma, Hypothyreose oder Kleinwuchs."
    first = document.index("Asthma")
    second = document.index("Hypothyreose")
    grounded_chunks = [
        {
            "chunk_id": 1,
            "text": "Asthma",
            "start_char": first,
            "end_char": first + 6,
            "prompt_text": "Kein Hinweis auf Asthma,",
        },
        {
            "chunk_id": 2,
            "text": "Hypothyreose",
            "start_char": second,
            "end_char": second + 12,
            "prompt_text": None,
        },
    ]
    extracted = [
        {
            "phrase": "Asthma",
            "evidence_text": "Kein Hinweis auf Asthma,",
            "chunk_ids": [1],
        }
    ]

    pipeline_module.TwoPhaseLLMPipeline._attach_document_offsets(
        extracted, text=document, grounded_chunks=grounded_chunks
    )

    item = extracted[0]
    assert document[item["start_char"] : item["end_char"]] == "Kein Hinweis auf Asthma,"
    assert item["offset_basis"] == "evidence_in_chunk"
