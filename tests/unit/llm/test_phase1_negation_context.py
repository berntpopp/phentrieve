"""Phase-1 chunk index keeps the context that chunk cleanup strips (#375).

``FinalChunkCleaner`` removes leading words such as ``no``/``not``. Grounded
chunks keep the cleaned ``text`` (retrieval, phase-2 mapping, API output), but
the phase-1 extraction prompt must still show the negation cue.
"""

from __future__ import annotations

from typing import Any

from phentrieve.benchmark.llm_benchmark import _build_grounded_chunks
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.pipeline_phase1 import (
    render_group_chunk_index_text,
    render_phase1_user_prompt,
)
from phentrieve.llm.pipeline_phase2 import build_grounded_context
from phentrieve.llm.preprocessing import (
    _prompt_texts,
    build_extraction_groups,
    build_grounded_chunks_from_text_pipeline,
)
from phentrieve.llm.prompts.loader import get_prompt
from phentrieve.llm.types import AnnotationMode, GroundedChunk

SOURCE = (
    "On examination, there were no plantar or palmar pits, facial "
    "abnormalities or macrocephaly. There was no hepatosplenomegaly."
)
CLEANED = [
    "examination",
    "plantar",
    "palmar pits",
    "facial abnormalities",
    "macrocephaly",
    "hepatosplenomegaly",
]


class _FakeTextProcessingPipeline:
    """Mimics the cleaned chunks and offsets of the default pipeline."""

    def __init__(self, **kwargs: Any) -> None:
        self.assertion_detector = None

    def process(self, raw_text: str, include_positions: bool = False):
        chunks, position = [], 0
        for text in CLEANED:
            start = raw_text.index(text, position)
            position = start + len(text)
            chunks.append(
                {
                    "text": text,
                    "status": "affirmed",
                    "start_char": start,
                    "end_char": position,
                }
            )
        return chunks


def _patch_text_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        "phentrieve.text_processing.pipeline.TextProcessingPipeline",
        _FakeTextProcessingPipeline,
    )
    monkeypatch.setattr(
        "phentrieve.embeddings.load_embedding_model", lambda model_name: object()
    )


def _build(monkeypatch) -> list[GroundedChunk]:
    _patch_text_pipeline(monkeypatch)
    return build_grounded_chunks_from_text_pipeline(
        text=SOURCE,
        language="en",
        chunking_pipeline_config=[],
        assertion_config={"disable": True},
        retrieval_model_name="FremyCompany/BioLORD-2023-M",
    )


def test_grounded_chunks_keep_cleaned_text_and_offsets(monkeypatch) -> None:
    chunks = _build(monkeypatch)

    assert [chunk.text for chunk in chunks] == CLEANED
    assert all(
        SOURCE[chunk.start_char : chunk.end_char] == chunk.text for chunk in chunks
    )


def test_grounded_chunks_restore_stripped_leading_context(monkeypatch) -> None:
    chunks = _build(monkeypatch)

    # Leading words that cleanup dropped, and the punctuation that follows a
    # chunk in the source, so a coordinated list stays recognisable.
    assert [chunk.prompt_text for chunk in chunks] == [
        "On examination,",
        "there were no plantar",
        "or palmar pits,",
        None,
        "or macrocephaly.",
        "There was no hepatosplenomegaly.",
    ]


def test_prompt_text_keeps_the_commas_of_a_negated_list() -> None:
    from phentrieve.llm.preprocessing import _prompt_texts

    text = "Kein Hinweis auf Asthma, Hypothyreose, Katarakt oder Kleinwuchs."
    pieces = ["Asthma", "Hypothyreose", "Katarakt", "oder Kleinwuchs"]
    chunks, position = [], 0
    for piece in pieces:
        start = text.index(piece, position)
        position = start + len(piece)
        chunks.append({"text": piece, "start_char": start, "end_char": position})

    assert _prompt_texts(text, chunks) == [
        "Kein Hinweis auf Asthma,",
        "Hypothyreose,",
        None,
        "oder Kleinwuchs.",
    ]


def test_prompt_text_collapses_line_breaks_in_restored_context() -> None:
    # A line break inside the restored gap would split the chunk-index line, and
    # the grouped renderers drop lines without a ``chunk_id=N:`` prefix.
    text = "He had hypotonia and\nno  seizures were observed."
    chunks = [
        {"text": "He had hypotonia", "start_char": 0, "end_char": 16},
        {"text": "seizures were observed", "start_char": 25, "end_char": 47},
    ]

    prompt_texts = _prompt_texts(text, chunks)

    assert prompt_texts == [None, "and no seizures were observed."]
    rendered = render_group_chunk_index_text(
        group={},
        grounded_chunks=[
            {"chunk_id": index, "text": chunk["text"], "prompt_text": prompt_text}
            for index, (chunk, prompt_text) in enumerate(
                zip(chunks, prompt_texts, strict=True), start=1
            )
        ],
    )
    assert "chunk_id=2: and no seizures were observed." in rendered.splitlines()


def test_prompt_text_skips_chunks_without_a_located_position() -> None:
    # An unlocated chunk (-1 positions) must not end the previous chunk's
    # trailing punctuation search at position -1.
    text = "Fever. Seizures."
    chunks = [
        {"text": "Fever", "start_char": 0, "end_char": 5},
        {"text": "Seizures", "start_char": -1, "end_char": -1},
    ]

    assert _prompt_texts(text, chunks) == ["Fever.", None]


def test_prompt_text_is_none_without_positions(monkeypatch) -> None:
    class _NoPositions(_FakeTextProcessingPipeline):
        def process(self, raw_text: str, include_positions: bool = False):
            return [{"text": "plantar pits", "status": "affirmed"}]

    _patch_text_pipeline(monkeypatch)
    monkeypatch.setattr(
        "phentrieve.text_processing.pipeline.TextProcessingPipeline", _NoPositions
    )

    chunks = build_grounded_chunks_from_text_pipeline(
        text="no plantar pits",
        language="en",
        chunking_pipeline_config=[],
        assertion_config={"disable": True},
        retrieval_model_name="FremyCompany/BioLORD-2023-M",
    )

    assert chunks[0].prompt_text is None


def _dict_chunks() -> list[dict[str, Any]]:
    return [
        {"chunk_id": 1, "text": "examination", "prompt_text": "On examination"},
        {"chunk_id": 2, "text": "plantar", "prompt_text": "there were no plantar"},
        {"chunk_id": 3, "text": "facial abnormalities", "prompt_text": None},
    ]


def test_phase1_user_prompt_renders_prompt_text() -> None:
    prompt = render_phase1_user_prompt(
        extraction_prompt=get_prompt(AnnotationMode.TWO_PHASE, "en"),
        text=SOURCE,
        grounded_chunks=_dict_chunks(),
    )

    assert "- chunk_id=2: there were no plantar\n" in prompt
    assert "- chunk_id=3: facial abnormalities" in prompt


def test_runtime_group_chunk_index_renders_prompt_text() -> None:
    rendered = TwoPhaseLLMPipeline._render_chunk_index(_dict_chunks())

    assert "- chunk_id=2: there were no plantar" in rendered.splitlines()


def test_group_chunk_index_fallback_renders_prompt_text() -> None:
    rendered = render_group_chunk_index_text(group={}, grounded_chunks=_dict_chunks())

    assert "chunk_id=2: there were no plantar" in rendered


class _CountingProvider:
    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        return {"prompt_tokens": 1, "total_tokens": 1}


def test_precomputed_extraction_groups_render_prompt_text() -> None:
    groups = build_extraction_groups(
        grounded_chunks=[
            GroundedChunk(
                chunk_id=1,
                text="plantar",
                start_char=None,
                end_char=None,
                status="affirmed",
                prompt_text="there were no plantar",
            )
        ],
        provider=_CountingProvider(),
        system_prompt="system",
        max_prompt_tokens=100,
    )

    assert groups[0].text == "chunk_id=1: there were no plantar"


def test_rebuilt_large_groups_keep_prompt_text() -> None:
    pipeline = TwoPhaseLLMPipeline(provider=_CountingProvider(), tool_executor=None)

    groups = pipeline._build_large_extraction_groups(
        text=SOURCE,
        grounded_chunks=_dict_chunks(),
        extraction_prompt=get_prompt(AnnotationMode.TWO_PHASE, "en"),
    )

    assert "chunk_id=2: there were no plantar" in groups[0]["text"]


def test_benchmark_grounded_chunk_dicts_carry_prompt_text(monkeypatch) -> None:
    _patch_text_pipeline(monkeypatch)

    chunks = _build_grounded_chunks(
        text=SOURCE,
        language="en",
        chunking_pipeline_config=[],
        assertion_config={"disable": True},
        retrieval_model_name="FremyCompany/BioLORD-2023-M",
    )

    assert chunks[1]["text"] == "plantar"
    assert chunks[1]["prompt_text"] == "there were no plantar"


def test_phase2_grounded_context_keeps_cleaned_chunk_text() -> None:
    context = build_grounded_context(
        item={"phrase": "plantar pits", "chunk_ids": [2]},
        grounded_chunks=_dict_chunks(),
    )

    assert context["primary_chunk_text"] == "plantar"
    assert context["neighbor_chunk_texts"] == ["examination", "facial abnormalities"]
