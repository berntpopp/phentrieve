from __future__ import annotations

from collections.abc import Sequence

from phentrieve.llm.preprocessing import (
    build_extraction_groups,
    build_grounded_chunks_from_text_pipeline,
)
from phentrieve.llm.types import ExtractionGroup, GroundedChunk


class FakeTokenCountingProvider:
    def __init__(self, token_value_per_chunk: int = 10) -> None:
        self.token_value_per_chunk = token_value_per_chunk
        self.calls: list[tuple[str, str]] = []

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        self.calls.append((system_prompt, user_prompt))
        chunk_count = user_prompt.count("chunk_id=")
        prompt_tokens = chunk_count * self.token_value_per_chunk + 1
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }


def _chunk(
    chunk_id: int,
    text: str,
    *,
    start_char: int,
    end_char: int,
) -> GroundedChunk:
    return GroundedChunk(
        chunk_id=chunk_id,
        text=text,
        start_char=start_char,
        end_char=end_char,
        status="grounded",
    )


def _flatten(grouped: Sequence[ExtractionGroup]) -> list[int]:
    return [chunk_id for group in grouped for chunk_id in group.chunk_ids]


def _assert_preserves_original_chunk_ids(
    *,
    grounded_chunks: Sequence[GroundedChunk],
    groups: Sequence[ExtractionGroup],
) -> None:
    original_chunk_ids = [chunk.chunk_id for chunk in grounded_chunks]
    flattened_chunk_ids = _flatten(groups)
    first_occurrence_ids = list(dict.fromkeys(flattened_chunk_ids))

    assert first_occurrence_ids == original_chunk_ids
    assert flattened_chunk_ids[0] == original_chunk_ids[0]
    assert flattened_chunk_ids[-1] == original_chunk_ids[-1]
    assert set(flattened_chunk_ids) == set(original_chunk_ids)
    assert all(chunk_id in original_chunk_ids for chunk_id in flattened_chunk_ids)
    for chunk_id in original_chunk_ids:
        positions = [
            index
            for index, value in enumerate(flattened_chunk_ids)
            if value == chunk_id
        ]
        assert positions == list(range(positions[0], positions[-1] + 1))


def test_build_extraction_groups_preserves_chunk_ids_and_positions() -> None:
    provider = FakeTokenCountingProvider()
    grounded_chunks = [
        _chunk(1, "Alpha beta.", start_char=0, end_char=11),
        _chunk(2, "Gamma delta!", start_char=12, end_char=24),
        _chunk(3, "Epsilon zeta?", start_char=25, end_char=38),
        _chunk(4, "Eta theta.", start_char=39, end_char=49),
    ]

    groups = build_extraction_groups(
        grounded_chunks=grounded_chunks,
        provider=provider,
        system_prompt="system prompt",
        max_prompt_tokens=25,
        neighbor_overlap=1,
    )

    _assert_preserves_original_chunk_ids(
        grounded_chunks=grounded_chunks,
        groups=groups,
    )
    chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in grounded_chunks}

    assert [group.group_id for group in groups] == list(range(1, len(groups) + 1))
    assert all(
        group.chunk_ids == list(range(group.chunk_ids[0], group.chunk_ids[-1] + 1))
        for group in groups
    )
    assert all(
        group.text
        == "\n".join(
            f"chunk_id={chunk_id}: {chunk_text_by_id[chunk_id]}"
            for chunk_id in group.chunk_ids
        )
        for group in groups
    )
    assert all(group.estimated_prompt_tokens <= 25 for group in groups)
    assert len(provider.calls) >= len(groups)


def test_build_extraction_groups_respects_token_budget() -> None:
    provider = FakeTokenCountingProvider()
    grounded_chunks = [
        _chunk(1, "First chunk.", start_char=0, end_char=12),
        _chunk(2, "Second chunk.", start_char=13, end_char=26),
        _chunk(3, "Third chunk.", start_char=27, end_char=39),
        _chunk(4, "Fourth chunk.", start_char=40, end_char=53),
        _chunk(5, "Fifth chunk.", start_char=54, end_char=66),
    ]

    groups = build_extraction_groups(
        grounded_chunks=grounded_chunks,
        provider=provider,
        system_prompt="system prompt",
        max_prompt_tokens=21,
        neighbor_overlap=1,
    )

    _assert_preserves_original_chunk_ids(
        grounded_chunks=grounded_chunks,
        groups=groups,
    )
    assert [group.group_id for group in groups] == list(range(1, len(groups) + 1))
    assert all(group.estimated_prompt_tokens <= 21 for group in groups)
    assert all(
        group.chunk_ids == list(range(group.chunk_ids[0], group.chunk_ids[-1] + 1))
        for group in groups
    )
    assert _flatten(groups)[0] == 1
    assert _flatten(groups)[-1] == 5
    assert any(len(group.chunk_ids) > 1 for group in groups)


def test_build_extraction_groups_keeps_adjacent_context_overlap_small() -> None:
    provider = FakeTokenCountingProvider()
    grounded_chunks = [
        _chunk(1, "Chunk one.", start_char=0, end_char=10),
        _chunk(2, "Chunk two.", start_char=11, end_char=21),
        _chunk(3, "Chunk three.", start_char=22, end_char=34),
        _chunk(4, "Chunk four.", start_char=35, end_char=46),
        _chunk(5, "Chunk five.", start_char=47, end_char=58),
    ]

    groups = build_extraction_groups(
        grounded_chunks=grounded_chunks,
        provider=provider,
        system_prompt="system prompt",
        max_prompt_tokens=21,
        neighbor_overlap=1,
    )

    _assert_preserves_original_chunk_ids(
        grounded_chunks=grounded_chunks,
        groups=groups,
    )
    chunk_to_group_ids: dict[int, list[int]] = {}
    for index, group in enumerate(groups):
        for chunk_id in group.chunk_ids:
            chunk_to_group_ids.setdefault(chunk_id, []).append(index)

    assert list(chunk_to_group_ids) == [1, 2, 3, 4, 5]
    assert all(
        max(group_ids) - min(group_ids) <= 1
        for group_ids in chunk_to_group_ids.values()
    )
    assert all(
        group.chunk_ids == list(range(group.chunk_ids[0], group.chunk_ids[-1] + 1))
        for group in groups
    )


def test_build_extraction_groups_rejects_oversized_single_chunk() -> None:
    provider = FakeTokenCountingProvider(token_value_per_chunk=50)
    grounded_chunks = [_chunk(1, "Oversized chunk.", start_char=0, end_char=16)]

    try:
        build_extraction_groups(
            grounded_chunks=grounded_chunks,
            provider=provider,
            system_prompt="system prompt",
            max_prompt_tokens=25,
            neighbor_overlap=1,
        )
    except ValueError as exc:
        assert "Single grounded chunk exceeds max_prompt_tokens" in str(exc)
    else:
        raise AssertionError("Expected oversized single chunk to be rejected")


def test_build_extraction_groups_does_not_emit_overlap_only_tail_group() -> None:
    provider = FakeTokenCountingProvider()
    grounded_chunks = [
        _chunk(1, "First chunk.", start_char=0, end_char=12),
        _chunk(2, "Second chunk.", start_char=13, end_char=26),
        _chunk(3, "Third chunk.", start_char=27, end_char=39),
    ]

    groups = build_extraction_groups(
        grounded_chunks=grounded_chunks,
        provider=provider,
        system_prompt="system prompt",
        max_prompt_tokens=100,
        neighbor_overlap=1,
    )

    assert [group.chunk_ids for group in groups] == [[1, 2, 3]]


def test_build_grounded_chunks_uses_left_context_for_negated_chunk_status(
    monkeypatch,
) -> None:
    class FakeStatus:
        def __init__(self, value: str) -> None:
            self.value = value

    class FakeDetector:
        def detect(self, text: str):
            if "no big head" in text:
                return FakeStatus("negated"), {"source": "context"}
            return FakeStatus("affirmed"), {"source": "chunk"}

    class FakeTextProcessingPipeline:
        def __init__(self, **kwargs) -> None:
            self.assertion_detector = FakeDetector()

        def process(self, raw_text: str, include_positions: bool = False):
            return [
                {
                    "text": "big head",
                    "status": FakeStatus("affirmed"),
                    "assertion_details": {"source": "chunk"},
                    "start_char": raw_text.index("big head"),
                    "end_char": raw_text.index("big head") + len("big head"),
                }
            ]

    monkeypatch.setattr(
        "phentrieve.text_processing.pipeline.TextProcessingPipeline",
        FakeTextProcessingPipeline,
    )
    monkeypatch.setattr(
        "phentrieve.embeddings.load_embedding_model",
        lambda model_name: object(),
    )

    chunks = build_grounded_chunks_from_text_pipeline(
        text="Bernt Popp is small and dumb and has blonde hair but no big head",
        language="en",
        chunking_pipeline_config=[],
        assertion_config={"disable": False, "preference": "dependency"},
        retrieval_model_name="FremyCompany/BioLORD-2023-M",
    )

    assert chunks[0].text == "big head"
    assert chunks[0].status == "negated"
