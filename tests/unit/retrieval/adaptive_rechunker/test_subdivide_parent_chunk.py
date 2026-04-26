"""Tests for subdivide_parent_chunk."""

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    subdivide_parent_chunk,
)


@pytest.fixture
def config():
    return AdaptiveRechunkingConfig(
        min_chunk_chars=20, max_sentences_per_subchunk=3, overlap_sentences=1
    )


def make_parent(text: str, start_char: int = 0) -> dict:
    return {
        "text": text,
        "status": "AFFIRMED",
        "assertion_details": {"trigger": "default"},
        "source_indices": {"processing_stages": ["paragraph", "sentence"]},
        "start_char": start_char,
        "end_char": start_char + len(text),
    }


class TestSubdivideParentChunk:
    def test_multi_sentence_parent_produces_subchunks(self, config):
        text = (
            "Patient has severe intellectual disability. "
            "He shows recurrent seizures since age 3. "
            "Brain MRI revealed cortical atrophy. "
            "Family history is unremarkable."
        )
        parent = make_parent(text)
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        assert len(children) >= 1
        # Every child preserves assertion_status.
        for child in children:
            assert child["status"] == "AFFIRMED"
            assert child["assertion_details"] == {"trigger": "default"}

    def test_single_sentence_parent_returns_empty(self, config):
        parent = make_parent("Patient has severe intellectual disability.")
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        # Either 0 (no useful split) or 1 (a single child equal to the
        # parent - filtered out).
        assert children == []

    def test_subchunks_track_depth_in_processing_stages(self, config):
        text = "First sentence here. Second sentence here. Third sentence."
        parent = make_parent(text)
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=2
        )
        for child in children:
            stages = child.get("source_indices", {}).get("processing_stages", [])
            assert any("adaptive_rechunker_depth_2" in s for s in stages)

    def test_subchunks_offset_by_parent_start_char(self, config):
        text = (
            "Patient has severe intellectual disability. "
            "He shows recurrent seizures since age 3. "
            "Brain MRI revealed cortical atrophy. "
            "Family history is unremarkable."
        )
        parent = make_parent(text, start_char=100)
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        assert len(children) >= 1
        for child in children:
            assert child["start_char"] >= 100
            assert child["end_char"] <= 100 + len(text)

    def test_short_subchunks_below_min_chunk_chars_dropped(self):
        config = AdaptiveRechunkingConfig(
            min_chunk_chars=80, max_sentences_per_subchunk=1, overlap_sentences=0
        )
        text = "Short. Tiny. Brief sentence. " + "x" * 100 + "."
        parent = make_parent(text)
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        # Sentences shorter than min_chunk_chars (80) should be dropped.
        for child in children:
            assert len(child["text"]) >= 80

    def test_empty_parent_text_returns_empty(self, config):
        parent = make_parent("")
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        assert children == []
