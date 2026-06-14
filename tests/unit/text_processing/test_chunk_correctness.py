"""D2/D3 -- chunk correctness: split multi-concept spans so co-occurring
findings each surface (D2), and drop degenerate function-word-only chunks (D3).
"""

import pytest

from phentrieve.text_processing.chunkers import ConjunctionChunker, FinalChunkCleaner

pytestmark = pytest.mark.unit


def test_conjunction_chunker_splits_progression_marker():
    chunker = ConjunctionChunker(language="en")
    out = [
        c.lower()
        for c in chunker.chunk(["initial hypotonia progressing to hypertonia"])
    ]
    assert any("hypotonia" in c and "hypertonia" not in c for c in out)
    assert any("hypertonia" in c and "hypotonia" not in c for c in out)


def test_conjunction_chunker_still_splits_plain_and():
    chunker = ConjunctionChunker(language="en")
    out = [c.lower() for c in chunker.chunk(["seizures and microcephaly"])]
    assert any("seizures" in c and "microcephaly" not in c for c in out)
    assert any("microcephaly" in c for c in out)


def test_degenerate_function_word_chunk_dropped():
    cleaner = FinalChunkCleaner(language="en")
    out = [c.strip().lower() for c in cleaner.chunk(["due", "seizures", "walk"])]
    assert "due" not in out
    assert "seizures" in out  # real phenotype kept
