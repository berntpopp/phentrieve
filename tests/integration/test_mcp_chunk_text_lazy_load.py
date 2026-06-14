"""B2 integration: chunk_text lazy-loads the real embedding model.

Exercises the model-dependent chunking strategies end-to-end against the real
cached embedding singleton (the same instance search/extract warm). Marked slow
because the first call pays the one-time model load; subsequent calls reuse the
process-level cache.
"""

from __future__ import annotations

import pytest

from api.mcp.service_adapters import chunk_text_service

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_CLINICAL_TEXT = (
    "The patient presented with microcephaly and global developmental delay. "
    "There was no seizure activity noted during the prolonged observation period. "
    "Mild hypotonia was also documented on examination."
)


@pytest.mark.parametrize("strategy", ["sliding_window", "detailed"])
def test_chunk_text_model_dependent_strategy_returns_chunks(strategy):
    out = chunk_text_service(text=_CLINICAL_TEXT, language="en", strategy=strategy)
    assert out["chunk_count"] >= 1
    assert out["chunks"][0]["chunk_id"] == 1
    assert out["chunks"][0]["text"]


def test_chunk_text_sliding_window_is_cached_on_second_call():
    first = chunk_text_service(
        text=_CLINICAL_TEXT, language="en", strategy="sliding_window"
    )
    # Second call reuses the cached singleton and must succeed identically.
    second = chunk_text_service(
        text=_CLINICAL_TEXT, language="en", strategy="sliding_window"
    )
    assert first["chunk_count"] == second["chunk_count"]
    assert first["chunk_count"] >= 1
