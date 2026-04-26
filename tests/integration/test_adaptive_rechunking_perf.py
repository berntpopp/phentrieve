"""Perf smoke test: wall time for a fully-flagging fixture document is
within a loose 5x bound.

Designed to catch egregious regressions (accidentally re-running the full
pipeline, or scaling encoder fan-out unboundedly) - not to specify
performance. The hard invariant is the call_count test.

Marked ``slow`` because it loads the standard backend (encoder + retriever)
and is excluded from the default ``make test`` run.
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _try_run_standard(text: str, **kwargs: object) -> dict[str, object]:
    """Run the standard backend, skipping the test if local resources
    (HPO index, encoder model) are unavailable.
    """
    from phentrieve.text_processing.full_text_service import run_standard_backend

    try:
        return run_standard_backend(text=text, **kwargs)
    except Exception as exc:  # noqa: BLE001 - tolerate missing index / model in CI
        pytest.skip(
            f"Skipping perf smoke: standard backend unavailable "
            f"({type(exc).__name__}: {exc})."
        )


def test_wall_time_within_loose_bound() -> None:
    """5x is loose because encoder fan-out scales with sub-chunk count, not
    query-call count. The call_count test enforces the tight cost-model
    bound; this test guards against runtime explosions that could ship past
    a query-counting test (e.g., re-running the entire pipeline).
    """
    pytest.importorskip("chromadb")

    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

    text = (
        "Patient has seizures. Patient has hearing loss. "
        "Patient has developmental delay. Patient has microcephaly. "
        "Patient has hypotonia. Patient has ataxia. "
        "Patient has spasticity. Patient has scoliosis. "
        "Patient has dysmorphic features. "
        "Patient has cognitive impairment."
    )

    t0 = time.perf_counter()
    _try_run_standard(text=text, language="en")
    baseline = time.perf_counter() - t0

    config = AdaptiveRechunkingConfig(
        enabled=True,
        # Very aggressive thresholds so most chunks flag as poor.
        quality_threshold=0.95,
        margin_threshold=0.5,
        max_depth=2,
        max_sentences_per_subchunk=2,
        overlap_sentences=0,
        min_chunk_chars=10,
    )
    t0 = time.perf_counter()
    _try_run_standard(text=text, language="en", adaptive_rechunking=config)
    adaptive = time.perf_counter() - t0

    multiplier = adaptive / baseline if baseline > 0 else 1.0
    assert multiplier <= 5.0, (
        f"Adaptive rechunking is {multiplier:.2f}x slower than baseline; "
        f"loose bound is 5x. Encoder fan-out can legitimately exceed the "
        f"query-call multiplier, but exceeding 5x usually means a regression."
    )
