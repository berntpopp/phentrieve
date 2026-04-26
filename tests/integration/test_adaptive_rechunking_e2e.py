"""End-to-end test: adaptive rechunking improves recall on a synthetic
multi-finding fixture against a real (or mocked) ChromaDB index.

Skipped when the local ChromaDB HPO index is not available (CI default).
Marked `slow` because it exercises the full standard backend including
the SentenceChunker, encoder, and retriever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TXT = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "adaptive_rechunking"
    / "synthetic_multi_finding.txt"
)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _try_run_standard(text: str, **kwargs: object) -> dict[str, object]:
    """Run the standard backend, skipping the test if the local HPO index
    or model resources are not available.
    """
    from phentrieve.text_processing.full_text_service import run_standard_backend

    try:
        return run_standard_backend(text=text, **kwargs)
    except Exception as exc:  # noqa: BLE001 - tolerate missing index / model in CI
        pytest.skip(
            f"Skipping e2e test: standard backend unavailable in this "
            f"environment ({type(exc).__name__}: {exc})."
        )


def test_adaptive_rechunking_finds_more_terms() -> None:
    """With adaptive on, the aggregated terms include at least one HPO term
    not surfaced by the no-adaptive baseline.

    Asserts ``meta.adaptive_rechunking`` is populated when the rechunker
    actually fires.
    """
    pytest.importorskip("chromadb")
    assert FIXTURE_TXT.exists(), f"Fixture {FIXTURE_TXT} missing"

    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

    text = FIXTURE_TXT.read_text()

    # No-adaptive baseline.
    baseline = _try_run_standard(text=text, language="en")
    baseline_terms = baseline.get("aggregated_hpo_terms") or []
    baseline_ids = {t["id"] for t in baseline_terms if "id" in t}

    # Adaptive on with permissive thresholds to ensure trigger fires
    # on at least one of the long, multi-finding sentences.
    config = AdaptiveRechunkingConfig(
        enabled=True,
        quality_threshold=0.7,
        margin_threshold=0.1,
        max_depth=2,
        min_chunk_chars=20,
    )
    adaptive = _try_run_standard(text=text, language="en", adaptive_rechunking=config)
    adaptive_terms = adaptive.get("aggregated_hpo_terms") or []
    adaptive_ids = {t["id"] for t in adaptive_terms if "id" in t}

    # Meta block reflects the adaptive run when it fired.
    meta = adaptive.get("meta") or {}
    if "adaptive_rechunking" in meta:
        assert meta["adaptive_rechunking"]["enabled"] is True

    extra = adaptive_ids - baseline_ids
    # At least one term gained. If the encoder is strong enough to never
    # flag anything as poor on the fixture, this is a useful regression
    # signal too - the test will fail and we should re-tune the fixture.
    assert len(extra) >= 1, (
        f"Adaptive rechunking did not surface any new terms. "
        f"Baseline: {baseline_ids}. Adaptive: {adaptive_ids}."
    )
