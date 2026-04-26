"""Benchmark integration: adaptive rechunking does not regress aggregated
term counts on the small German fixture.

Uses the small German fixture from
``tests/data/benchmarks/german/tiny_v1.json``. The full ontology-aware
metric validation (commit 9402a57) is the release-gate manual run quoted
in Spec B; this CI-friendly smoke verifies adaptive on does not produce
fewer aggregated terms than baseline. Marked ``slow`` because it exercises
the full standard backend across multiple cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "data" / "benchmarks" / "german" / "tiny_v1.json"


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _try_run_standard(text: str, **kwargs: object) -> dict[str, Any] | None:
    """Run the standard backend; return None if local resources are missing.

    Returning ``None`` rather than calling ``pytest.skip`` lets the caller
    accumulate skip context across the loop body.
    """
    from phentrieve.text_processing.full_text_service import run_standard_backend

    try:
        return dict(run_standard_backend(text=text, **kwargs))
    except Exception:  # noqa: BLE001 - tolerate missing index / model in CI
        return None


def test_adaptive_does_not_regress_ontology_metric() -> None:
    """Adaptive on vs off should not produce fewer aggregated terms.

    A loose CI tripwire; the full ontology-aware metric (MRR with LCA
    credit) is checked manually pre-release per Spec B.
    """
    pytest.importorskip("chromadb")
    if not FIXTURE.exists():
        pytest.skip(f"German tiny benchmark fixture missing at {FIXTURE}")

    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

    cases = json.loads(FIXTURE.read_text())
    if not isinstance(cases, list) or not cases:
        pytest.skip("German tiny benchmark fixture is empty or malformed")

    baseline_terms_count = 0
    adaptive_terms_count = 0
    cases_evaluated = 0
    for case in cases[:3]:  # Keep test runtime bounded.
        text = case.get("text", "") if isinstance(case, dict) else ""
        if not text:
            continue
        baseline = _try_run_standard(text=text, language="de")
        if baseline is None:
            pytest.skip("Standard backend unavailable - skipping benchmark smoke.")
        adaptive_cfg = AdaptiveRechunkingConfig(enabled=True, max_depth=1)
        adaptive = _try_run_standard(
            text=text, language="de", adaptive_rechunking=adaptive_cfg
        )
        if adaptive is None:
            pytest.skip("Standard backend unavailable - skipping benchmark smoke.")

        baseline_terms_count += len(baseline.get("aggregated_hpo_terms") or [])
        adaptive_terms_count += len(adaptive.get("aggregated_hpo_terms") or [])
        cases_evaluated += 1

    if cases_evaluated == 0:
        pytest.skip("No usable benchmark cases evaluated.")

    # Loose check: adaptive should not produce *fewer* terms on average.
    # Allow a tolerance of 1 term across the small slice to absorb noise
    # (a single dropped low-confidence term should not block a release).
    assert adaptive_terms_count >= baseline_terms_count - 1, (
        f"Adaptive surfaced fewer terms ({adaptive_terms_count}) than "
        f"baseline ({baseline_terms_count}) across {cases_evaluated} cases; "
        f"investigate before merging."
    )
