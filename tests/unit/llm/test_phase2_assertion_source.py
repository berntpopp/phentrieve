"""Phase-2 assertion source tests (extraction contract v2, block B1).

These tests pin the requirement that ``phenotype_from_candidate`` treats the
model's own RAW WIRE ``assertion`` (present | absent | uncertain) as the source
of truth for the pipeline polarity, mapping it through ``parse_assertion`` to
the pipeline vocabulary (present | negated | uncertain). The model WINS over the
legacy category; the legacy ``category`` becomes a DERIVED-compat field
projected back from the (experiencer, assertion) axes (D3).

The retriever and the LLM provider are never invoked here: these are direct
unit tests of ``phenotype_from_candidate`` / ``derive_category_from_axes``.
"""

from __future__ import annotations

from typing import Any

from phentrieve.llm.pipeline_phase2 import (
    derive_category_from_axes,
    phenotype_from_candidate,
)


def _candidate(**overrides: Any) -> dict[str, Any]:
    candidate = {
        "hpo_id": "HP:0001250",
        "term_name": "Seizure",
        "score": 0.9,
    }
    candidate.update(overrides)
    return candidate


def _item(**overrides: Any) -> dict[str, Any]:
    item = {
        "phrase": "seizures",
        "category": "abnormal",
        "negated_qualifier": None,
        "chunk_ids": [1],
        "evidence_text": "recurrent seizures",
        "start_char": 0,
        "end_char": 10,
    }
    item.update(overrides)
    return item


def test_model_assertion_wins_over_category() -> None:
    """(a) MODEL WINS: category "abnormal" would map to ``present`` via
    ``CATEGORY_TO_ASSERTION``, but the model's ``assertion="absent"`` must win
    and resolve to the pipeline vocabulary ``negated``."""
    item = _item(category="abnormal", assertion="absent")

    phenotype = phenotype_from_candidate(item=item, candidate=_candidate())

    assert phenotype.assertion == "negated"


def test_missing_assertion_falls_back_to_category() -> None:
    """(b) FALLBACK: with no ``assertion`` key the resolved assertion is derived
    from the legacy category (backward compatible). "normal" -> ``negated``."""
    item = _item(category="normal")
    item.pop("assertion", None)
    assert "assertion" not in item

    phenotype = phenotype_from_candidate(item=item, candidate=_candidate())

    assert phenotype.assertion == "negated"


def test_conflicting_axes_store_derived_compat_category() -> None:
    """(c) CONFLICT/COMPAT-CATEGORY: ``assertion="absent"`` with the raw legacy
    ``category="Abnormal"`` -> pipeline ``assertion == "negated"`` AND the STORED
    category is the axis-derived compat value ``"normal"`` (NOT the raw
    "Abnormal"), proving ``derive_category_from_axes`` result is stored."""
    item = _item(category="Abnormal", assertion="absent")

    phenotype = phenotype_from_candidate(item=item, candidate=_candidate())

    assert phenotype.assertion == "negated"
    assert phenotype.category == "normal"


def test_uncertain_model_assertion_maps_to_suspected_category() -> None:
    """A model ``assertion="uncertain"`` for a proband phrase resolves to the
    pipeline ``uncertain`` and a stored compat category of ``"suspected"``."""
    item = _item(category="abnormal", assertion="uncertain")

    phenotype = phenotype_from_candidate(item=item, candidate=_candidate())

    assert phenotype.assertion == "uncertain"
    assert phenotype.category == "suspected"


def test_model_experiencer_wins_over_category() -> None:
    """The model's own ``experiencer`` is preserved and, being family_history,
    the stored compat category collapses to ``"family_history"``."""
    item = _item(
        category="abnormal",
        assertion="present",
        experiencer="family_history",
    )

    phenotype = phenotype_from_candidate(item=item, candidate=_candidate())

    assert phenotype.experiencer == "family_history"
    assert phenotype.category == "family_history"


def test_derive_category_from_axes_projection() -> None:
    """Direct unit coverage of ``derive_category_from_axes`` across all axes."""
    # Experiencer wins for the two non-proband axes.
    assert (
        derive_category_from_axes("family_history", "present", "abnormal")
        == "family_history"
    )
    assert derive_category_from_axes("other", "present", "abnormal") == "other"

    # Proband: pipeline assertion projects back onto the category enum.
    assert derive_category_from_axes("proband", "present", "abnormal") == "abnormal"
    assert derive_category_from_axes("proband", "negated", "abnormal") == "normal"
    assert derive_category_from_axes("proband", "uncertain", "abnormal") == "suspected"

    # Axes absent / unmappable assertion -> fallback category is preserved.
    assert derive_category_from_axes(None, None, "other") == "other"
    assert derive_category_from_axes("proband", None, "abnormal") == "abnormal"
    assert (
        derive_category_from_axes("proband", "family_history", "family_history")
        == "family_history"
    )
