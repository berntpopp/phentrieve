"""LLM-2 -- orthogonal experiencer/assertion axes + negation-scope qualifier.

The Phase-1 grounded schema gains explicit experiencer/assertion enums and a
negated_qualifier so the model reasons about experiencer and assertion
independently (Gemini structured-output guidance) and captures the negated
portion of an "X without Y" phrase, while the legacy category enum stays
authoritative for backward compatibility.
"""

from __future__ import annotations

from typing import get_args

import pytest

from phentrieve.llm.prompts.loader import get_prompt
from phentrieve.llm.types import (
    LLMExtractedPhenotype,
    LLMGroundedExtractedPhenotype,
)

pytestmark = pytest.mark.unit


def test_grounded_schema_has_orthogonal_axes_with_defaults():
    p = LLMGroundedExtractedPhenotype(
        phrase="seizures", category="Abnormal", chunk_ids=[1]
    )
    assert p.experiencer == "proband"
    assert p.assertion == "present"
    assert p.negated_qualifier is None


def test_grounded_schema_enum_values():
    fields = LLMGroundedExtractedPhenotype.model_fields
    assert set(get_args(fields["experiencer"].annotation)) == {
        "proband",
        "family_history",
        "other",
    }
    assert set(get_args(fields["assertion"].annotation)) == {
        "present",
        "absent",
        "uncertain",
    }


def test_reasoning_fields_declared_before_category_label():
    # Gemini emits keys in schema order; reasoning (experiencer/assertion/
    # negated_qualifier) must precede the category label so it is generated first.
    order = list(LLMGroundedExtractedPhenotype.model_fields)
    assert order.index("experiencer") < order.index("category")
    assert order.index("assertion") < order.index("category")
    assert order.index("negated_qualifier") < order.index("category")


def test_grounded_schema_accepts_legacy_category_only_payload():
    # Backward compatibility: a payload with only the legacy category validates.
    model = LLMGroundedExtractedPhenotype.model_validate(
        {"phrase": "skeletal anomalies", "category": "Normal", "chunk_ids": [1]}
    )
    assert model.category == "Normal"


def test_non_grounded_schema_also_has_orthogonal_axes():
    p = LLMExtractedPhenotype(phrase="seizures", category="Abnormal")
    assert p.experiencer == "proband"
    assert p.assertion == "present"
    assert p.negated_qualifier is None


def test_two_phase_prompt_has_negation_scope_rule_and_bumped_version():
    prompt = get_prompt("two_phase", "en")
    system_prompt = prompt.render_system_prompt()
    lowered = system_prompt.lower()
    # The negation-scope rule must teach that a cue negates only the phrase it
    # directly modifies ("X without Y" -> X present, only Y absent).
    assert "without" in lowered
    assert "negated_qualifier" in lowered
    # Schema/prompt change rolls the prompt version (warm-client signal).
    assert prompt.version != "v3.0.0"
