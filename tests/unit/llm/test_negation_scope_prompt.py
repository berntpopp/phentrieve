"""WS6 contract: the two-phase extraction prompt carries explicit negation-scope
guidance and a contrastive 'X without Y' few-shot so the LLM stops over-negating
the head finding (evaluation defect L7). This is a static contract check; the
behavioral fix is validated by the mapping benchmark (gated, run separately).
"""

import json

import pytest

from phentrieve.llm.prompts import loader
from phentrieve.llm.types import AnnotationMode

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_cache():
    loader.load_prompt_template.cache_clear()
    yield
    loader.load_prompt_template.cache_clear()


def _prompt():
    return loader.get_prompt(AnnotationMode.TWO_PHASE, "en")


def test_prompt_version_bumped():
    assert _prompt().version == "v3.1.0"


def test_system_prompt_has_negation_scope_rule():
    sp = _prompt().system_prompt.lower()
    assert "negation scope" in sp
    assert "without" in sp  # the "X without Y" guidance
    assert "non-progressive" in sp  # modifier-level negation guidance


def test_contrastive_without_few_shot_present():
    examples = _prompt().few_shot_examples
    blob = json.dumps(examples)
    assert "without regression" in blob
    # the head finding stays Abnormal; only the modifier is Normal
    has_id_abnormal = any(
        "severe intellectual disability" in str(ex.get("output", ""))
        and '"category": "Abnormal"' in str(ex.get("output", ""))
        for ex in examples
    )
    assert has_id_abnormal
