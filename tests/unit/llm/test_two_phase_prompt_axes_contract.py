"""Guard test: the phase-1 extraction prompt covers all assertion values.

The prompt asks for exhaustive extraction of present, absent and uncertain
findings, and a later mention of an already extracted concept with a
different assertion is its own item. Before, the prompt was exhaustive for
present abnormalities only and forbade any duplicate concept, so negated
findings and later negated mentions were often not extracted.
"""

from __future__ import annotations

from phentrieve.llm.prompts.loader import get_prompt


def _system_prompt() -> str:
    return get_prompt("two_phase", "en").render_system_prompt()


def test_prompt_is_exhaustive_for_absent_and_uncertain_findings() -> None:
    prompt = _system_prompt()

    assert "Be exhaustive for present, absent and uncertain findings" in prompt
    assert "Be exhaustive for present abnormalities" not in prompt


def test_prompt_keeps_a_later_mention_with_another_assertion() -> None:
    prompt = _system_prompt()

    assert "with a different assertion" in prompt
    assert "Do not duplicate the same concept multiple times" not in prompt


def test_prompt_defines_uncertain_with_cues_in_each_language() -> None:
    prompt = _system_prompt()

    assert "uncertain means" in prompt
    for cue in (
        "suspected",
        "differential diagnosis",
        "V. a.",
        "fraglich",
        "suspicion de",
        "sospecha de",
    ):
        assert cue in prompt, cue


def test_prompt_defines_other_as_a_person_who_is_not_a_relative() -> None:
    prompt = _system_prompt()

    assert "another person who is not a relative" in prompt
    assert "other (non-clinical metadata)" not in prompt


def test_schema_describes_other_as_a_person_who_is_not_a_relative() -> None:
    from phentrieve.llm.types import LLMGroundedExtractedPhenotype

    description = LLMGroundedExtractedPhenotype.model_fields["experiencer"].description

    assert "not a relative" in description
    assert "metadata" not in description
