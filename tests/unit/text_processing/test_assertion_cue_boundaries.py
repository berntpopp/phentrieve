import pytest

from phentrieve.text_processing.assertion_detection import (
    AssertionStatus,
    CombinedAssertionDetector,
    retrieval_text_for_assertion,
)

pytestmark = pytest.mark.unit


def test_cue_before_hyphen_is_a_compound_not_a_negation() -> None:
    # French cue "non" must not fire on an English compound in auto-language mode
    status, _ = CombinedAssertionDetector(language="fr").detect(
        "non-progressive ataxia"
    )
    assert status == AssertionStatus.AFFIRMED


@pytest.mark.parametrize("text", ["no-reflow phenomenon", "patient with no-reflow"])
def test_parser_ignores_cue_glued_to_hyphen(text) -> None:
    # spaCy parses "no-reflow" as det(no) -> reflow; it is a compound, not negation
    status, _ = CombinedAssertionDetector(language="en").detect(text)
    assert status == AssertionStatus.AFFIRMED


def test_standalone_cue_still_negates() -> None:
    status, _ = CombinedAssertionDetector(language="fr").detect("non convulsions")
    assert status == AssertionStatus.NEGATED


def test_retrieval_text_strips_cue_only_when_negated() -> None:
    details = {"negated_scope_texts": ["seizures"]}
    assert (
        retrieval_text_for_assertion("no seizures", AssertionStatus.NEGATED, details)
        == "seizures"
    )
    assert (
        retrieval_text_for_assertion(" seizures ", AssertionStatus.AFFIRMED, details)
        == "seizures"
    )
    assert retrieval_text_for_assertion("no", AssertionStatus.NEGATED, {}) == "no"
