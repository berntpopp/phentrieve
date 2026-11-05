from typing import Any

import pytest

from phentrieve.text_processing.assertion_detection import (
    AssertionStatus,
    CombinedAssertionDetector,
)


class FakeAssertionDetector:
    def __init__(self, status: AssertionStatus, source: str) -> None:
        self.status = status
        self.source = source

    def detect(self, text_chunk: str) -> tuple[AssertionStatus, dict[str, Any]]:
        return self.status, {f"{self.source}_detail": text_chunk}


def make_detector(
    *,
    preference: str,
    keyword_status: AssertionStatus,
    dependency_status: AssertionStatus,
) -> CombinedAssertionDetector:
    detector = CombinedAssertionDetector(
        language="en",
        enable_keyword=False,
        enable_dependency=False,
        preference=preference,
    )
    detector.enable_keyword = True
    detector.enable_dependency = True
    detector.keyword_detector = FakeAssertionDetector(keyword_status, "keyword")
    detector.dependency_detector = FakeAssertionDetector(
        dependency_status, "dependency"
    )
    return detector


def test_keyword_preference_uses_keyword_result_when_detectors_disagree() -> None:
    detector = make_detector(
        preference="keyword",
        keyword_status=AssertionStatus.AFFIRMED,
        dependency_status=AssertionStatus.NEGATED,
    )

    status, details = detector.detect("fever")

    assert status is AssertionStatus.AFFIRMED
    assert details["final_status"] == "affirmed"
    assert details["keyword_status"] == "affirmed"
    assert details["dependency_status"] == "negated"


def test_dependency_preference_uses_dependency_result_when_detectors_disagree() -> None:
    detector = make_detector(
        preference="dependency",
        keyword_status=AssertionStatus.NEGATED,
        dependency_status=AssertionStatus.AFFIRMED,
    )

    status, details = detector.detect("fever")

    assert status is AssertionStatus.AFFIRMED
    assert details["final_status"] == "affirmed"
    assert details["keyword_status"] == "negated"
    assert details["dependency_status"] == "affirmed"


def test_any_negative_preference_keeps_uncertain_result_when_detectors_disagree() -> (
    None
):
    detector = make_detector(
        preference="any_negative",
        keyword_status=AssertionStatus.UNCERTAIN,
        dependency_status=AssertionStatus.AFFIRMED,
    )

    status, details = detector.detect("possible fever")

    assert status is AssertionStatus.UNCERTAIN
    assert details["final_status"] == "uncertain"
    assert details["keyword_status"] == "uncertain"
    assert details["dependency_status"] == "affirmed"


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "seizures were ruled out"),
        ("en", "Epilepsy has been ruled out"),
        ("en", "negative for seizures"),
        ("en", "absence of seizures"),
        ("en", "free of seizures"),
        ("de", "Krampfanfälle wurden ausgeschlossen"),
    ],
)
def test_default_detector_negates_multi_word_cues(language: str, text: str) -> None:
    """Multi-word cues must survive the dependency preference (#459)."""
    detector = CombinedAssertionDetector(language=language, preference="dependency")

    assert detector.detect(text)[0] is AssertionStatus.NEGATED
    assert detector.detect_batch([text])[0][0] is AssertionStatus.NEGATED


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "seizures cannot be ruled out"),
        ("en", "seizures were not ruled out"),
        ("en", "cannot rule out seizures"),
        ("en", "a seizure disorder could not be excluded"),
        ("en", "not only seizures"),
        ("de", "Krampfanfälle wurden nicht ausgeschlossen"),
        ("de", "Krampfanfälle sind nicht auszuschließen"),
        ("es", "no se puede descartar epilepsia"),
        ("fr", "une épilepsie ne peut être exclue"),
        ("nl", "epilepsie is niet uit te sluiten"),
    ],
)
def test_default_detector_keeps_open_findings_unnegated(
    language: str, text: str
) -> None:
    """A finding that was not excluded must never read as absent (#459)."""
    detector = CombinedAssertionDetector(language=language, preference="dependency")

    assert detector.detect(text)[0] is not AssertionStatus.NEGATED
