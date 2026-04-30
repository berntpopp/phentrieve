from typing import Any

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
