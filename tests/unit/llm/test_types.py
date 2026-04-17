from __future__ import annotations

from phentrieve.llm.types import AnnotationMode, AssertionStatus, PostProcessingStep


def test_assertion_status_keeps_present_alias() -> None:
    assert AssertionStatus.PRESENT.value == "present"
    assert AssertionStatus.AFFIRMED is AssertionStatus.PRESENT


def test_post_processing_step_includes_expected_values() -> None:
    assert PostProcessingStep.VALIDATION.value == "validation"
    assert PostProcessingStep.COMBINED.value == "combined"


def test_annotation_mode_does_not_expose_retrieval_only() -> None:
    assert "retrieval_only" not in {mode.value for mode in AnnotationMode}
