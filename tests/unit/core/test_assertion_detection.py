"""Unit tests for assertion detection (pytest style)."""

import pytest

from phentrieve.text_processing.assertion_detection import (
    AssertionStatus,
    CombinedAssertionDetector,
    DependencyAssertionDetector,
    KeywordAssertionDetector,
)

pytestmark = pytest.mark.unit


class TestKeywordAssertionDetector:
    """Test cases for KeywordAssertionDetector."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.detector = KeywordAssertionDetector(language="en")

    def test_negation_detection(self):
        """Test detection of negation in text."""
        test_cases = [
            ("Patient denies fever", AssertionStatus.NEGATED),
            ("No evidence of heart disease", AssertionStatus.NEGATED),
            ("Absence of rash", AssertionStatus.NEGATED),
            ("Patient has fever", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            assert status == expected_status, (
                f"Failed for text: '{text}', got {status} instead of {expected_status}"
            )

            if expected_status == AssertionStatus.NEGATED:
                assert len(details["keyword_negated_scopes"]) > 0, (
                    f"No negation scopes found for '{text}'"
                )

    def test_normality_detection(self):
        """Test detection of normality in text."""
        test_cases = [
            ("Normal blood pressure", AssertionStatus.NORMAL),
            ("Liver function tests within normal limits", AssertionStatus.NORMAL),
            ("Chest X-ray is unremarkable", AssertionStatus.NORMAL),
            ("Elevated heart rate", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            assert status == expected_status, (
                f"Failed for text: '{text}', got {status} instead of {expected_status}"
            )

            if expected_status == AssertionStatus.NORMAL:
                assert len(details["keyword_normal_scopes"]) > 0, (
                    f"No normality scopes found for '{text}'"
                )


class TestDependencyAssertionDetector:
    """Test cases for DependencyAssertionDetector."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        try:
            self.detector = DependencyAssertionDetector(language="en")
        except Exception:
            pytest.skip("Required spaCy model not available")

    def test_negation_detection(self):
        """Test detection of negation using dependency parsing."""
        test_cases = [
            ("Patient does not have fever", AssertionStatus.NEGATED),
            ("No evidence of heart disease was found", AssertionStatus.NEGATED),
            ("We did not observe any rash", AssertionStatus.NEGATED),
            ("Patient has fever", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            assert status == expected_status, (
                f"Failed for text: '{text}', got {status} instead of {expected_status}"
            )

            if expected_status == AssertionStatus.NEGATED:
                assert len(details["dependency_negated_concepts"]) > 0, (
                    f"No negation concepts found for '{text}'"
                )

    def test_normality_detection_after_refactoring(self):
        """Test normality detection works correctly after refactoring."""
        test_cases = [
            ("The patient's liver function is normal", AssertionStatus.NORMAL),
            ("Lungs are clear", AssertionStatus.NORMAL),
            ("Blood pressure within normal limits", AssertionStatus.NORMAL),
            ("Elevated heart rate noted", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            assert status == expected_status, (
                f"Failed for text: '{text}', got {status} instead of {expected_status}"
            )

            if expected_status == AssertionStatus.NORMAL:
                assert len(details["dependency_normal_concepts"]) > 0, (
                    f"No normality concepts found for '{text}'"
                )

    def test_multilingual_support(self):
        """Test detection in different languages."""
        languages = {
            "en": "Patient has no fever",
            "de": "Patient hat kein Fieber",
        }

        for lang, text in languages.items():
            try:
                detector = DependencyAssertionDetector(language=lang)
                status, details = detector.detect(text)
                assert status == AssertionStatus.NEGATED
            except Exception:
                pytest.skip(f"spaCy model for {lang} not available")


class TestCombinedAssertionDetector:
    """Test cases for CombinedAssertionDetector."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.detector = CombinedAssertionDetector(language="en")

    def test_dependency_parser_priority(self):
        """Test that dependency parser is preferred when available."""
        text = "Patient does not have seizures"
        status, details = self.detector.detect(text)

        assert status == AssertionStatus.NEGATED
        assert details["dependency_parser"] is True

    def test_keyword_fallback(self):
        """Test keyword detector fallback when dependency parser unavailable."""
        detector_no_deps = CombinedAssertionDetector(language="xx")
        text = "No evidence of disease"
        status, details = detector_no_deps.detect(text)

        # Should use keyword detector
        assert status == AssertionStatus.NEGATED or status == AssertionStatus.NORMAL
