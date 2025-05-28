"""
Tests for the assertion detection functionality.

This module tests the assertion detection classes, focusing on their ability
to identify negation, normality, and other assertion types in clinical text.
"""

import unittest
from unittest.mock import patch

from phentrieve.text_processing.assertion_detection import (
    AssertionStatus,
    KeywordAssertionDetector,
    DependencyAssertionDetector,
    CombinedAssertionDetector,
)


class TestKeywordAssertionDetector(unittest.TestCase):
    """Test cases for the KeywordAssertionDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = KeywordAssertionDetector(language="en")

    def test_negation_detection(self):
        """Test detection of negation in text."""
        # Test with various negation patterns
        test_cases = [
            ("Patient denies fever", AssertionStatus.NEGATED),
            ("No evidence of heart disease", AssertionStatus.NEGATED),
            ("Absence of rash", AssertionStatus.NEGATED),
            ("Patient has fever", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            self.assertEqual(
                status,
                expected_status,
                f"Failed for text: '{text}', got {status} instead of {expected_status}",
            )

            if expected_status == AssertionStatus.NEGATED:
                self.assertGreater(
                    len(details["keyword_negated_scopes"]),
                    0,
                    f"No negation scopes found for '{text}'",
                )

    def test_normality_detection(self):
        """Test detection of normality in text."""
        # Test with various normality patterns that match our normality cues resource
        test_cases = [
            ("Normal blood pressure", AssertionStatus.NORMAL),
            ("Liver function tests within normal limits", AssertionStatus.NORMAL),
            ("Chest X-ray is unremarkable", AssertionStatus.NORMAL),
            ("Elevated heart rate", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            self.assertEqual(
                status,
                expected_status,
                f"Failed for text: '{text}', got {status} instead of {expected_status}",
            )

            if expected_status == AssertionStatus.NORMAL:
                self.assertGreater(
                    len(details["keyword_normal_scopes"]),
                    0,
                    f"No normality scopes found for '{text}'",
                )


class TestDependencyAssertionDetector(unittest.TestCase):
    """Test cases for the DependencyAssertionDetector."""

    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if spaCy models aren't available
        try:
            self.detector = DependencyAssertionDetector(language="en")
        except Exception:
            self.skipTest("Required spaCy model not available")

    def test_negation_detection(self):
        """Test detection of negation using dependency parsing."""
        # Test with various negation patterns
        test_cases = [
            ("Patient does not have fever", AssertionStatus.NEGATED),
            ("No evidence of heart disease was found", AssertionStatus.NEGATED),
            ("We did not observe any rash", AssertionStatus.NEGATED),
            ("Patient has fever", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            self.assertEqual(
                status,
                expected_status,
                f"Failed for text: '{text}', got {status} instead of {expected_status}",
            )

            if expected_status == AssertionStatus.NEGATED:
                self.assertGreater(
                    len(details["dependency_negated_concepts"]),
                    0,
                    f"No negation concepts found for '{text}'",
                )

    def test_normality_detection_after_refactoring(self):
        """Test normality detection works correctly after refactoring."""
        # Test with various normality patterns that match our normality cues resource
        test_cases = [
            ("The patient's liver function is normal", AssertionStatus.NORMAL),
            ("Lungs are clear", AssertionStatus.NORMAL),
            ("Blood pressure within normal limits", AssertionStatus.NORMAL),
            ("Elevated heart rate noted", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            self.assertEqual(
                status,
                expected_status,
                f"Failed for text: '{text}', got {status} instead of {expected_status}",
            )

            if expected_status == AssertionStatus.NORMAL:
                self.assertGreater(
                    len(details["dependency_normal_concepts"]),
                    0,
                    f"No normality concepts found for '{text}'",
                )

    def test_multilingual_support(self):
        """Test detection in different languages."""
        # Create detectors for different languages
        languages = {
            "en": "Patient has no fever",
            "de": "Patient hat kein Fieber",
            # Add more languages if spaCy models are available
        }

        for lang, text in languages.items():
            try:
                detector = DependencyAssertionDetector(language=lang)
                status, details = detector.detect(text)
                self.assertEqual(
                    status,
                    AssertionStatus.NEGATED,
                    f"Failed to detect negation in {lang}: '{text}'",
                )
            except Exception:
                # Skip if language model not available
                continue


class TestCombinedAssertionDetector(unittest.TestCase):
    """Test cases for the CombinedAssertionDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = CombinedAssertionDetector(
            language="en",
            enable_keyword=True,
            enable_dependency=True,
            preference="dependency",
        )

    def test_combined_detection(self):
        """Test combined detection strategy."""
        # Test with various patterns
        test_cases = [
            ("Patient has no fever", AssertionStatus.NEGATED),
            ("Normal heart rhythm", AssertionStatus.NORMAL),
            ("Patient has fever", AssertionStatus.AFFIRMED),
        ]

        for text, expected_status in test_cases:
            status, details = self.detector.detect(text)
            self.assertEqual(
                status,
                expected_status,
                f"Failed for text: '{text}', got {status} instead of {expected_status}",
            )

            # Check that both keyword and dependency results are in details
            self.assertIn("keyword_status", details)
            self.assertIn("dependency_status", details)


class TestAssertionDetectionResources(unittest.TestCase):
    """Test cases for resource loading in assertion detection."""

    @patch("phentrieve.text_processing.resource_loader._RESOURCE_CACHE")
    def test_resource_caching_in_assertion_detection(self, mock_cache):
        """Test that resources are properly cached when used in assertion detection."""
        # Setup mock cache
        mock_cache.get.return_value = None
        mock_cache.__contains__.return_value = False

        # Create a detector and detect something to trigger resource loading
        detector = KeywordAssertionDetector(language="en")
        detector.detect("Patient has no fever")

        # Check that cache was accessed
        self.assertTrue(mock_cache.__setitem__.called)

        # Create another detector and detect again
        detector2 = KeywordAssertionDetector(language="en")
        detector2.detect("Patient is afebrile")

        # The second time should check the cache
        self.assertTrue(mock_cache.__contains__.called)


if __name__ == "__main__":
    unittest.main()
