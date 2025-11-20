"""Unit tests for assertion detection (pytest style)."""

import pytest

from phentrieve.text_processing.assertion_detection import (
    AssertionStatus,
    CombinedAssertionDetector,
    ConTextRule,
    DependencyAssertionDetector,
    Direction,
    KeywordAssertionDetector,
    TriggerCategory,
    parse_context_rules,
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


class TestConTextRule:
    """Test cases for ConTextRule dataclass."""

    def test_valid_rule_creation(self):
        """Test creating a valid ConText rule."""
        rule = ConTextRule(
            literal="no",
            category=TriggerCategory.NEGATED_EXISTENCE,
            direction=Direction.FORWARD,
            metadata={"source": "test"},
        )

        assert rule.literal == "no"
        assert rule.category == TriggerCategory.NEGATED_EXISTENCE
        assert rule.direction == Direction.FORWARD
        assert rule.metadata == {"source": "test"}

    def test_rule_immutability(self):
        """Test that ConTextRule is immutable (frozen)."""
        rule = ConTextRule(
            literal="no",
            category=TriggerCategory.NEGATED_EXISTENCE,
            direction=Direction.FORWARD,
        )

        with pytest.raises(AttributeError):
            rule.literal = "yes"  # type: ignore

    def test_empty_literal_raises_error(self):
        """Test that empty literal raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ConTextRule(
                literal="",
                category=TriggerCategory.NEGATED_EXISTENCE,
                direction=Direction.FORWARD,
            )

        with pytest.raises(ValueError, match="cannot be empty"):
            ConTextRule(
                literal="   ",  # Whitespace only
                category=TriggerCategory.NEGATED_EXISTENCE,
                direction=Direction.FORWARD,
            )


class TestParseContextRules:
    """Test cases for parse_context_rules() function."""

    def test_parse_valid_json(self):
        """Test parsing valid ConText JSON."""
        json_data = {
            "context_rules": [
                {
                    "literal": "no",
                    "category": "NEGATED_EXISTENCE",
                    "direction": "FORWARD",
                    "metadata": {"source": "medspaCy"},
                },
                {
                    "literal": "ausgeschlossen",
                    "category": "NEGATED_EXISTENCE",
                    "direction": "BACKWARD",
                },
            ]
        }

        rules = parse_context_rules(json_data)

        assert len(rules) == 2
        assert rules[0].literal == "no"
        assert rules[0].category == TriggerCategory.NEGATED_EXISTENCE
        assert rules[0].direction == Direction.FORWARD
        assert rules[1].literal == "ausgeschlossen"
        assert rules[1].direction == Direction.BACKWARD

    def test_parse_missing_context_rules_key(self):
        """Test error when 'context_rules' key is missing."""
        json_data = {"rules": []}  # Wrong key name

        with pytest.raises(ValueError, match="must contain 'context_rules'"):
            parse_context_rules(json_data)

    def test_parse_missing_required_field(self):
        """Test error when required field is missing."""
        json_data = {
            "context_rules": [
                {
                    "literal": "no",
                    # Missing category and direction
                }
            ]
        }

        with pytest.raises(KeyError, match="Missing required field"):
            parse_context_rules(json_data)

    def test_parse_invalid_category(self):
        """Test error when category is invalid."""
        json_data = {
            "context_rules": [
                {
                    "literal": "no",
                    "category": "INVALID_CATEGORY",
                    "direction": "FORWARD",
                }
            ]
        }

        with pytest.raises(ValueError, match="Invalid category"):
            parse_context_rules(json_data)

    def test_parse_invalid_direction(self):
        """Test error when direction is invalid."""
        json_data = {
            "context_rules": [
                {
                    "literal": "no",
                    "category": "NEGATED_EXISTENCE",
                    "direction": "INVALID_DIRECTION",
                }
            ]
        }

        with pytest.raises(ValueError, match="Invalid direction"):
            parse_context_rules(json_data)

    def test_parse_all_categories(self):
        """Test parsing rules with all category types."""
        json_data = {
            "context_rules": [
                {
                    "literal": "no",
                    "category": "NEGATED_EXISTENCE",
                    "direction": "FORWARD",
                },
                {
                    "literal": "possible",
                    "category": "POSSIBLE_EXISTENCE",
                    "direction": "FORWARD",
                },
                {
                    "literal": "history of",
                    "category": "HISTORICAL",
                    "direction": "BACKWARD",
                },
                {
                    "literal": "if",
                    "category": "HYPOTHETICAL",
                    "direction": "FORWARD",
                },
                {
                    "literal": "mother has",
                    "category": "FAMILY",
                    "direction": "BACKWARD",
                },
                {
                    "literal": "but",
                    "category": "TERMINATE",
                    "direction": "TERMINATE",
                },
                {
                    "literal": "not only",
                    "category": "PSEUDO",
                    "direction": "PSEUDO",
                },
            ]
        }

        rules = parse_context_rules(json_data)
        assert len(rules) == 7

    def test_parse_all_directions(self):
        """Test parsing rules with all direction types."""
        json_data = {
            "context_rules": [
                {
                    "literal": "no",
                    "category": "NEGATED_EXISTENCE",
                    "direction": "FORWARD",
                },
                {
                    "literal": "is absent",
                    "category": "NEGATED_EXISTENCE",
                    "direction": "BACKWARD",
                },
                {
                    "literal": "neither",
                    "category": "NEGATED_EXISTENCE",
                    "direction": "BIDIRECTIONAL",
                },
                {
                    "literal": "but",
                    "category": "TERMINATE",
                    "direction": "TERMINATE",
                },
                {
                    "literal": "not only",
                    "category": "PSEUDO",
                    "direction": "PSEUDO",
                },
            ]
        }

        rules = parse_context_rules(json_data)
        assert len(rules) == 5
        assert rules[0].direction == Direction.FORWARD
        assert rules[1].direction == Direction.BACKWARD
        assert rules[2].direction == Direction.BIDIRECTIONAL
        assert rules[3].direction == Direction.TERMINATE
        assert rules[4].direction == Direction.PSEUDO
