"""Unit tests for text attribution functionality.

Tests for identifying text spans that correspond to HPO terms:
- get_text_attributions: Finding matches in source text

Following best practices:
- Clear Arrange-Act-Assert structure
- Edge case testing (empty strings, special characters)
- Regex pattern testing (whitespace variations, case insensitivity)
- Overlap detection testing
"""

import pytest

from phentrieve.retrieval.text_attribution import get_text_attributions

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for get_text_attributions()
# =============================================================================


class TestGetTextAttributions:
    """Test get_text_attributions() function."""

    def test_empty_source_text(self):
        """Test that empty source text returns no attributions."""
        # Arrange
        source_text = ""
        hpo_label = "fever"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert result == []

    def test_simple_exact_match(self):
        """Test finding a single exact match."""
        # Arrange
        source_text = "Patient has fever and cough"
        hpo_label = "fever"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert len(result) == 1
        assert result[0]["start_char"] == 12
        assert result[0]["end_char"] == 17
        assert result[0]["matched_text_in_chunk"] == "fever"

    def test_case_insensitive_match(self):
        """Test that matching is case-insensitive."""
        # Arrange
        source_text = "Patient has FEVER and Cough"
        hpo_label = "fever"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert len(result) == 1
        assert result[0]["matched_text_in_chunk"] == "FEVER"

    def test_multiple_matches_same_term(self):
        """Test finding multiple occurrences of the same term."""
        # Arrange
        source_text = "Patient has fever, high fever, persistent fever"
        hpo_label = "fever"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert len(result) == 3
        # Verify all three matches
        matched_texts = [r["matched_text_in_chunk"] for r in result]
        assert matched_texts.count("fever") == 3

    def test_whitespace_variation_matching(self):
        """Test that whitespace variations are matched."""
        # Arrange
        source_text = "Patient has chronic   kidney  disease"
        hpo_label = "chronic kidney disease"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert len(result) == 1
        assert "chronic" in result[0]["matched_text_in_chunk"]
        assert "kidney" in result[0]["matched_text_in_chunk"]
        assert "disease" in result[0]["matched_text_in_chunk"]

    def test_synonym_matching(self):
        """Test matching against synonyms."""
        # Arrange
        source_text = "Patient has pyrexia and elevated temperature"
        hpo_label = "fever"
        synonyms = ["pyrexia", "elevated temperature"]

        # Act
        result = get_text_attributions(source_text, hpo_label, synonyms)

        # Assert
        assert len(result) == 2
        matched_texts = [r["matched_text_in_chunk"] for r in result]
        assert "pyrexia" in matched_texts
        assert "elevated temperature" in matched_texts

    def test_longer_phrases_prioritized(self):
        """Test that longer matching phrases are prioritized."""
        # Arrange
        source_text = "Patient has chronic kidney disease"
        hpo_label = "disease"
        synonyms = ["chronic kidney disease", "kidney disease"]

        # Act
        result = get_text_attributions(source_text, hpo_label, synonyms)

        # Assert
        # Should match the longest phrase "chronic kidney disease"
        # and not also match the shorter overlapping phrases
        assert len(result) == 1
        assert result[0]["matched_text_in_chunk"] == "chronic kidney disease"

    def test_no_overlapping_matches(self):
        """Test that overlapping matches are excluded."""
        # Arrange
        source_text = "Patient has severe headache"
        hpo_label = "headache"
        synonyms = ["severe headache", "headache"]

        # Act
        result = get_text_attributions(source_text, hpo_label, synonyms)

        # Assert
        # Should only match "severe headache" once, not both overlapping matches
        assert len(result) == 1
        assert result[0]["matched_text_in_chunk"] == "severe headache"

    def test_empty_synonyms_list(self):
        """Test with empty synonyms list."""
        # Arrange
        source_text = "Patient has fever"
        hpo_label = "fever"
        synonyms = []

        # Act
        result = get_text_attributions(source_text, hpo_label, synonyms)

        # Assert
        assert len(result) == 1
        assert result[0]["matched_text_in_chunk"] == "fever"

    def test_none_synonyms(self):
        """Test with None synonyms."""
        # Arrange
        source_text = "Patient has fever"
        hpo_label = "fever"

        # Act
        result = get_text_attributions(source_text, hpo_label, None)

        # Assert
        assert len(result) == 1
        assert result[0]["matched_text_in_chunk"] == "fever"

    def test_special_regex_characters_in_term(self):
        """Test handling of special regex characters in HPO terms."""
        # Arrange
        source_text = "Patient has 2/3 syndrome"
        hpo_label = "2/3 syndrome"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert len(result) == 1
        assert result[0]["matched_text_in_chunk"] == "2/3 syndrome"

    def test_term_not_found(self):
        """Test when the term is not present in the text."""
        # Arrange
        source_text = "Patient has cough and cold"
        hpo_label = "fever"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert result == []

    def test_partial_word_not_matched(self):
        """Test that partial word matches are not returned."""
        # Arrange
        source_text = "Patient has preferences for treatment"
        hpo_label = "fever"  # Should not match "fever" in "preferences"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert result == []

    def test_duplicate_synonyms_handled(self):
        """Test that duplicate synonyms are deduplicated."""
        # Arrange
        source_text = "Patient has fever"
        hpo_label = "fever"
        synonyms = ["fever", "pyrexia", "fever"]  # "fever" appears twice

        # Act
        result = get_text_attributions(source_text, hpo_label, synonyms)

        # Assert
        # Should only match "fever" once despite duplicates in search list
        assert len(result) == 1

    def test_empty_phrase_in_synonyms_ignored(self):
        """Test that empty phrases in synonyms are ignored."""
        # Arrange
        source_text = "Patient has fever"
        hpo_label = "fever"
        synonyms = ["", "pyrexia", None]  # Empty and None values

        # Act - should not crash
        result = get_text_attributions(source_text, hpo_label, synonyms)

        # Assert
        assert len(result) == 1  # Only "fever" should match

    def test_with_hpo_id_for_logging(self):
        """Test that providing HPO ID doesn't affect functionality."""
        # Arrange
        source_text = "Patient has fever"
        hpo_label = "fever"
        hpo_id = "HP:0001945"

        # Act
        result = get_text_attributions(source_text, hpo_label, None, hpo_id)

        # Assert
        assert len(result) == 1
        assert result[0]["matched_text_in_chunk"] == "fever"

    def test_multiple_non_overlapping_synonyms(self):
        """Test multiple synonym matches that don't overlap."""
        # Arrange
        source_text = "Patient has pyrexia and elevated temperature with hyperthermia"
        hpo_label = "fever"
        synonyms = ["pyrexia", "elevated temperature", "hyperthermia"]

        # Act
        result = get_text_attributions(source_text, hpo_label, synonyms)

        # Assert
        assert len(result) == 3
        matched_texts = {r["matched_text_in_chunk"] for r in result}
        assert matched_texts == {"pyrexia", "elevated temperature", "hyperthermia"}

    def test_complex_medical_term_with_punctuation(self):
        """Test matching complex medical terms with punctuation."""
        # Arrange
        source_text = (
            "Diagnosis: 3-hydroxy-3-methylglutaryl-CoA lyase deficiency confirmed"
        )
        hpo_label = "3-hydroxy-3-methylglutaryl-CoA lyase deficiency"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert len(result) == 1
        assert (
            "3-hydroxy-3-methylglutaryl-CoA lyase deficiency"
            in result[0]["matched_text_in_chunk"]
        )

    def test_attribution_span_positions_accurate(self):
        """Test that start and end positions are accurate."""
        # Arrange
        source_text = "Patient presents with fever and cough"
        hpo_label = "fever"

        # Act
        result = get_text_attributions(source_text, hpo_label)

        # Assert
        assert len(result) == 1
        start = result[0]["start_char"]
        end = result[0]["end_char"]
        # Extract the substring using the positions
        extracted = source_text[start:end]
        assert extracted == "fever"
