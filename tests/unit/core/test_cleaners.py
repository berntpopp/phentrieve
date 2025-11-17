"""Unit tests for text cleaning utilities.

Tests for text normalization and cleaning functions:
- normalize_line_endings: Line ending normalization (Windows/Mac/Unix)
- clean_internal_newlines_and_extra_spaces: Whitespace and newline cleanup

Following best practices:
- Clear Arrange-Act-Assert structure
- Edge case testing (empty strings, None-like inputs)
- Comprehensive character testing (various line endings, spaces)
"""

import pytest

from phentrieve.text_processing.cleaners import (
    clean_internal_newlines_and_extra_spaces,
    normalize_line_endings,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for normalize_line_endings()
# =============================================================================


class TestNormalizeLineEndings:
    """Test normalize_line_endings() function."""

    def test_normalizes_windows_line_endings(self):
        """Test normalization of Windows-style CRLF line endings."""
        # Arrange
        text = "line1\r\nline2\r\nline3"

        # Act
        result = normalize_line_endings(text)

        # Assert
        assert result == "line1\nline2\nline3"
        assert "\r\n" not in result
        assert "\r" not in result

    def test_normalizes_old_mac_line_endings(self):
        """Test normalization of old Mac-style CR line endings."""
        # Arrange
        text = "line1\rline2\rline3"

        # Act
        result = normalize_line_endings(text)

        # Assert
        assert result == "line1\nline2\nline3"
        assert "\r" not in result

    def test_preserves_unix_line_endings(self):
        """Test that Unix-style LF line endings are preserved."""
        # Arrange
        text = "line1\nline2\nline3"

        # Act
        result = normalize_line_endings(text)

        # Assert
        assert result == "line1\nline2\nline3"

    def test_handles_mixed_line_endings(self):
        """Test normalization of mixed line ending styles."""
        # Arrange
        text = "line1\r\nline2\nline3\rline4"

        # Act
        result = normalize_line_endings(text)

        # Assert
        assert result == "line1\nline2\nline3\nline4"
        assert "\r\n" not in result
        assert "\r" not in result

    def test_handles_empty_string(self):
        """Test that empty string returns empty string."""
        # Arrange
        text = ""

        # Act
        result = normalize_line_endings(text)

        # Assert
        assert result == ""

    def test_handles_text_without_line_endings(self):
        """Test that text without line endings is unchanged."""
        # Arrange
        text = "single line of text"

        # Act
        result = normalize_line_endings(text)

        # Assert
        assert result == "single line of text"

    def test_handles_multiple_consecutive_line_endings(self):
        """Test normalization with multiple consecutive line endings."""
        # Arrange
        text = "line1\r\n\r\nline2\n\nline3"

        # Act
        result = normalize_line_endings(text)

        # Assert
        assert result == "line1\n\nline2\n\nline3"
        assert "\r" not in result


# =============================================================================
# Tests for clean_internal_newlines_and_extra_spaces()
# =============================================================================


class TestCleanInternalNewlinesAndExtraSpaces:
    """Test clean_internal_newlines_and_extra_spaces() function."""

    def test_replaces_newlines_with_spaces(self):
        """Test that internal newlines are replaced with single spaces."""
        # Arrange
        text = "line1\nline2\nline3"

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == "line1 line2 line3"
        assert "\n" not in result

    def test_normalizes_multiple_spaces(self):
        """Test that multiple spaces are reduced to single space."""
        # Arrange
        text = "word1  word2   word3    word4"

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == "word1 word2 word3 word4"
        assert "  " not in result

    def test_handles_newlines_with_surrounding_whitespace(self):
        """Test that newlines with surrounding spaces are normalized."""
        # Arrange
        text = "line1  \n  line2   \n   line3"

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == "line1 line2 line3"
        assert "\n" not in result
        assert "  " not in result

    def test_strips_leading_and_trailing_whitespace(self):
        """Test that leading and trailing whitespace is removed."""
        # Arrange
        text = "   content with spaces   "

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == "content with spaces"
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_handles_empty_string(self):
        """Test that empty string returns empty string."""
        # Arrange
        text = ""

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == ""

    def test_handles_whitespace_only_string(self):
        """Test that whitespace-only string returns empty string."""
        # Arrange
        text = "   \n  \n   "

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == ""

    def test_handles_single_word(self):
        """Test that single word without whitespace is unchanged."""
        # Arrange
        text = "word"

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == "word"

    def test_handles_complex_mixed_whitespace(self):
        """Test handling of complex mixed whitespace patterns."""
        # Arrange
        text = "  line1   \n\n  line2  \n   line3   "

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == "line1 line2 line3"
        assert "\n" not in result
        assert "  " not in result
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_preserves_single_spaces_between_words(self):
        """Test that single spaces between words are preserved."""
        # Arrange
        text = "word1 word2 word3"

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        assert result == "word1 word2 word3"

    def test_handles_tabs_and_other_whitespace(self):
        """Test handling of tabs and other whitespace characters."""
        # Arrange
        text = "word1\t\tword2\n\nword3"

        # Act
        result = clean_internal_newlines_and_extra_spaces(text)

        # Assert
        # Tabs and newlines should be normalized to single spaces
        assert "\t" not in result
        assert "\n" not in result
        assert "word1" in result and "word2" in result and "word3" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestCleanerIntegration:
    """Test integration of cleaning functions."""

    def test_normalize_then_clean_pipeline(self):
        """Test typical pipeline: normalize line endings then clean."""
        # Arrange
        text = "line1\r\nline2  \r\n  line3"

        # Act - typical pipeline
        normalized = normalize_line_endings(text)
        cleaned = clean_internal_newlines_and_extra_spaces(normalized)

        # Assert
        assert cleaned == "line1 line2 line3"

    def test_handles_clinical_text_example(self):
        """Test with realistic clinical text example."""
        # Arrange
        clinical_text = "Patient presents with:\r\n  fever  \n  cough\n\n  fatigue   "

        # Act
        normalized = normalize_line_endings(clinical_text)
        cleaned = clean_internal_newlines_and_extra_spaces(normalized)

        # Assert
        assert cleaned == "Patient presents with: fever cough fatigue"
        assert "\r" not in cleaned
        assert "\n" not in cleaned
        assert "  " not in cleaned
