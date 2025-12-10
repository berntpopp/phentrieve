"""Unit tests for TextSpan and find_span_in_text."""

import pytest

from phentrieve.text_processing.spans import TextSpan, find_span_in_text


class TestTextSpan:
    """Tests for TextSpan dataclass."""

    def test_creation(self) -> None:
        """Test basic TextSpan creation."""
        span = TextSpan("hello", 0, 5)
        assert span.text == "hello"
        assert span.start_char == 0
        assert span.end_char == 5

    def test_str_returns_text(self) -> None:
        """Test that str() returns the text content."""
        span = TextSpan("hello", 10, 15)
        assert str(span) == "hello"

    def test_len_returns_text_length(self) -> None:
        """Test that len() returns length of text content."""
        span = TextSpan("hello", 10, 15)
        assert len(span) == 5

    def test_immutable(self) -> None:
        """Test that TextSpan is immutable (frozen)."""
        span = TextSpan("hello", 0, 5)
        with pytest.raises(AttributeError):
            span.text = "world"  # type: ignore[misc]

    def test_validation_negative_start(self) -> None:
        """Test that negative start_char raises ValueError."""
        with pytest.raises(ValueError, match="start_char must be >= 0"):
            TextSpan("hello", -1, 5)

    def test_validation_end_before_start(self) -> None:
        """Test that end_char < start_char raises ValueError."""
        with pytest.raises(ValueError, match="end_char .* must be >= start_char"):
            TextSpan("hello", 10, 5)

    def test_zero_length_span_allowed(self) -> None:
        """Test that zero-length spans are valid (start == end)."""
        span = TextSpan("", 5, 5)
        assert span.start_char == span.end_char
        assert len(span) == 0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        span = TextSpan("hello", 10, 15)
        assert span.to_dict() == {"text": "hello", "start_char": 10, "end_char": 15}

    def test_equality(self) -> None:
        """Test equality comparison (frozen dataclass)."""
        span1 = TextSpan("hello", 0, 5)
        span2 = TextSpan("hello", 0, 5)
        span3 = TextSpan("world", 0, 5)
        assert span1 == span2
        assert span1 != span3

    def test_hashable(self) -> None:
        """Test that TextSpan can be used in sets/dicts (frozen)."""
        span1 = TextSpan("hello", 0, 5)
        span2 = TextSpan("hello", 0, 5)
        span_set = {span1, span2}
        assert len(span_set) == 1


class TestFindSpanInText:
    """Tests for find_span_in_text function."""

    def test_exact_match(self) -> None:
        """Test finding exact match in text."""
        span = find_span_in_text("world", "hello world")
        assert span is not None
        assert span.start_char == 6
        assert span.end_char == 11
        assert span.text == "world"

    def test_exact_match_at_start(self) -> None:
        """Test finding match at document start."""
        span = find_span_in_text("hello", "hello world")
        assert span is not None
        assert span.start_char == 0
        assert span.end_char == 5

    def test_not_found(self) -> None:
        """Test that None is returned when text not found."""
        assert find_span_in_text("xyz", "hello world") is None

    def test_empty_needle(self) -> None:
        """Test empty needle returns None."""
        assert find_span_in_text("", "hello") is None

    def test_empty_haystack(self) -> None:
        """Test empty haystack returns None."""
        assert find_span_in_text("hello", "") is None

    def test_both_empty(self) -> None:
        """Test both empty returns None."""
        assert find_span_in_text("", "") is None

    def test_search_start(self) -> None:
        """Test search_start parameter for handling duplicates."""
        text = "hello hello"
        span1 = find_span_in_text("hello", text, search_start=0)
        span2 = find_span_in_text("hello", text, search_start=1)
        assert span1 is not None and span1.start_char == 0
        assert span2 is not None and span2.start_char == 6

    def test_search_start_past_match(self) -> None:
        """Test search_start past all matches returns None."""
        assert find_span_in_text("hello", "hello world", search_start=10) is None

    def test_whitespace_fallback(self) -> None:
        """Test whitespace-normalized fallback matching."""
        # Multiple spaces in haystack, single in needle
        span = find_span_in_text("hello world", "hello  world")
        assert span is not None

    def test_whitespace_fallback_tabs(self) -> None:
        """Test whitespace fallback handles tabs."""
        span = find_span_in_text("hello world", "hello\tworld")
        assert span is not None

    def test_whitespace_fallback_newlines(self) -> None:
        """Test whitespace fallback handles newlines."""
        span = find_span_in_text("hello world", "hello\nworld")
        assert span is not None

    def test_unicode_german(self) -> None:
        """Test finding German text with umlauts."""
        span = find_span_in_text("Trinkschwäche", "Kind mit Trinkschwäche")
        assert span is not None
        assert span.start_char == 9
        assert span.text == "Trinkschwäche"

    def test_unicode_french(self) -> None:
        """Test finding French text with accents."""
        span = find_span_in_text("été", "L'été est chaud")
        assert span is not None
        assert span.start_char == 2

    def test_unicode_spanish(self) -> None:
        """Test finding Spanish text with special characters."""
        span = find_span_in_text("niño", "El niño tiene fiebre")
        assert span is not None
        assert span.start_char == 3

    def test_case_sensitive(self) -> None:
        """Test that search is case-sensitive."""
        assert find_span_in_text("Hello", "hello world") is None
        assert find_span_in_text("hello", "Hello world") is None

    def test_multiline_text(self) -> None:
        """Test finding text across lines."""
        text = "First line.\nSecond line.\nThird line."
        span = find_span_in_text("Second line", text)
        assert span is not None
        assert span.start_char == 12

    def test_returned_text_is_from_haystack(self) -> None:
        """Test that returned span.text comes from haystack."""
        haystack = "hello world"
        span = find_span_in_text("world", haystack)
        assert span is not None
        # Verify the returned text is the exact slice from haystack
        assert span.text == haystack[span.start_char : span.end_char]
