"""
Unit tests for HPO parser safe dictionary access helpers.

Tests the defensive programming helpers introduced to resolve issue #23:
- safe_get_nested(): Nested dictionary access without KeyError
- safe_get_list(): Type-safe list field access
- log_missing_field(): Structured logging for missing fields
- log_parsing_summary(): Statistics summary logging

These helpers prevent crashes when the HPO Consortium modifies the JSON schema.
"""

import logging

from phentrieve.data_processing.hpo_parser import (
    log_missing_field,
    log_parsing_summary,
    safe_get_list,
    safe_get_nested,
)


class TestSafeGetNested:
    """Tests for safe_get_nested() helper function."""

    def test_successful_nested_access(self):
        """Test successful access to deeply nested value."""
        data = {"a": {"b": {"c": "target_value"}}}
        result = safe_get_nested(data, "a", "b", "c")
        assert result == "target_value"

    def test_successful_shallow_access(self):
        """Test access to top-level key."""
        data = {"key": "value"}
        result = safe_get_nested(data, "key")
        assert result == "value"

    def test_missing_leaf_key_returns_default(self):
        """Test that missing final key returns default."""
        data = {"a": {"b": {}}}
        result = safe_get_nested(data, "a", "b", "c", default="fallback")
        assert result == "fallback"

    def test_missing_intermediate_key_returns_default(self):
        """Test that missing intermediate key returns default."""
        data = {"a": {}}
        result = safe_get_nested(data, "a", "b", "c", default="fallback")
        assert result == "fallback"

    def test_missing_root_key_returns_default(self):
        """Test that missing root key returns default."""
        data: dict = {}
        result = safe_get_nested(data, "a", "b", "c", default="fallback")
        assert result == "fallback"

    def test_none_value_returns_default(self):
        """Test that explicit None value returns default."""
        data = {"a": {"b": {"c": None}}}
        result = safe_get_nested(data, "a", "b", "c", default="fallback")
        assert result == "fallback"

    def test_default_none_without_explicit_default(self):
        """Test that missing key returns None if no default specified."""
        data: dict = {}
        result = safe_get_nested(data, "missing")
        assert result is None

    def test_non_dict_intermediate_returns_default(self):
        """Test that non-dict intermediate value returns default."""
        data = {"a": "not_a_dict"}
        result = safe_get_nested(data, "a", "b", "c", default="fallback")
        assert result == "fallback"

    def test_empty_string_value_preserved(self):
        """Test that empty string is returned (not replaced with default)."""
        data = {"a": {"b": {"c": ""}}}
        result = safe_get_nested(data, "a", "b", "c", default="fallback")
        assert result == ""

    def test_zero_value_preserved(self):
        """Test that zero is returned (not replaced with default)."""
        data = {"a": {"b": {"c": 0}}}
        result = safe_get_nested(data, "a", "b", "c", default=999)
        assert result == 0

    def test_false_value_preserved(self):
        """Test that False is returned (not replaced with default)."""
        data = {"a": {"b": {"c": False}}}
        result = safe_get_nested(data, "a", "b", "c", default=True)
        assert result is False

    def test_empty_list_value_preserved(self):
        """Test that empty list is returned (not replaced with default)."""
        data = {"a": {"b": {"c": []}}}
        result = safe_get_nested(data, "a", "b", "c", default=["fallback"])
        assert result == []

    def test_complex_nested_structure(self):
        """Test with complex realistic HPO-like structure."""
        data = {
            "meta": {
                "definition": {"val": "A seizure is...", "xrefs": ["PMID:123"]},
                "synonyms": [{"val": "Epilepsy"}],
            }
        }
        result = safe_get_nested(data, "meta", "definition", "val", default="")
        assert result == "A seizure is..."


class TestSafeGetList:
    """Tests for safe_get_list() helper function."""

    def test_successful_list_access(self):
        """Test successful access to list field."""
        data = {"meta": {"synonyms": [{"val": "syn1"}, {"val": "syn2"}]}}
        result = safe_get_list(data, "meta", "synonyms")
        assert result == [{"val": "syn1"}, {"val": "syn2"}]

    def test_missing_field_returns_empty_list_default(self):
        """Test that missing field returns empty list by default."""
        data: dict = {"meta": {}}
        result = safe_get_list(data, "meta", "synonyms")
        assert result == []

    def test_missing_field_returns_custom_default(self):
        """Test that missing field returns custom default list."""
        data: dict = {"meta": {}}
        custom_default = [{"placeholder": True}]
        result = safe_get_list(data, "meta", "synonyms", default=custom_default)
        assert result == custom_default

    def test_non_list_value_returns_default(self):
        """Test that non-list value returns default (type safety)."""
        data = {"meta": {"synonyms": "not_a_list"}}
        result = safe_get_list(data, "meta", "synonyms", default=[])
        assert result == []

    def test_none_value_returns_default(self):
        """Test that None value returns default."""
        data = {"meta": {"synonyms": None}}
        result = safe_get_list(data, "meta", "synonyms", default=[])
        assert result == []

    def test_integer_value_returns_default(self):
        """Test that integer value returns default (wrong type)."""
        data = {"meta": {"synonyms": 123}}
        result = safe_get_list(data, "meta", "synonyms", default=[])
        assert result == []

    def test_dict_value_returns_default(self):
        """Test that dict value returns default (wrong type)."""
        data = {"meta": {"synonyms": {"val": "not_a_list"}}}
        result = safe_get_list(data, "meta", "synonyms", default=[])
        assert result == []

    def test_empty_list_preserved(self):
        """Test that empty list is returned (not replaced with default)."""
        data = {"meta": {"synonyms": []}}
        result = safe_get_list(data, "meta", "synonyms", default=[{"fallback": True}])
        assert result == []

    def test_nested_lists_preserved(self):
        """Test that nested list structures are preserved."""
        data = {"meta": {"nested": [[1, 2], [3, 4]]}}
        result = safe_get_list(data, "meta", "nested")
        assert result == [[1, 2], [3, 4]]

    def test_list_of_mixed_types(self):
        """Test list containing mixed types."""
        data = {"meta": {"mixed": [1, "two", {"three": 3}, [4]]}}
        result = safe_get_list(data, "meta", "mixed")
        assert result == [1, "two", {"three": 3}, [4]]


class TestLogMissingField:
    """Tests for log_missing_field() helper function."""

    def test_logs_at_debug_level_by_default(self, caplog):
        """Test that missing field is logged at DEBUG level by default."""
        with caplog.at_level(logging.DEBUG):
            log_missing_field("HP:0001250", "meta.definition.val")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "DEBUG"
        assert "HP:0001250" in caplog.text
        assert "meta.definition.val" in caplog.text
        assert "missing optional field" in caplog.text

    def test_logs_at_info_level(self, caplog):
        """Test logging at INFO level."""
        with caplog.at_level(logging.INFO):
            log_missing_field("HP:0001250", "meta.synonyms", level="info")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"

    def test_logs_at_warning_level(self, caplog):
        """Test logging at WARNING level."""
        with caplog.at_level(logging.WARNING):
            log_missing_field("HP:0001250", "meta.comments", level="warning")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"

    def test_logs_at_error_level(self, caplog):
        """Test logging at ERROR level."""
        with caplog.at_level(logging.ERROR):
            log_missing_field("HP:0001250", "critical.field", level="error")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"

    def test_invalid_level_defaults_to_debug(self, caplog):
        """Test that invalid level parameter defaults to DEBUG."""
        with caplog.at_level(logging.DEBUG):
            log_missing_field("HP:0001250", "meta.field", level="invalid_level")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "DEBUG"

    def test_message_format_includes_all_info(self, caplog):
        """Test that log message contains term ID and field path."""
        with caplog.at_level(logging.DEBUG):
            log_missing_field("HP:0001250", "meta.definition.val")

        message = caplog.records[0].message
        assert "HP:0001250" in message
        assert "meta.definition.val" in message
        assert "missing optional field" in message


class TestLogParsingSummary:
    """Tests for log_parsing_summary() helper function."""

    def test_logs_summary_with_statistics(self, caplog):
        """Test that summary logs all statistics correctly."""
        stats = {"definitions": 150, "synonyms": 45, "comments": 1200}
        with caplog.at_level(logging.INFO):
            log_parsing_summary(stats, total_terms=19534)

        assert "HPO Parsing Summary" in caplog.text
        assert "Total terms parsed: 19534" in caplog.text
        assert "definitions: 150" in caplog.text
        assert "synonyms: 45" in caplog.text
        assert "comments: 1200" in caplog.text

    def test_calculates_percentages_correctly(self, caplog):
        """Test that percentages are calculated correctly."""
        stats = {"definitions": 100}
        with caplog.at_level(logging.INFO):
            log_parsing_summary(stats, total_terms=1000)

        # Should show 10.0%
        assert "10.0%" in caplog.text

    def test_skips_zero_count_fields(self, caplog):
        """Test that fields with zero count are not logged."""
        stats = {"definitions": 0, "synonyms": 50, "comments": 0}
        with caplog.at_level(logging.INFO):
            log_parsing_summary(stats, total_terms=1000)

        assert "definitions" not in caplog.text
        assert "synonyms: 50" in caplog.text
        assert "comments" not in caplog.text  # Zero count, should be skipped

    def test_handles_empty_stats(self, caplog):
        """Test handling of empty statistics dictionary."""
        stats: dict[str, int] = {}
        with caplog.at_level(logging.INFO):
            log_parsing_summary(stats, total_terms=1000)

        assert "HPO Parsing Summary" in caplog.text
        assert "Total terms parsed: 1000" in caplog.text
        # No field-specific lines should appear

    def test_handles_zero_total_terms(self, caplog):
        """Test handling of zero total terms (edge case)."""
        stats = {"definitions": 100}
        with caplog.at_level(logging.WARNING):
            log_parsing_summary(stats, total_terms=0)

        assert "No terms parsed" in caplog.text
        assert "cannot calculate percentages" in caplog.text

    def test_sorts_field_names_alphabetically(self, caplog):
        """Test that field names are sorted in output."""
        stats = {"zebra": 10, "alpha": 20, "beta": 30}
        with caplog.at_level(logging.INFO):
            log_parsing_summary(stats, total_terms=100)

        lines = caplog.text.split("\n")
        # Find lines with field names
        field_lines = [line for line in lines if "Missing" in line]

        # Verify we have exactly 3 field lines before accessing by index
        assert len(field_lines) == 3, f"Expected 3 field lines, got {len(field_lines)}"

        # Should be in alphabetical order
        assert "alpha" in field_lines[0]
        assert "beta" in field_lines[1]
        assert "zebra" in field_lines[2]

    def test_handles_large_numbers(self, caplog):
        """Test with realistic large HPO dataset numbers."""
        stats = {
            "definitions": 1523,
            "synonyms": 458,
            "comments": 12456,
            "xrefs": 89,
        }
        with caplog.at_level(logging.INFO):
            log_parsing_summary(stats, total_terms=19534)

        assert "19534" in caplog.text
        # All stats should be present
        for field, count in stats.items():
            assert f"{field}: {count}" in caplog.text
