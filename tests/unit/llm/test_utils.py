"""Unit tests for shared LLM utility functions."""

from phentrieve.llm.types import AssertionStatus
from phentrieve.llm.utils import extract_json, normalize_hpo_id, parse_assertion


class TestExtractJson:
    """Tests for extract_json."""

    def test_extract_from_code_block(self):
        text = '```json\n{"annotations": [{"hpo_id": "HP:0001250"}]}\n```'
        result = extract_json(text)
        assert result is not None
        assert result["annotations"][0]["hpo_id"] == "HP:0001250"

    def test_extract_from_code_block_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_extract_raw_json(self):
        text = 'Here is the result: {"annotations": []}'
        result = extract_json(text)
        assert result == {"annotations": []}

    def test_no_json_returns_none(self):
        assert extract_json("no json here") is None

    def test_invalid_json_returns_none(self):
        text = "```json\n{invalid json}\n```"
        assert extract_json(text) is None

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = extract_json(text)
        assert result is not None
        assert result["outer"]["inner"] == [1, 2, 3]


class TestNormalizeHpoId:
    """Tests for normalize_hpo_id."""

    def test_already_correct(self):
        assert normalize_hpo_id("HP:0001250") == "HP:0001250"

    def test_missing_prefix(self):
        assert normalize_hpo_id("0001250") == "HP:0001250"

    def test_lowercase(self):
        assert normalize_hpo_id("hp:0001250") == "HP:0001250"

    def test_underscore_separator(self):
        assert normalize_hpo_id("HP_1250") == "HP:0001250"

    def test_dash_separator(self):
        assert normalize_hpo_id("HP-1250") == "HP:0001250"

    def test_space_separator(self):
        assert normalize_hpo_id("HP 1250") == "HP:0001250"

    def test_no_separator(self):
        assert normalize_hpo_id("HP1250") == "HP:0001250"

    def test_whitespace_stripping(self):
        assert normalize_hpo_id("  HP:0001250  ") == "HP:0001250"

    def test_invalid_format(self):
        assert normalize_hpo_id("INVALID") is None

    def test_empty_string(self):
        assert normalize_hpo_id("") is None


class TestParseAssertion:
    """Tests for parse_assertion."""

    def test_affirmed(self):
        assert parse_assertion("affirmed") == AssertionStatus.AFFIRMED

    def test_negated_variants(self):
        for val in ("negated", "negative", "absent", "excluded", "no", "denied"):
            assert parse_assertion(val) == AssertionStatus.NEGATED, (
                f"Failed for '{val}'"
            )

    def test_uncertain_variants(self):
        for val in ("uncertain", "possible", "suspected", "probable"):
            assert parse_assertion(val) == AssertionStatus.UNCERTAIN, (
                f"Failed for '{val}'"
            )

    def test_case_insensitive(self):
        assert parse_assertion("NEGATED") == AssertionStatus.NEGATED
        assert parse_assertion("Uncertain") == AssertionStatus.UNCERTAIN

    def test_whitespace_stripping(self):
        assert parse_assertion("  negated  ") == AssertionStatus.NEGATED

    def test_unknown_defaults_to_affirmed(self):
        assert parse_assertion("present") == AssertionStatus.AFFIRMED
        assert parse_assertion("yes") == AssertionStatus.AFFIRMED

    def test_enum_passthrough(self):
        assert parse_assertion(AssertionStatus.NEGATED) == AssertionStatus.NEGATED
        assert parse_assertion(AssertionStatus.AFFIRMED) == AssertionStatus.AFFIRMED
