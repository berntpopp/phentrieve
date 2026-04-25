"""Tests for new config constants added by Plan A."""

import pytest

from phentrieve import config as phentrieve_config

pytestmark = pytest.mark.unit


def test_default_num_results_exists_and_aliases_top_k():
    assert phentrieve_config.DEFAULT_NUM_RESULTS == phentrieve_config.DEFAULT_TOP_K


def test_default_chunk_confidence_default_value():
    assert phentrieve_config.DEFAULT_CHUNK_CONFIDENCE == 0.2


def test_default_assertion_preference_default_value():
    assert phentrieve_config.DEFAULT_ASSERTION_PREFERENCE == "dependency"


def test_default_output_format_query_default_value():
    assert phentrieve_config.DEFAULT_OUTPUT_FORMAT_QUERY == "text"


def test_default_output_format_process_default_value():
    assert phentrieve_config.DEFAULT_OUTPUT_FORMAT_PROCESS == "json_lines"
