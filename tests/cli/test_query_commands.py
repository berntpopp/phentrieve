"""
Tests for the HPO term query CLI commands.

This module tests the CLI commands for querying HPO terms, including
the new output format and file output capabilities.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from phentrieve.cli import app
from phentrieve.retrieval.output_formatters import (
    format_results_as_text,
    format_results_as_json,
    format_results_as_jsonl,
)

runner = CliRunner()

# Mock query results for testing
MOCK_QUERY_RESULTS = [
    {
        "query_text_processed": "seizures",
        "header_info": "Found 2 matching HPO terms:",
        "results": [
            {
                "rank": 1,
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.92,
            },
            {
                "rank": 2,
                "hpo_id": "HP:0012638",
                "label": "Abnormality of nervous system physiology",
                "similarity": 0.75,
            },
        ],
    }
]

# Mock reranked query results for testing
MOCK_RERANKED_RESULTS = [
    {
        "query_text_processed": "seizures",
        "header_info": "Found 2 matching HPO terms (reranked):",
        "results": [
            {
                "rank": 1,
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.92,
                "cross_encoder_score": 0.98,
                "original_rank": 1,
            },
            {
                "rank": 2,
                "hpo_id": "HP:0012638",
                "label": "Abnormality of nervous system physiology",
                "similarity": 0.75,
                "cross_encoder_score": 0.80,
                "original_rank": 2,
            },
        ],
    }
]


@pytest.fixture
def mock_query_results():
    """Mock the query results returned by the query orchestrator."""
    with patch(
        "phentrieve.retrieval.query_orchestrator.orchestrate_query"
    ) as mock_orchestrate:
        # Return the mock results
        mock_orchestrate.return_value = MOCK_QUERY_RESULTS
        yield mock_orchestrate


@pytest.fixture
def mock_reranked_query_results():
    """Mock the reranked query results returned by the query orchestrator."""
    with patch(
        "phentrieve.retrieval.query_orchestrator.orchestrate_query"
    ) as mock_orchestrate:
        # Return the mock reranked results
        mock_orchestrate.return_value = MOCK_RERANKED_RESULTS
        yield mock_orchestrate


def test_query_basic(mock_query_results):
    """Test basic query command with default text output."""
    result = runner.invoke(app, ["query", "seizures"])

    assert result.exit_code == 0
    assert "Querying for HPO terms with text: 'seizures'" in result.stdout
    assert "Query processing completed successfully!" in result.stdout

    # Verify correct orchestrator parameters
    mock_query_results.assert_called_once()
    args, kwargs = mock_query_results.call_args
    assert kwargs["query_text"] == "seizures"
    assert kwargs["similarity_threshold"] == 0.3  # Default threshold
    assert kwargs["num_results"] == 10  # Default num_results

    # Verify the output format (text)
    expected_text = format_results_as_text(MOCK_QUERY_RESULTS, sentence_mode=False)
    assert expected_text in result.stdout


def test_query_with_output_format_json(mock_query_results):
    """Test query command with JSON output format."""
    result = runner.invoke(app, ["query", "seizures", "--output-format", "json"])

    assert result.exit_code == 0
    assert "Querying for HPO terms with text: 'seizures'" in result.stdout
    assert "Query processing completed successfully!" in result.stdout

    # Verify the output format (JSON)
    expected_json = format_results_as_json(MOCK_QUERY_RESULTS, sentence_mode=False)
    assert expected_json in result.stdout


def test_query_with_output_format_jsonl(mock_query_results):
    """Test query command with JSON Lines output format."""
    result = runner.invoke(app, ["query", "seizures", "--output-format", "json_lines"])

    assert result.exit_code == 0
    assert "Querying for HPO terms with text: 'seizures'" in result.stdout
    assert "Query processing completed successfully!" in result.stdout

    # Verify the output format (JSON Lines)
    expected_jsonl = format_results_as_jsonl(MOCK_QUERY_RESULTS)
    assert expected_jsonl in result.stdout


def test_query_with_output_file(mock_query_results):
    """Test query command with output to file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "test_output.txt"

        result = runner.invoke(
            app, ["query", "seizures", "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert "Querying for HPO terms with text: 'seizures'" in result.stdout
        assert f"Results saved to {output_file}" in result.stdout
        assert "Query processing completed successfully!" in result.stdout

        # Verify file was created with correct content
        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            file_content = f.read()
            expected_text = format_results_as_text(
                MOCK_QUERY_RESULTS, sentence_mode=False
            )
            assert file_content == expected_text


def test_query_with_output_file_json(mock_query_results):
    """Test query command with output to file in JSON format."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "test_output.json"

        result = runner.invoke(
            app,
            [
                "query",
                "seizures",
                "--output-file",
                str(output_file),
                "--output-format",
                "json",
            ],
        )

        assert result.exit_code == 0
        assert f"Results saved to {output_file}" in result.stdout

        # Verify file was created with correct content
        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            file_content = f.read()
            expected_json = format_results_as_json(
                MOCK_QUERY_RESULTS, sentence_mode=False
            )
            assert file_content == expected_json


def test_query_with_output_file_jsonl(mock_query_results):
    """Test query command with output to file in JSON Lines format."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "test_output.jsonl"

        result = runner.invoke(
            app,
            [
                "query",
                "seizures",
                "--output-file",
                str(output_file),
                "--output-format",
                "json_lines",
            ],
        )

        assert result.exit_code == 0
        assert f"Results saved to {output_file}" in result.stdout

        # Verify file was created with correct content
        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            file_content = f.read()
            expected_jsonl = format_results_as_jsonl(MOCK_QUERY_RESULTS)
            assert file_content == expected_jsonl


def test_query_with_reranking(mock_reranked_query_results):
    """Test query command with reranking enabled."""
    result = runner.invoke(app, ["query", "seizures", "--enable-reranker"])

    assert result.exit_code == 0
    assert "Querying for HPO terms with text: 'seizures'" in result.stdout
    assert "Query processing completed successfully!" in result.stdout

    # Verify correct orchestrator parameters
    mock_reranked_query_results.assert_called_once()
    args, kwargs = mock_reranked_query_results.call_args
    assert kwargs["enable_reranker"] is True

    # Verify the output shows reranking information
    expected_text = format_results_as_text(MOCK_RERANKED_RESULTS, sentence_mode=False)
    assert expected_text in result.stdout
    assert "reranked" in result.stdout


def test_query_invalid_output_format(mock_query_results):
    """Test query command with invalid output format."""
    result = runner.invoke(
        app, ["query", "seizures", "--output-format", "invalid_format"]
    )

    assert result.exit_code == 1
    assert "Error: Unsupported output format 'invalid_format'" in result.stdout
    assert "Supported formats: text, json, json_lines" in result.stdout


def test_query_missing_text():
    """Test query command without providing text."""
    result = runner.invoke(app, ["query"])

    assert result.exit_code == 1
    assert (
        "Error: Text argument is required when not in interactive mode" in result.stdout
    )


def test_query_output_file_write_error(mock_query_results):
    """Test query command with an invalid output file path."""
    # Try to write to a directory that doesn't exist
    invalid_path = "/nonexistent/dir/output.txt"

    with patch("phentrieve.cli.query_commands.open") as mock_open:
        # Simulate a permission or filesystem error
        mock_open.side_effect = PermissionError("Permission denied")

        result = runner.invoke(
            app, ["query", "seizures", "--output-file", invalid_path]
        )

        assert result.exit_code == 1
        assert "Error writing to file: Permission denied" in result.stdout
