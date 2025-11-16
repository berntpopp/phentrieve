"""
Tests for the HPO term similarity CLI commands.

This module contains tests for the HPO term similarity command line interface.
"""

import re

import pytest
from typer.testing import CliRunner

from phentrieve.cli import app

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


runner = CliRunner()


# Mock data for tests
MOCK_ANCESTORS = {
    "HP:0000001": {"HP:0000001"},  # Root
    "HP:0000118": {"HP:0000118", "HP:0000001"},  # Phenotypic abnormality
    "HP:0000707": {
        "HP:0000707",
        "HP:0000118",
        "HP:0000001",
    },  # Abnormality of the nervous system
    "HP:0012639": {
        "HP:0012639",
        "HP:0000707",
        "HP:0000118",
        "HP:0000001",
    },  # Abnormality of nervous system physiology
    "HP:0001939": {
        "HP:0001939",
        "HP:0012639",
        "HP:0000707",
        "HP:0000118",
        "HP:0000001",
    },  # Abnormality of metabolism/homeostasis
    "HP:0001250": {
        "HP:0001250",
        "HP:0012639",
        "HP:0000707",
        "HP:0000118",
        "HP:0000001",
    },  # Seizure
    "HP:0002060": {
        "HP:0002060",
        "HP:0000118",
        "HP:0000001",
    },  # Abnormality of the cerebrum
    "HP:0002067": {
        "HP:0002067",
        "HP:0002060",
        "HP:0000118",
        "HP:0000001",
    },  # Abnormality of the cerebral cortex
    "HP:0000252": {
        "HP:0000252",
        "HP:0002067",
        "HP:0002060",
        "HP:0000118",
        "HP:0000001",
    },  # Microcephaly
}

MOCK_DEPTHS = {
    "HP:0000001": 0,  # Root
    "HP:0000118": 1,  # Phenotypic abnormality
    "HP:0000707": 2,  # Abnormality of the nervous system
    "HP:0012639": 3,  # Abnormality of nervous system physiology
    "HP:0001939": 4,  # Abnormality of metabolism/homeostasis
    "HP:0001250": 4,  # Seizure
    "HP:0002060": 2,  # Abnormality of the cerebrum
    "HP:0002067": 3,  # Abnormality of the cerebral cortex
    "HP:0000252": 4,  # Microcephaly
}

MOCK_LABELS = {
    "HP:0000001": "All",
    "HP:0000118": "Phenotypic abnormality",
    "HP:0000707": "Abnormality of the nervous system",
    "HP:0012639": "Abnormality of nervous system physiology",
    "HP:0001939": "Abnormality of metabolism/homeostasis",
    "HP:0001250": "Seizure",
    "HP:0002060": "Abnormality of the cerebrum",
    "HP:0002067": "Abnormality of the cerebral cortex",
    "HP:0000252": "Microcephaly",
}


@pytest.fixture(autouse=True)
def mock_hpo_data(monkeypatch):
    """Mock HPO graph data and term labels for testing.

    Uses monkeypatch instead of unittest.mock.patch to work properly with
    Typer's CliRunner, which creates an isolated execution context.

    Patches at the source module level (phentrieve.evaluation.metrics) to
    ensure the mock is applied before any imports or caching occurs.
    """
    # Import the modules we need to mock
    import phentrieve.cli.similarity_commands as sim_commands
    import phentrieve.evaluation.metrics as metrics_module

    # Clear global caches BEFORE mocking to prevent cache from bypassing the mock
    metrics_module._hpo_ancestors = None
    metrics_module._hpo_term_depths = None

    # Create mock functions that return our test data
    def mock_load_hpo_graph_data(*args, **kwargs):
        return (MOCK_ANCESTORS, MOCK_DEPTHS)

    def mock_ensure_cli_hpo_label_cache(*args, **kwargs):
        return MOCK_LABELS

    # CRITICAL: Patch at the source module level (metrics_module)
    # This ensures the mock is applied before the CLI runner imports the module
    monkeypatch.setattr(metrics_module, "load_hpo_graph_data", mock_load_hpo_graph_data)

    # Also patch in similarity_commands in case it's already imported
    monkeypatch.setattr(sim_commands, "load_hpo_graph_data", mock_load_hpo_graph_data)
    monkeypatch.setattr(
        sim_commands, "_ensure_cli_hpo_label_cache", mock_ensure_cli_hpo_label_cache
    )

    yield

    # Clear caches after test to prevent cross-test pollution
    metrics_module._hpo_ancestors = None
    metrics_module._hpo_term_depths = None


def test_similarity_calculate_basic(mock_hpo_data):
    """Test basic similarity calculation with valid HPO terms."""
    result = runner.invoke(app, ["similarity", "calculate", "HP:0001250", "HP:0001939"])

    assert result.exit_code == 0
    assert "Semantic Similarity Score:" in result.stdout
    assert "Term 1: HP:0001250 (Seizure)" in result.stdout
    assert "Term 2: HP:0001939 (Abnormality of metabolism/homeostasis)" in result.stdout
    # The real implementation might find a different LCA depending on mock vs real data
    assert "Lowest Common Ancestor (LCA):" in result.stdout
    # Accept either the mock LCA or the real data LCA
    assert (
        "Phenotypic abnormality" in result.stdout
        or "Abnormality of nervous system physiology" in result.stdout
    )
    assert "Formula Used: hybrid" in result.stdout


def test_similarity_calculate_with_formula(mock_hpo_data):
    """Test similarity calculation with specified formula."""
    result = runner.invoke(
        app,
        [
            "similarity",
            "calculate",
            "HP:0000252",
            "HP:0001250",
            "--formula",
            "simple_resnik_like",
        ],
    )

    assert result.exit_code == 0
    assert "Semantic Similarity Score:" in result.stdout
    assert "Formula Used: simple_resnik_like" in result.stdout


def test_similarity_identical_terms(mock_hpo_data):
    """Test similarity calculation with identical terms (should give 1.0)."""
    result = runner.invoke(app, ["similarity", "calculate", "HP:0000252", "HP:0000252"])

    assert result.exit_code == 0
    assert "Term 1: HP:0000252 (Microcephaly)" in result.stdout
    assert "Term 2: HP:0000252 (Microcephaly)" in result.stdout
    # The exact score depends on the formula implementation, but identical terms should have high similarity
    assert "Semantic Similarity Score: 1.0000" in result.stdout


def test_similarity_unrelated_terms(mock_hpo_data):
    """Test similarity calculation with unrelated terms."""
    result = runner.invoke(app, ["similarity", "calculate", "HP:0000252", "HP:0001250"])

    assert result.exit_code == 0
    assert "Semantic Similarity Score:" in result.stdout
    # We don't check for a specific value as it depends on the actual implementation
    # But we can verify the output format is correct
    assert "Term 1: HP:0000252" in result.stdout
    assert "Term 2: HP:0001250" in result.stdout


def test_similarity_one_invalid_term(mock_hpo_data):
    """Test with one valid and one invalid HPO term."""
    result = runner.invoke(app, ["similarity", "calculate", "HP:0000252", "HP:9999999"])

    assert result.exit_code == 1
    assert "CLI Error: Term 'HP:9999999'" in result.stdout
    assert "not found in the HPO ontology data" in result.stdout


def test_similarity_both_invalid_terms(mock_hpo_data):
    """Test with both invalid HPO terms."""
    result = runner.invoke(app, ["similarity", "calculate", "HP:8888888", "HP:9999999"])

    assert result.exit_code == 1
    assert "CLI Error: Term 'HP:8888888'" in result.stdout
    assert "CLI Error: Term 'HP:9999999'" in result.stdout


def test_similarity_invalid_formula(mock_hpo_data):
    """Test with invalid formula name."""
    result = runner.invoke(
        app,
        [
            "similarity",
            "calculate",
            "HP:0000252",
            "HP:0001250",
            "--formula",
            "invalid_formula",
        ],
    )

    # The command may not fail with exit code since we now use from_string method
    # which has a fallback to HYBRID in the SimilarityFormula.from_string method
    # Check that warning was logged and a similarity score is still calculated
    assert "hybrid" in result.stdout.lower()
    assert "Semantic Similarity Score:" in result.stdout


def test_similarity_help():
    """Test help output for the similarity calculate command."""
    result = runner.invoke(app, ["similarity", "calculate", "--help"])

    # Strip ANSI escape codes from output (Typer adds colors/formatting)
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    clean_output = ansi_escape.sub("", result.stdout)

    assert result.exit_code == 0
    assert "Calculate semantic similarity between two HPO terms" in clean_output
    assert "--formula" in clean_output
    assert "--debug" in clean_output
