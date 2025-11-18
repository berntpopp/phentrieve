"""Shared fixtures for all tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_clinical_texts() -> list[str]:
    """Sample clinical texts for testing."""
    return [
        "Patient presents with seizures and developmental delay",
        "No evidence of heart disease",
        "Family history of autism spectrum disorder",
    ]


# Mocks (function-scoped for isolation)
@pytest.fixture
def mock_embedding_model(mocker):
    """Mock sentence transformer model."""
    mock = mocker.MagicMock()
    mock.encode.return_value = [[0.1] * 384]  # Mock 384-dim embedding
    return mock


@pytest.fixture
def mock_chromadb_collection(mocker):
    """Mock ChromaDB collection."""
    mock = mocker.MagicMock()
    mock.query.return_value = {
        "ids": [["HP:0001250"]],
        "distances": [[0.15]],
        "metadatas": [[{"label": "Seizure"}]],
    }
    return mock


@pytest.fixture
def benchmark_data_dir():
    """Return path to benchmark data directory."""
    return Path(__file__).parent / "data" / "benchmarks"
