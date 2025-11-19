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


# HPO Graph Data Caching Fixtures
@pytest.fixture
def fresh_hpo_graph_data():
    """
    Opt-in fixture for tests that need fresh HPO graph data.

    Clears the cache before and after the test to ensure isolation.
    Use this fixture when your test:
    - Needs to load real HPO data (not mocked)
    - Requires fresh data without cached state
    - Tests caching behavior itself

    For most unit tests that mock load_hpo_graph_data, this fixture
    is NOT needed (mocks bypass the cache).

    Usage:
        def test_something(fresh_hpo_graph_data):
            ancestors, depths = fresh_hpo_graph_data
            assert len(ancestors) > 0

    Scope: Function (runs before/after each test)
    """
    from phentrieve.evaluation.metrics import load_hpo_graph_data

    # Clear cache before test
    load_hpo_graph_data.cache_clear()

    # Load fresh data
    data = load_hpo_graph_data()

    yield data

    # Clear cache after test for isolation
    load_hpo_graph_data.cache_clear()


@pytest.fixture
def clear_hpo_cache():
    """
    Minimal fixture that just clears the HPO graph data cache.

    Use this when you need cache clearing but don't need the actual data.
    Useful for integration tests that call functions which internally
    load HPO data.

    Usage:
        def test_similarity_calculation(clear_hpo_cache):
            # Your test that calls functions using HPO data
            result = calculate_semantic_similarity("HP:0001197", "HP:0000750")
            assert 0.0 <= result <= 1.0

    Scope: Function (runs before/after each test)
    """
    from phentrieve.evaluation.metrics import load_hpo_graph_data

    # Clear before test
    load_hpo_graph_data.cache_clear()

    yield

    # Clear after test
    load_hpo_graph_data.cache_clear()
