"""Unit test fixtures (mocked dependencies)."""

import pytest


@pytest.fixture
def mock_spacy_nlp(mocker):
    """Mock spaCy NLP model."""
    mock_nlp = mocker.MagicMock()
    mock_doc = mocker.MagicMock()
    mock_nlp.return_value = mock_doc
    return mock_nlp
