"""Unit test fixtures (mocked dependencies)."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_spacy_nlp(mocker):
    """Mock spaCy NLP model."""
    mock_nlp = mocker.MagicMock()
    mock_doc = mocker.MagicMock()
    mock_nlp.return_value = mock_doc
    return mock_nlp


@pytest.fixture
def mock_sbert_model():
    """Shared mock SentenceTransformer model for unit tests."""
    model = MagicMock()
    model.encode.return_value = [[0.1] * 384]
    model.get_embedding_dimension.return_value = 384
    model.get_sentence_embedding_dimension.return_value = 384
    return model
