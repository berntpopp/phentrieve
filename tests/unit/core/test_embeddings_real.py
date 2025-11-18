"""Real unit tests for embeddings module (actual code execution)."""

from unittest.mock import Mock, patch

import pytest

from phentrieve.config import DEFAULT_BIOLORD_MODEL, JINA_MODEL_ID
from phentrieve.embeddings import clear_model_registry, load_embedding_model

pytestmark = pytest.mark.unit


class TestLoadEmbeddingModel:
    """Test load_embedding_model function with real logic execution."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_model_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_model_registry()

    @patch("phentrieve.embeddings.SentenceTransformer")
    @patch("phentrieve.embeddings.torch.cuda.is_available")
    def test_default_model_cpu(self, mock_cuda, mock_st):
        """Test loading default model on CPU."""
        # Arrange
        mock_cuda.return_value = False
        mock_model = Mock()
        mock_st.return_value = mock_model

        # Act
        load_embedding_model()

        # Assert
        mock_st.assert_called_once_with(DEFAULT_BIOLORD_MODEL, trust_remote_code=True)
        mock_model.to.assert_called_once_with("cpu")
        # model.to() returns the model itself

    @patch("phentrieve.embeddings.SentenceTransformer")
    @patch("phentrieve.embeddings.torch.cuda.is_available")
    def test_cuda_device_selection(self, mock_cuda, mock_st):
        """Test CUDA device selection when available."""
        # Arrange
        mock_cuda.return_value = True
        mock_model = Mock()
        mock_st.return_value = mock_model

        # Act
        load_embedding_model()

        # Assert
        mock_model.to.assert_called_once_with("cuda")

    @patch("phentrieve.embeddings.SentenceTransformer")
    @patch("phentrieve.embeddings.torch.cuda.is_available")
    def test_explicit_device_override(self, mock_cuda, mock_st):
        """Test explicit device parameter overrides auto-detection."""
        # Arrange
        mock_cuda.return_value = True  # CUDA available
        mock_model = Mock()
        mock_st.return_value = mock_model

        # Act
        load_embedding_model(device="cpu")

        # Assert
        mock_model.to.assert_called_once_with("cpu")  # Should use explicit CPU

    @patch("phentrieve.embeddings.SentenceTransformer")
    @patch("phentrieve.embeddings.torch.cuda.is_available")
    def test_jina_model_special_handling(self, mock_cuda, mock_st):
        """Test Jina model loads with trust_remote_code=True."""
        # Arrange
        mock_cuda.return_value = False
        mock_model = Mock()
        mock_st.return_value = mock_model

        # Act
        load_embedding_model(model_name=JINA_MODEL_ID)

        # Assert
        mock_st.assert_called_once_with(JINA_MODEL_ID, trust_remote_code=True)

    @patch("phentrieve.embeddings.SentenceTransformer")
    @patch("phentrieve.embeddings.torch.cuda.is_available")
    def test_biolord_model_special_handling(self, mock_cuda, mock_st):
        """Test BioLORD model loads with trust_remote_code=True."""
        # Arrange
        mock_cuda.return_value = False
        mock_model = Mock()
        mock_st.return_value = mock_model

        # Act
        load_embedding_model(model_name=DEFAULT_BIOLORD_MODEL)

        # Assert
        mock_st.assert_called_once_with(DEFAULT_BIOLORD_MODEL, trust_remote_code=True)

    @patch("phentrieve.embeddings.SentenceTransformer")
    @patch("phentrieve.embeddings.torch.cuda.is_available")
    def test_trust_remote_code_parameter(self, mock_cuda, mock_st):
        """Test trust_remote_code parameter is respected."""
        # Arrange
        mock_cuda.return_value = False
        mock_model = Mock()
        mock_st.return_value = mock_model
        custom_model = "some-custom/model"

        # Act
        load_embedding_model(model_name=custom_model, trust_remote_code=True)

        # Assert
        mock_st.assert_called_once_with(custom_model, trust_remote_code=True)

    @patch("phentrieve.embeddings.SentenceTransformer")
    @patch("phentrieve.embeddings.torch.cuda.is_available")
    def test_standard_model_loading(self, mock_cuda, mock_st):
        """Test standard model loading without trust_remote_code."""
        # Arrange
        mock_cuda.return_value = False
        mock_model = Mock()
        mock_st.return_value = mock_model
        standard_model = "sentence-transformers/all-MiniLM-L6-v2"

        # Act
        load_embedding_model(model_name=standard_model)

        # Assert
        mock_st.assert_called_once_with(standard_model)

    @patch("phentrieve.embeddings.SentenceTransformer")
    @patch("phentrieve.embeddings.torch.cuda.is_available")
    def test_error_handling(self, mock_cuda, mock_st):
        """Test error handling when model loading fails."""
        # Arrange
        mock_cuda.return_value = False
        mock_st.side_effect = Exception("Model not found")

        # Act & Assert
        with pytest.raises(ValueError, match="Error loading SentenceTransformer model"):
            load_embedding_model(model_name="invalid-model")
