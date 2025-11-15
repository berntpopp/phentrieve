"""Real unit tests for dense_retriever module (actual code execution)."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from phentrieve.config import MIN_SIMILARITY_THRESHOLD
from phentrieve.retrieval.dense_retriever import DenseRetriever, connect_to_chroma

pytestmark = pytest.mark.unit


class TestConnectToChroma:
    """Test connect_to_chroma function with real logic execution."""

    @patch("phentrieve.retrieval.dense_retriever.chromadb.PersistentClient")
    def test_successful_connection(self, mock_client_class):
        """Test successful connection to ChromaDB collection."""
        # Arrange
        mock_collection = Mock()
        mock_collection.count.return_value = 100

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        # Act
        result = connect_to_chroma("/fake/index", "test_collection")

        # Assert
        assert result == mock_collection
        mock_client.get_collection.assert_called_once_with(name="test_collection")
        mock_collection.count.assert_called_once()

    @patch("phentrieve.retrieval.dense_retriever.chromadb.PersistentClient")
    def test_collection_not_found_no_alternates(self, mock_client_class):
        """Test when collection not found and no alternate collections exist."""
        # Arrange
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.list_collections.return_value = []
        mock_client_class.return_value = mock_client

        # Act
        result = connect_to_chroma("/fake/index", "missing_collection")

        # Assert
        assert result is None

    @patch("phentrieve.retrieval.dense_retriever.chromadb.PersistentClient")
    @patch("phentrieve.retrieval.dense_retriever.generate_collection_name")
    def test_collection_not_found_with_alternate(
        self, mock_gen_name, mock_client_class
    ):
        """Test finding alternate collection when primary not found."""
        # Arrange
        mock_alternate_collection = Mock()
        mock_alternate_collection.count.return_value = 50

        mock_collection_obj_1 = Mock()
        mock_collection_obj_1.name = "alternate_name"

        mock_client = Mock()
        mock_client.get_collection.side_effect = [
            Exception("Not found"),  # First call fails
            mock_alternate_collection,  # Second call succeeds
        ]
        mock_client.list_collections.return_value = [mock_collection_obj_1]
        mock_client_class.return_value = mock_client

        mock_gen_name.return_value = "alternate_name"

        # Act
        result = connect_to_chroma(
            "/fake/index", "original_name", model_name="some-model"
        )

        # Assert
        assert result == mock_alternate_collection

    @patch("phentrieve.retrieval.dense_retriever.chromadb.PersistentClient")
    def test_chromadb_connection_error(self, mock_client_class):
        """Test error handling when ChromaDB connection fails."""
        # Arrange
        mock_client_class.side_effect = Exception("Connection failed")

        # Act
        result = connect_to_chroma("/fake/index", "test_collection")

        # Assert
        assert result is None


class TestDenseRetrieverInit:
    """Test DenseRetriever initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()

        # Act
        retriever = DenseRetriever(mock_model, mock_collection)

        # Assert
        assert retriever.model == mock_model
        assert retriever.collection == mock_collection
        assert retriever.min_similarity == MIN_SIMILARITY_THRESHOLD
        assert retriever.model_name is None
        assert retriever.index_base_path is None

    def test_init_with_custom_threshold(self):
        """Test initialization with custom similarity threshold."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()
        custom_threshold = 0.8

        # Act
        retriever = DenseRetriever(
            mock_model, mock_collection, min_similarity=custom_threshold
        )

        # Assert
        assert retriever.min_similarity == custom_threshold


class TestDenseRetrieverFromModelName:
    """Test DenseRetriever.from_model_name class method."""

    @patch("phentrieve.retrieval.dense_retriever.connect_to_chroma")
    @patch("phentrieve.retrieval.dense_retriever.resolve_data_path")
    @patch("phentrieve.retrieval.dense_retriever.generate_collection_name")
    def test_successful_creation_with_default_index(
        self, mock_gen_name, mock_resolve_path, mock_connect
    ):
        """Test successful retriever creation with default index directory."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()
        mock_index_dir = Path("/fake/indexes")

        mock_gen_name.return_value = "test_collection"
        mock_resolve_path.return_value = mock_index_dir
        mock_connect.return_value = mock_collection

        # Mock Path.exists() and is_dir()
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_dir", return_value=True),
        ):
            # Act
            retriever = DenseRetriever.from_model_name(mock_model, "test-model")

            # Assert
            assert retriever is not None
            assert retriever.model == mock_model
            assert retriever.collection == mock_collection
            assert retriever.model_name == "test-model"
            assert retriever.index_base_path == mock_index_dir

    @patch("phentrieve.retrieval.dense_retriever.connect_to_chroma")
    @patch("phentrieve.retrieval.dense_retriever.generate_collection_name")
    def test_successful_creation_with_explicit_index(self, mock_gen_name, mock_connect):
        """Test successful retriever creation with explicit index directory."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()
        explicit_index = "/explicit/index"

        mock_gen_name.return_value = "test_collection"
        mock_connect.return_value = mock_collection

        # Mock Path.exists() and is_dir()
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_dir", return_value=True),
        ):
            # Act
            retriever = DenseRetriever.from_model_name(
                mock_model, "test-model", index_dir=explicit_index
            )

            # Assert
            assert retriever is not None
            assert retriever.index_base_path == Path(explicit_index)

    @patch("phentrieve.retrieval.dense_retriever.resolve_data_path")
    @patch("phentrieve.retrieval.dense_retriever.generate_collection_name")
    def test_index_directory_not_found(self, mock_gen_name, mock_resolve_path):
        """Test when index directory doesn't exist."""
        # Arrange
        mock_model = Mock()
        mock_index_dir = Path("/nonexistent/indexes")

        mock_gen_name.return_value = "test_collection"
        mock_resolve_path.return_value = mock_index_dir

        # Mock Path.exists() to return False
        with patch.object(Path, "exists", return_value=False):
            # Act
            retriever = DenseRetriever.from_model_name(mock_model, "test-model")

            # Assert
            assert retriever is None

    @patch("phentrieve.retrieval.dense_retriever.connect_to_chroma")
    @patch("phentrieve.retrieval.dense_retriever.resolve_data_path")
    @patch("phentrieve.retrieval.dense_retriever.generate_collection_name")
    def test_connection_failure(self, mock_gen_name, mock_resolve_path, mock_connect):
        """Test when connection to ChromaDB fails."""
        # Arrange
        mock_model = Mock()
        mock_index_dir = Path("/fake/indexes")

        mock_gen_name.return_value = "test_collection"
        mock_resolve_path.return_value = mock_index_dir
        mock_connect.return_value = None  # Connection failed

        # Mock Path.exists() and is_dir()
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_dir", return_value=True),
        ):
            # Act
            retriever = DenseRetriever.from_model_name(mock_model, "test-model")

            # Assert
            assert retriever is None


class TestDenseRetrieverQuery:
    """Test DenseRetriever.query method."""

    def test_successful_query(self):
        """Test successful query execution."""
        # Arrange
        mock_model = Mock()
        mock_device = Mock()
        mock_device.type = "cpu"
        mock_model.parameters.return_value = iter([Mock(device=mock_device)])
        # Return numpy array - model.encode()[0] will be a numpy array with .tolist()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [["HP:0001"]],
            "documents": [["Test document"]],
            "metadatas": [[{"hpo_id": "HP:0001"}]],
            "distances": [[0.5]],
        }

        retriever = DenseRetriever(mock_model, mock_collection)

        # Act
        results = retriever.query("test query", n_results=5)

        # Assert
        assert "ids" in results
        assert "documents" in results
        assert "similarities" in results
        mock_model.encode.assert_called_once()
        mock_collection.query.assert_called_once()

    def test_query_without_similarities(self):
        """Test query without including similarity scores."""
        # Arrange
        mock_model = Mock()
        mock_device = Mock()
        mock_device.type = "cpu"
        mock_model.parameters.return_value = iter([Mock(device=mock_device)])
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [["HP:0001"]],
            "documents": [["Test document"]],
            "metadatas": [[{"hpo_id": "HP:0001"}]],
            "distances": [[0.5]],
        }

        retriever = DenseRetriever(mock_model, mock_collection)

        # Act
        results = retriever.query("test query", include_similarities=False)

        # Assert
        assert "ids" in results
        assert "similarities" not in results

    def test_query_with_cuda_device(self):
        """Test query with CUDA device."""
        # Arrange
        mock_model = Mock()
        mock_device = Mock()
        mock_device.type = "cuda"
        mock_model.parameters.return_value = iter([Mock(device=mock_device)])
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever = DenseRetriever(mock_model, mock_collection)

        # Act
        retriever.query("test query")

        # Assert
        # Verify encode was called with device="cuda"
        call_args = mock_model.encode.call_args
        assert call_args[1]["device"] == "cuda"

    def test_query_error_handling(self):
        """Test error handling during query."""
        # Arrange
        mock_model = Mock()
        mock_model.parameters.side_effect = Exception("Model error")

        mock_collection = Mock()

        retriever = DenseRetriever(mock_model, mock_collection)

        # Act
        results = retriever.query("test query")

        # Assert
        assert results == {"ids": [], "documents": [], "metadatas": [], "distances": []}


class TestDenseRetrieverFilterResults:
    """Test DenseRetriever.filter_results method."""

    @patch("phentrieve.retrieval.dense_retriever.calculate_similarity")
    def test_filter_by_similarity_threshold(self, mock_calc_sim):
        """Test filtering results by similarity threshold."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()
        retriever = DenseRetriever(mock_model, mock_collection, min_similarity=0.5)

        # Mock calculate_similarity to return decreasing values
        mock_calc_sim.side_effect = [0.9, 0.7, 0.3, 0.2]

        results = {
            "ids": [["HP:0001", "HP:0002", "HP:0003", "HP:0004"]],
            "documents": [["doc1", "doc2", "doc3", "doc4"]],
            "metadatas": [[{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}]],
            "distances": [[0.1, 0.3, 0.7, 0.8]],
        }

        # Act
        filtered = retriever.filter_results(results)

        # Assert
        assert len(filtered["ids"][0]) == 2  # Only 2 results above 0.5 threshold
        assert filtered["ids"][0] == ["HP:0001", "HP:0002"]

    @patch("phentrieve.retrieval.dense_retriever.calculate_similarity")
    def test_filter_by_max_results(self, mock_calc_sim):
        """Test filtering results by maximum count."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()
        retriever = DenseRetriever(mock_model, mock_collection, min_similarity=0.1)

        # All similarities above threshold
        mock_calc_sim.side_effect = [0.9, 0.8, 0.7, 0.6]

        results = {
            "ids": [["HP:0001", "HP:0002", "HP:0003", "HP:0004"]],
            "documents": [["doc1", "doc2", "doc3", "doc4"]],
            "metadatas": [[{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}]],
            "distances": [[0.1, 0.2, 0.3, 0.4]],
        }

        # Act
        filtered = retriever.filter_results(results, max_results=2)

        # Assert
        assert len(filtered["ids"][0]) == 2
        assert filtered["ids"][0] == ["HP:0001", "HP:0002"]  # Top 2 by similarity

    def test_filter_empty_results(self):
        """Test filtering with empty results."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()
        retriever = DenseRetriever(mock_model, mock_collection)

        results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # Act
        filtered = retriever.filter_results(results)

        # Assert
        assert filtered == results

    @patch("phentrieve.retrieval.dense_retriever.calculate_similarity")
    def test_filter_with_existing_similarities(self, mock_calc_sim):
        """Test filtering when similarities already included in results."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()
        retriever = DenseRetriever(mock_model, mock_collection, min_similarity=0.5)

        results = {
            "ids": [["HP:0001", "HP:0002"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"id": "1"}, {"id": "2"}]],
            "distances": [[0.1, 0.5]],
            "similarities": [[0.9, 0.5]],  # Already calculated
        }

        # Act
        filtered = retriever.filter_results(results)

        # Assert
        # Should not call calculate_similarity since similarities already exist
        mock_calc_sim.assert_not_called()
        assert len(filtered["ids"][0]) == 2

    @patch("phentrieve.retrieval.dense_retriever.calculate_similarity")
    def test_filter_override_threshold(self, mock_calc_sim):
        """Test overriding default similarity threshold."""
        # Arrange
        mock_model = Mock()
        mock_collection = Mock()
        retriever = DenseRetriever(mock_model, mock_collection, min_similarity=0.3)

        mock_calc_sim.side_effect = [0.9, 0.6, 0.4]

        results = {
            "ids": [["HP:0001", "HP:0002", "HP:0003"]],
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [[{"id": "1"}, {"id": "2"}, {"id": "3"}]],
            "distances": [[0.1, 0.4, 0.6]],
        }

        # Act
        filtered = retriever.filter_results(results, min_similarity=0.7)

        # Assert
        assert len(filtered["ids"][0]) == 1  # Only 1 result above 0.7
        assert filtered["ids"][0] == ["HP:0001"]
