"""
Integration tests for vector store stability and configuration.

These tests ensure that the vector store configuration and implementation
are robust enough to support future migrations to alternative backends.
They validate correctness, reproducibility, and stability of the vector
store operations.

Test Strategy:
1. Index Build Reproducibility: Same inputs → Same index
2. Query Results Stability: Stable results across restarts
3. Full Pipeline E2E: Text → Chunks → HPO terms
4. Configuration Validation: Config creates valid connections

Design Rationale:
- These tests act as a "safety net" for future vector store migrations
- If we switch from ChromaDB to another backend (Milvus, Qdrant, etc.),
  these tests will catch any behavioral differences
- Following the expert recommendation: "Integration tests are better than
  premature abstraction" - we test the interface, not the implementation
"""

import tempfile
from pathlib import Path

import pytest

from phentrieve.config import VectorStoreConfig
from phentrieve.indexing.chromadb_indexer import build_chromadb_index
from phentrieve.retrieval.dense_retriever import DenseRetriever, connect_to_chroma
from phentrieve.utils import generate_collection_name


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for index testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_hpo_documents():
    """Provide sample HPO term documents for testing."""
    return [
        "Microcephaly: Abnormally small head circumference",
        "Seizure: Abnormal electrical activity in the brain",
        "Ataxia: Impaired coordination and balance",
    ]


@pytest.fixture
def sample_hpo_metadatas():
    """Provide sample HPO term metadata for testing."""
    return [
        {"hpo_id": "HP:0000252", "label": "Microcephaly"},
        {"hpo_id": "HP:0001250", "label": "Seizure"},
        {"hpo_id": "HP:0001251", "label": "Ataxia"},
    ]


@pytest.fixture
def sample_hpo_ids():
    """Provide sample HPO term IDs for testing."""
    return ["HP:0000252", "HP:0001250", "HP:0001251"]


class TestVectorStoreConfigCreation:
    """Test VectorStoreConfig creation and validation."""

    def test_for_chromadb_creates_valid_config(self, temp_index_dir):
        """Test that for_chromadb factory method creates valid configuration."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        config = VectorStoreConfig.for_chromadb(
            model_name=model_name,
            index_dir=temp_index_dir,
        )

        # Validate configuration attributes
        assert config.path == str(temp_index_dir)
        assert config.collection_name == generate_collection_name(model_name)
        assert config.distance_metric == "cosine"
        assert config.settings["anonymized_telemetry"] is False
        assert config.settings["allow_reset"] is True
        assert config.settings["is_persistent"] is True

    def test_config_is_immutable(self, temp_index_dir):
        """Test that VectorStoreConfig is frozen (immutable)."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        config = VectorStoreConfig.for_chromadb(
            model_name=model_name,
            index_dir=temp_index_dir,
        )

        # Attempt to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # dataclass.FrozenInstanceError
            config.path = "/new/path"  # type: ignore[misc]

    def test_config_custom_settings_override_defaults(self, temp_index_dir):
        """Test that custom settings override defaults."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        custom_settings = {
            "anonymized_telemetry": True,  # Override default (False)
            "custom_setting": "custom_value",
        }

        config = VectorStoreConfig.for_chromadb(
            model_name=model_name,
            index_dir=temp_index_dir,
            custom_settings=custom_settings,
        )

        # Custom settings should override defaults
        assert config.settings["anonymized_telemetry"] is True
        assert config.settings["custom_setting"] == "custom_value"
        # Other defaults should remain
        assert config.settings["allow_reset"] is True

    def test_to_chromadb_settings_creates_valid_settings(self, temp_index_dir):
        """Test that to_chromadb_settings() creates valid ChromaDB Settings object."""
        import chromadb

        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        config = VectorStoreConfig.for_chromadb(
            model_name=model_name,
            index_dir=temp_index_dir,
        )

        settings = config.to_chromadb_settings()

        # Validate it's a ChromaDB Settings instance
        assert isinstance(settings, chromadb.Settings)


class TestIndexBuildReproducibility:
    """Test that index building is reproducible."""

    @pytest.mark.integration
    def test_index_build_is_reproducible(
        self,
        tiny_sbert_model,
        temp_index_dir,
        sample_hpo_documents,
        sample_hpo_metadatas,
        sample_hpo_ids,
    ):
        """
        Test that building an index twice with the same data produces consistent results.

        This test ensures that:
        1. Index building is deterministic
        2. Same inputs produce same indexed data
        3. Query results are consistent across rebuilds

        This is critical for verifying that future backend migrations
        preserve data integrity.
        """
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Build index first time
        success1 = build_chromadb_index(
            documents=sample_hpo_documents,
            metadatas=sample_hpo_metadatas,
            ids=sample_hpo_ids,
            model=tiny_sbert_model,
            model_name=model_name,
            batch_size=10,
            recreate=True,
            index_dir=temp_index_dir,
        )
        assert success1, "First index build failed"

        # Query the first index
        retriever1 = DenseRetriever.from_model_name(
            model=tiny_sbert_model,
            model_name=model_name,
            index_dir=temp_index_dir,
        )
        assert retriever1 is not None, "First retriever creation failed"

        query_text = "small head"
        results1 = retriever1.query(query_text, n_results=3)

        # Rebuild index (recreate=True)
        success2 = build_chromadb_index(
            documents=sample_hpo_documents,
            metadatas=sample_hpo_metadatas,
            ids=sample_hpo_ids,
            model=tiny_sbert_model,
            model_name=model_name,
            batch_size=10,
            recreate=True,
            index_dir=temp_index_dir,
        )
        assert success2, "Second index build failed"

        # Query the rebuilt index
        retriever2 = DenseRetriever.from_model_name(
            model=tiny_sbert_model,
            model_name=model_name,
            index_dir=temp_index_dir,
        )
        assert retriever2 is not None, "Second retriever creation failed"

        results2 = retriever2.query(query_text, n_results=3)

        # Results should be identical (same order, same scores)
        assert results1["ids"] == results2["ids"], "Query IDs differ between builds"
        assert (
            results1["distances"] == results2["distances"]
        ), "Query distances differ between builds"


class TestQueryResultsStability:
    """Test that query results are stable across different scenarios."""

    @pytest.mark.integration
    def test_query_results_stable_across_connections(
        self,
        tiny_sbert_model,
        temp_index_dir,
        sample_hpo_documents,
        sample_hpo_metadatas,
        sample_hpo_ids,
    ):
        """
        Test that query results are stable when reconnecting to the same index.

        This validates that:
        1. Data persists correctly
        2. Reconnection doesn't affect results
        3. Collection state is maintained

        Critical for ensuring backend migrations don't lose data.
        """
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Build index
        build_chromadb_index(
            documents=sample_hpo_documents,
            metadatas=sample_hpo_metadatas,
            ids=sample_hpo_ids,
            model=tiny_sbert_model,
            model_name=model_name,
            recreate=True,
            index_dir=temp_index_dir,
        )

        # Query with first connection
        retriever1 = DenseRetriever.from_model_name(
            model=tiny_sbert_model,
            model_name=model_name,
            index_dir=temp_index_dir,
        )
        results1 = retriever1.query("brain activity", n_results=3)

        # Create new connection (simulates restart)
        retriever2 = DenseRetriever.from_model_name(
            model=tiny_sbert_model,
            model_name=model_name,
            index_dir=temp_index_dir,
        )
        results2 = retriever2.query("brain activity", n_results=3)

        # Results should be identical
        assert (
            results1["ids"] == results2["ids"]
        ), "Query IDs differ across connections"
        assert (
            results1["distances"] == results2["distances"]
        ), "Query distances differ across connections"

    @pytest.mark.integration
    def test_batch_query_produces_consistent_results(
        self,
        tiny_sbert_model,
        temp_index_dir,
        sample_hpo_documents,
        sample_hpo_metadatas,
        sample_hpo_ids,
    ):
        """
        Test that batch queries produce consistent results with single queries.

        This ensures that:
        1. Batch query optimization doesn't affect correctness
        2. Results match between query() and query_batch()
        3. Order and scores are preserved

        Important for performance optimization validation.
        """
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Build index
        build_chromadb_index(
            documents=sample_hpo_documents,
            metadatas=sample_hpo_metadatas,
            ids=sample_hpo_ids,
            model=tiny_sbert_model,
            model_name=model_name,
            recreate=True,
            index_dir=temp_index_dir,
        )

        retriever = DenseRetriever.from_model_name(
            model=tiny_sbert_model,
            model_name=model_name,
            index_dir=temp_index_dir,
        )
        assert retriever is not None

        query_texts = ["small head", "brain activity"]

        # Single queries
        single_results = [retriever.query(text, n_results=2) for text in query_texts]

        # Batch query
        batch_results = retriever.query_batch(query_texts, n_results=2)

        # Results should match
        for i, (single_result, batch_result) in enumerate(
            zip(single_results, batch_results)
        ):
            assert (
                single_result["ids"] == batch_result["ids"]
            ), f"Query {i}: IDs differ between single and batch"
            assert (
                single_result["distances"] == batch_result["distances"]
            ), f"Query {i}: Distances differ between single and batch"


class TestFullPipelineE2E:
    """End-to-end tests for the full vector store pipeline."""

    @pytest.mark.integration
    def test_full_pipeline_index_query_results(
        self,
        tiny_sbert_model,
        temp_index_dir,
        sample_hpo_documents,
        sample_hpo_metadatas,
        sample_hpo_ids,
    ):
        """
        Full end-to-end test: build index → query → validate results.

        This test simulates the complete workflow:
        1. Build index from HPO terms
        2. Query with clinical text
        3. Verify relevant results are returned
        4. Validate result structure

        This is the most important integration test - it validates that
        all components work together correctly.
        """
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Step 1: Build index
        build_success = build_chromadb_index(
            documents=sample_hpo_documents,
            metadatas=sample_hpo_metadatas,
            ids=sample_hpo_ids,
            model=tiny_sbert_model,
            model_name=model_name,
            batch_size=10,
            recreate=True,
            index_dir=temp_index_dir,
        )
        assert build_success, "Index build failed"

        # Step 2: Create retriever
        retriever = DenseRetriever.from_model_name(
            model=tiny_sbert_model,
            model_name=model_name,
            index_dir=temp_index_dir,
        )
        assert retriever is not None, "Retriever creation failed"

        # Step 3: Query with clinical text
        clinical_text = "Patient exhibits abnormally small head circumference"
        results = retriever.query(clinical_text, n_results=3)

        # Step 4: Validate results structure
        assert "ids" in results, "Results missing 'ids' key"
        assert "distances" in results, "Results missing 'distances' key"
        assert "metadatas" in results, "Results missing 'metadatas' key"
        assert "similarities" in results, "Results missing 'similarities' key"

        # Step 5: Validate result correctness
        assert len(results["ids"]) > 0, "No results returned"
        assert len(results["ids"][0]) > 0, "No IDs in results"

        # Expected: "Microcephaly" should be top result for "small head circumference"
        top_id = results["ids"][0][0]
        assert top_id == "HP:0000252", f"Expected HP:0000252 (Microcephaly), got {top_id}"


class TestVectorStoreConfigIntegration:
    """Test VectorStoreConfig integration with actual ChromaDB operations."""

    @pytest.mark.integration
    def test_config_creates_valid_chromadb_client(self, temp_index_dir):
        """Test that VectorStoreConfig can create a valid ChromaDB client."""
        import chromadb

        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        config = VectorStoreConfig.for_chromadb(
            model_name=model_name,
            index_dir=temp_index_dir,
        )

        # Create client using config
        client = chromadb.PersistentClient(
            path=config.path,
            settings=config.to_chromadb_settings(),
        )

        # Validate client is functional
        assert client is not None
        collections = client.list_collections()
        assert isinstance(collections, list)  # Should return empty list

    @pytest.mark.integration
    def test_connect_to_chroma_uses_config_correctly(
        self,
        tiny_sbert_model,
        temp_index_dir,
        sample_hpo_documents,
        sample_hpo_metadatas,
        sample_hpo_ids,
    ):
        """
        Test that connect_to_chroma() uses VectorStoreConfig correctly.

        This validates the refactoring maintains backward compatibility
        while using the new configuration system.
        """
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Build index first
        build_chromadb_index(
            documents=sample_hpo_documents,
            metadatas=sample_hpo_metadatas,
            ids=sample_hpo_ids,
            model=tiny_sbert_model,
            model_name=model_name,
            batch_size=10,
            recreate=True,
            index_dir=temp_index_dir,
        )

        # Connect using connect_to_chroma
        collection_name = generate_collection_name(model_name)
        collection = connect_to_chroma(
            index_dir=str(temp_index_dir),
            collection_name=collection_name,
            model_name=model_name,
        )

        # Validate connection
        assert collection is not None, "Connection failed"
        assert collection.count() == len(
            sample_hpo_documents
        ), "Collection count mismatch"
