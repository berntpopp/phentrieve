"""Integration test fixtures (real dependencies)."""


import pytest


@pytest.fixture(scope="module")
def real_embedding_model():
    """Real embedding model (cached at module scope)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def real_chromadb_collection(tmp_path_factory):
    """Real ChromaDB collection (test isolation)."""
    import chromadb

    persist_dir = tmp_path_factory.mktemp("chromadb")
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection("test_hpo_terms")

    yield collection

    # Cleanup
    client.delete_collection("test_hpo_terms")


@pytest.fixture
def temp_test_dir(tmp_path):
    """Temporary directory for test file operations."""
    return tmp_path
