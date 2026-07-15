"""Tests for strict Chroma index build completion semantics."""

from types import SimpleNamespace

import pytest

from phentrieve.indexing import chromadb_indexer

pytestmark = pytest.mark.unit


class _Embeddings:
    def __init__(self, count: int) -> None:
        self.count = count

    def tolist(self) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in range(self.count)]


class _Model:
    def __init__(self, fail_encode: bool = False) -> None:
        self.fail_encode = fail_encode

    def parameters(self):
        return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])

    def encode(self, documents: list[str], device: str) -> _Embeddings:
        if self.fail_encode:
            raise RuntimeError("embedding failed")
        assert device == "cpu"
        return _Embeddings(len(documents))


class _Collection:
    def __init__(self, metadata: dict[str, object], persist_adds: bool = True) -> None:
        self.metadata = metadata
        self.persist_adds = persist_adds
        self._count = 0

    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
        ids: list[str],
    ) -> None:
        assert len(documents) == len(embeddings) == len(metadatas) == len(ids)
        if self.persist_adds:
            self._count += len(ids)

    def count(self) -> int:
        return self._count


class _Client:
    def __init__(self, persist_adds: bool = True) -> None:
        self.collection: _Collection | None = None
        self.persist_adds = persist_adds

    def get_collection(self, name: str) -> _Collection:
        if self.collection is None:
            raise KeyError(name)
        return self.collection

    def delete_collection(self, name: str) -> None:
        self.collection = None

    def create_collection(self, name: str, metadata: dict[str, object]) -> _Collection:
        self.collection = _Collection(metadata, persist_adds=self.persist_adds)
        return self.collection


@pytest.fixture
def documents() -> list[str]:
    return ["term one", "term two"]


@pytest.fixture
def metadatas() -> list[dict[str, object]]:
    return [{"hpo_id": "HP:0000001"}, {"hpo_id": "HP:0000118"}]


@pytest.fixture
def ids() -> list[str]:
    return ["HP:0000001", "HP:0000118"]


def _install_client(monkeypatch, persist_adds: bool = True) -> _Client:
    client = _Client(persist_adds=persist_adds)
    monkeypatch.setattr(
        chromadb_indexer.chromadb, "PersistentClient", lambda **_: client
    )
    return client


def test_build_writes_pinned_collection_metadata(
    monkeypatch, tmp_path, documents, metadatas, ids
):
    client = _install_client(monkeypatch)

    success = chromadb_indexer.build_chromadb_index(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        model=_Model(),
        model_name="unit/model",
        batch_size=1,
        recreate=True,
        index_dir=tmp_path,
        index_type="single_vector",
        hpo_version="v2026-06-23",
        hpo_source_sha256="a" * 64,
        model_revision="b" * 40,
    )

    assert success
    assert client.collection is not None
    assert client.collection.metadata["hpo_version"] == "v2026-06-23"
    assert client.collection.metadata["hpo_source_sha256"] == "a" * 64
    assert client.collection.metadata["model"] == "unit/model"
    assert client.collection.metadata["model_revision"] == "b" * 40
    assert client.collection.metadata["index_type"] == "single_vector"
    assert client.collection.metadata["dimension"] == 768
    assert client.collection.metadata["expected_document_count"] == len(documents)


def test_build_fails_immediately_when_embedding_fails(
    monkeypatch, tmp_path, documents, metadatas, ids
):
    client = _install_client(monkeypatch)

    success = chromadb_indexer.build_chromadb_index(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        model=_Model(fail_encode=True),
        model_name="unit/model",
        recreate=True,
        index_dir=tmp_path,
        hpo_version="v2026-06-23",
    )

    assert not success
    assert client.collection is not None
    assert client.collection.count() == 0


def test_build_rejects_a_persisted_count_mismatch(
    monkeypatch, tmp_path, documents, metadatas, ids
):
    _install_client(monkeypatch, persist_adds=False)

    success = chromadb_indexer.build_chromadb_index(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        model=_Model(),
        model_name="unit/model",
        recreate=True,
        index_dir=tmp_path,
        hpo_version="v2026-06-23",
    )

    assert not success


def test_build_rejects_an_unpinned_hpo_version(
    monkeypatch, tmp_path, documents, metadatas, ids
):
    _install_client(monkeypatch)

    success = chromadb_indexer.build_chromadb_index(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        model=_Model(),
        model_name="unit/model",
        recreate=True,
        index_dir=tmp_path,
    )

    assert not success
