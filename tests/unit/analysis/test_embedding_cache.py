"""Unit tests for phentrieve.analysis.embedding_cache."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_client_with_collection(ids, embeddings) -> MagicMock:
    """Return a mock chromadb.PersistentClient that exposes a collection
    whose .get(include=['embeddings']) returns the given ids/embeddings."""
    client = MagicMock()
    collection = MagicMock()
    collection.get.return_value = {
        "ids": list(ids),
        "embeddings": [list(row) for row in embeddings],
    }
    client.get_collection.return_value = collection
    return client


def test_embedding_cache_first_call_reads_chroma_and_writes_files(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    ids = ["HP:0000001", "HP:0000002", "HP:0000003"]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    client = _make_mock_client_with_collection(ids, vecs)

    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=client,
    ):
        got_ids, got_vecs = load_cached_embeddings(
            model_name="FremyCompany/BioLORD-2023-M",
            index_dir_override=str(tmp_path),
        )

    assert got_ids == ids
    np.testing.assert_allclose(got_vecs, vecs)

    cache_root = tmp_path / "ontology_fidelity_cache"
    subdirs = list(cache_root.iterdir())
    assert len(subdirs) == 1
    cache_dir = subdirs[0]
    assert (cache_dir / "embeddings.npy").exists()
    assert (cache_dir / "hpo_ids.json").exists()
    assert (cache_dir / "meta.json").exists()
    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["model_name"] == "FremyCompany/BioLORD-2023-M"
    assert meta["n_terms"] == 3
    assert meta["dim"] == 2


def test_embedding_cache_uses_telemetry_disabled_chromadb_settings(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    ids = ["HP:0000001"]
    vecs = np.array([[1.0, 0.0]], dtype=np.float32)
    client = _make_mock_client_with_collection(ids, vecs)

    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=client,
    ) as mock_client_class:
        load_cached_embeddings(
            model_name="FremyCompany/BioLORD-2023-M",
            index_dir_override=str(tmp_path),
        )

    settings = mock_client_class.call_args.kwargs["settings"]
    assert settings.anonymized_telemetry is False
    assert settings.is_persistent is True


def test_embedding_cache_second_call_skips_chroma(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    ids = ["HP:0000001", "HP:0000002"]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    client = _make_mock_client_with_collection(ids, vecs)

    # Warm the cache.
    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=client,
    ):
        load_cached_embeddings(
            "FremyCompany/BioLORD-2023-M", index_dir_override=str(tmp_path)
        )

    # Second call must not construct PersistentClient at all.
    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        side_effect=AssertionError("should not be called"),
    ):
        got_ids, got_vecs = load_cached_embeddings(
            "FremyCompany/BioLORD-2023-M", index_dir_override=str(tmp_path)
        )

    assert got_ids == ids
    np.testing.assert_allclose(got_vecs, vecs)


def test_embedding_cache_refresh_overwrites(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    ids_v1 = ["HP:0000001", "HP:0000002"]
    vecs_v1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids_v2 = ["HP:0000001", "HP:0000002", "HP:0000003"]
    vecs_v2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=_make_mock_client_with_collection(ids_v1, vecs_v1),
    ):
        load_cached_embeddings(
            "FremyCompany/BioLORD-2023-M", index_dir_override=str(tmp_path)
        )

    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=_make_mock_client_with_collection(ids_v2, vecs_v2),
    ):
        got_ids, got_vecs = load_cached_embeddings(
            "FremyCompany/BioLORD-2023-M",
            refresh=True,
            index_dir_override=str(tmp_path),
        )

    assert got_ids == ids_v2
    assert got_vecs.shape == (3, 2)


def test_embedding_cache_missing_collection_raises(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    client = MagicMock()
    client.get_collection.side_effect = ValueError("not found")
    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=client,
    ):
        with pytest.raises(FileNotFoundError):
            load_cached_embeddings(
                "FremyCompany/BioLORD-2023-M", index_dir_override=str(tmp_path)
            )
