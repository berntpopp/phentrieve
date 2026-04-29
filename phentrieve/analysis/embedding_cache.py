"""Read HPO embeddings once from ChromaDB, then serve from a local `.npy` cache.

Cache layout:
    <index_dir>/ontology_fidelity_cache/<collection_name>/
        embeddings.npy    # (N, D) float32
        hpo_ids.json      # ["HP:0000001", ...] aligned with embeddings rows
        meta.json         # {"model_name", "collection_name", "written_at",
                          #  "n_terms", "dim"}

index_dir defaults to phentrieve.utils.get_default_index_dir(), override via
`index_dir_override`. collection_name comes from
phentrieve.utils.generate_collection_name(model_name).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import chromadb
import numpy as np

from phentrieve.utils import generate_collection_name, get_default_index_dir

logger = logging.getLogger(__name__)


def _cache_dir_for(index_dir: Path, collection_name: str) -> Path:
    return index_dir / "ontology_fidelity_cache" / collection_name


def _read_cache(cache_dir: Path) -> tuple[list[str], np.ndarray] | None:
    """Return (ids, embeddings) if the cache is valid, else None.

    Any missing/malformed file yields None (caller will refresh from Chroma).
    """
    emb_path = cache_dir / "embeddings.npy"
    ids_path = cache_dir / "hpo_ids.json"
    meta_path = cache_dir / "meta.json"
    if not (emb_path.exists() and ids_path.exists() and meta_path.exists()):
        return None
    try:
        embeddings = np.load(emb_path)
        ids = json.loads(ids_path.read_text())
        meta = json.loads(meta_path.read_text())
    except (ValueError, OSError, json.JSONDecodeError) as e:
        logger.error("Ontology-fidelity cache unreadable at %s: %s", cache_dir, e)
        return None
    if not isinstance(ids, list) or len(ids) != embeddings.shape[0]:
        logger.error(
            "Ontology-fidelity cache mismatch: %d ids vs %d embedding rows",
            len(ids) if isinstance(ids, list) else -1,
            embeddings.shape[0],
        )
        return None
    if meta.get("n_terms") != embeddings.shape[0]:
        logger.error("Ontology-fidelity cache meta.n_terms mismatch")
        return None
    return list(ids), embeddings


def _write_cache(
    cache_dir: Path,
    ids: list[str],
    embeddings: np.ndarray,
    model_name: str,
    collection_name: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings.astype(np.float32))
    (cache_dir / "hpo_ids.json").write_text(json.dumps(list(ids)))
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "collection_name": collection_name,
                "written_at": datetime.now(UTC).isoformat(),
                "n_terms": int(embeddings.shape[0]),
                "dim": int(embeddings.shape[1]),
            }
        )
    )


def _clear_cache(cache_dir: Path) -> None:
    for fname in ("embeddings.npy", "hpo_ids.json", "meta.json"):
        p = cache_dir / fname
        if p.exists():
            p.unlink()


def _read_from_chroma(
    index_dir: Path, collection_name: str
) -> tuple[list[str], np.ndarray]:
    try:
        client = chromadb.PersistentClient(path=str(index_dir))
        collection = client.get_collection(collection_name)
    except Exception as e:  # chromadb raises a variety of types
        raise FileNotFoundError(
            f"ChromaDB collection {collection_name!r} not found in {index_dir}. "
            "Run 'phentrieve index build --model-name ...' first."
        ) from e

    got = collection.get(include=["embeddings"])
    ids = list(got["ids"])
    embeddings = np.asarray(got["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError(
            f"Malformed Chroma payload: {len(ids)} ids vs embeddings shape "
            f"{embeddings.shape}"
        )
    return ids, embeddings


def load_cached_embeddings(
    model_name: str,
    refresh: bool = False,
    index_dir_override: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return (hpo_ids, embeddings) for the given model.

    First call: queries ChromaDB, writes cache, returns arrays.
    Later calls: reads cache only. `refresh=True` forces a re-read.

    Raises FileNotFoundError if the ChromaDB collection for `model_name`
    does not exist.
    """
    index_dir = (
        Path(index_dir_override) if index_dir_override else get_default_index_dir()
    )
    collection_name = generate_collection_name(model_name)
    cache_dir = _cache_dir_for(index_dir, collection_name)

    if not refresh:
        cached = _read_cache(cache_dir)
        if cached is not None:
            logger.info("Loaded ontology-fidelity cache from %s", cache_dir)
            return cached
        if any(
            (cache_dir / f).exists()
            for f in ("embeddings.npy", "hpo_ids.json", "meta.json")
        ):
            logger.warning(
                "Ontology-fidelity cache at %s is partial/malformed; refreshing from Chroma",
                cache_dir,
            )

    _clear_cache(cache_dir)
    ids, embeddings = _read_from_chroma(index_dir, collection_name)
    _write_cache(cache_dir, ids, embeddings, model_name, collection_name)
    logger.info(
        "Wrote ontology-fidelity cache: %d terms x %d dims -> %s",
        len(ids),
        embeddings.shape[1],
        cache_dir,
    )
    return ids, embeddings
