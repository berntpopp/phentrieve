"""End-to-end smoke test for scripts/analyze_embedding_ontology.py.

Runs the orchestrator against the real HPO SQLite DB but mocks the embedding
cache (so ChromaDB is never touched and UMAP operates on tiny synthetic vectors).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "analyze_embedding_ontology.py"


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _write_stub_cache(tmp_path: Path, term_ids: list[str]) -> Path:
    """Write a fake ontology-fidelity cache with deterministic synthetic vectors."""
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((len(term_ids), 16)).astype(np.float32)

    # Mirrors the real cache layout.
    from phentrieve.utils import generate_collection_name

    collection = generate_collection_name("FremyCompany/BioLORD-2023-M")
    cache_dir = tmp_path / "indexes" / "ontology_fidelity_cache" / collection
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", vecs)
    (cache_dir / "hpo_ids.json").write_text(json.dumps(term_ids))
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": "FremyCompany/BioLORD-2023-M",
                "collection_name": collection,
                "written_at": "2026-04-17T00:00:00+00:00",
                "n_terms": len(term_ids),
                "dim": 16,
            }
        )
    )
    return cache_dir


def test_analyze_script_smoke(tmp_path, monkeypatch):
    """Run the script against a real HPO DB with a stubbed embedding cache."""
    from phentrieve.data_processing.hpo_database import HPODatabase
    from phentrieve.utils import get_default_data_dir, resolve_data_path

    data_dir = resolve_data_path(None, "data_dir", get_default_data_dir)
    db_path = data_dir / "hpo_data.db"
    if not db_path.exists():
        pytest.skip(f"HPO SQLite DB not present at {db_path}")

    db = HPODatabase(db_path)
    try:
        all_terms = db.load_all_terms()
    finally:
        db.close()
    # Use all terms so the divergence check passes; --sample 150 limits computation.
    term_ids = [t["id"] for t in all_terms]

    # Point PHENTRIEVE_INDEX_DIR at a temp tree and write a stubbed cache there.
    tmp_index_root = tmp_path
    _write_stub_cache(tmp_index_root, term_ids)
    monkeypatch.setenv("PHENTRIEVE_INDEX_DIR", str(tmp_index_root / "indexes"))

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--sample",
        "150",
        "--n-pairs",
        "300",
        "--k",
        "5",
        "--umap-neighbors",
        "10",
        "--output-dir",
        str(out_dir),
        "--log-level",
        "INFO",
        "--skip-interactive",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    # Exactly one run subdir.
    run_dirs = list(out_dir.iterdir())
    assert len(run_dirs) == 1
    run = run_dirs[0]

    expected = [
        "summary.json",
        "per_term_fidelity.csv",
        "umap_coords.csv",
        "umap_by_branch.png",
        "umap_by_fidelity.png",
        "distance_correlation.png",
        "branch_fidelity.png",
    ]
    for name in expected:
        assert (run / name).exists(), f"missing {name}; dir has {list(run.iterdir())}"

    summary = json.loads((run / "summary.json").read_text())
    assert "metrics" in summary
    assert "config" in summary
    assert summary["config"]["model_name"] == "FremyCompany/BioLORD-2023-M"
