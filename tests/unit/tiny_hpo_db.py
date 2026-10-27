"""Tiny prepared HPO database for similarity tests that need real graph data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.evaluation.metrics import load_hpo_graph_data

# term_id -> (label, depth, ancestors including the term itself)
TINY_HPO_TERMS: dict[str, tuple[str, int, list[str]]] = {
    "HP:0000001": ("All", 0, ["HP:0000001"]),
    "HP:0000118": ("Phenotypic abnormality", 1, ["HP:0000118", "HP:0000001"]),
    "HP:0000707": (
        "Abnormality of the nervous system",
        2,
        ["HP:0000707", "HP:0000118", "HP:0000001"],
    ),
    "HP:0001250": (
        "Seizure",
        3,
        ["HP:0001250", "HP:0000707", "HP:0000118", "HP:0000001"],
    ),
    "HP:0002133": (
        "Status epilepticus",
        3,
        ["HP:0002133", "HP:0000707", "HP:0000118", "HP:0000001"],
    ),
}


def write_tiny_hpo_db(data_dir: Path) -> Path:
    """Write a prepared ``hpo_data.db`` with ``TINY_HPO_TERMS`` into data_dir."""
    db_path = data_dir / DEFAULT_HPO_DB_FILENAME
    with HPODatabase(db_path) as db:
        db.initialize_schema()
        db.bulk_insert_terms(
            [
                {
                    "id": term_id,
                    "label": label,
                    "definition": "",
                    "synonyms": "[]",
                    "comments": "[]",
                }
                for term_id, (label, _depth, _anc) in TINY_HPO_TERMS.items()
            ]
        )
        db.bulk_insert_graph_metadata(
            [
                {"term_id": term_id, "depth": depth, "ancestors": json.dumps(anc)}
                for term_id, (_label, depth, anc) in TINY_HPO_TERMS.items()
            ]
        )
    return db_path


def use_hpo_data_dir(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    """Point HPO data resolution at data_dir (initially empty) with a cold cache."""
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PHENTRIEVE_DATA_DIR", str(data_dir))
    # _resolve_hpo_db_path prefers ./data/hpo_data.db; run from data_dir so the
    # repository's prepared data (if any) is not picked up.
    monkeypatch.chdir(data_dir)
    load_hpo_graph_data.cache_clear()  # type: ignore[attr-defined]
