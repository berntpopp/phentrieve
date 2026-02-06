"""Shared data loading utilities for PhenoBERT-format benchmark datasets.

This module provides functions for loading PhenoBERT-format gold-standard
data from directory structures, as well as parsing gold HPO terms into
evaluation-compatible formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mapping from PhenoBERT/internal assertion status to benchmark evaluation format.
# Shared between extraction benchmark and LLM benchmark.
ASSERTION_STATUS_MAP: dict[str, str] = {
    "affirmed": "PRESENT",
    "negated": "ABSENT",
    "uncertain": "UNCERTAIN",
    "normal": "PRESENT",  # normal findings are present
}

# Mapping from LLM AssertionStatus enum values to benchmark evaluation format.
LLM_ASSERTION_TO_BENCHMARK: dict[str, str] = {
    "affirmed": "PRESENT",
    "negated": "ABSENT",
    "uncertain": "UNCERTAIN",
}

# Known PhenoBERT dataset subdirectories
PHENOBERT_DATASETS: list[str] = ["GSC_plus", "ID_68", "GeneReviews"]


def load_phenobert_data(
    base_dir: Path,
    dataset: str = "all",
) -> dict[str, Any]:
    """Load PhenoBERT-format data from directory structure.

    Expected structure::

        base_dir/
            GSC_plus/annotations/*.json
            ID_68/annotations/*.json
            GeneReviews/annotations/*.json

    Each JSON file follows the PhenoBERT annotation schema with fields:
    ``doc_id``, ``full_text``, ``annotations`` (list of HPO annotations
    with ``hpo_id``, ``label``, ``assertion_status``, ``evidence_spans``).

    Args:
        base_dir: Root directory containing dataset subdirectories.
        dataset: Which dataset(s) to load. One of ``"all"``,
            ``"GSC_plus"``, ``"ID_68"``, or ``"GeneReviews"``.

    Returns:
        Dict with ``metadata`` and ``documents`` keys. Each document has
        ``id``, ``text``, ``gold_hpo_terms``, and ``source_dataset``.

    Raises:
        ValueError: If the specified dataset name is not recognized.
    """
    dataset_dirs = {
        name: base_dir / name / "annotations" for name in PHENOBERT_DATASETS
    }

    if dataset != "all":
        if dataset not in dataset_dirs:
            raise ValueError(
                f"Unknown dataset: {dataset}. Available: {PHENOBERT_DATASETS}"
            )
        dataset_dirs = {dataset: dataset_dirs[dataset]}

    documents: list[dict[str, Any]] = []
    total_annotations = 0

    for dataset_name, annotations_dir in dataset_dirs.items():
        if not annotations_dir.exists():
            logger.warning(f"Dataset directory not found: {annotations_dir}")
            continue

        for json_file in sorted(annotations_dir.glob("*.json")):
            with open(json_file) as f:
                doc_data = json.load(f)

            gold_hpo_terms = _convert_phenobert_annotations(
                doc_data.get("annotations", [])
            )

            documents.append(
                {
                    "id": doc_data.get("doc_id", json_file.stem),
                    "text": doc_data.get("full_text", ""),
                    "gold_hpo_terms": gold_hpo_terms,
                    "source_dataset": dataset_name,
                }
            )
            total_annotations += len(gold_hpo_terms)

    logger.info(
        f"Loaded {len(documents)} documents with {total_annotations} annotations "
        f"from PhenoBERT data"
    )

    return {
        "metadata": {
            "dataset_name": f"phenobert_{dataset}",
            "source": "phenobert",
            "total_documents": len(documents),
            "total_annotations": total_annotations,
        },
        "documents": documents,
    }


def _convert_phenobert_annotations(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert PhenoBERT annotation list to benchmark gold format.

    Args:
        annotations: List of PhenoBERT annotation dicts.

    Returns:
        List of gold term dicts with ``id``, ``label``, ``assertion``,
        and ``evidence_spans``.
    """
    gold_terms: list[dict[str, Any]] = []
    for ann in annotations:
        hpo_id = ann.get("hpo_id", "")
        label = ann.get("label", "")
        status = ann.get("assertion_status", "affirmed")
        assertion = ASSERTION_STATUS_MAP.get(status, "PRESENT")
        evidence_spans = ann.get("evidence_spans", [])
        gold_terms.append(
            {
                "id": hpo_id,
                "label": label,
                "assertion": assertion,
                "evidence_spans": evidence_spans,
            }
        )
    return gold_terms


def parse_gold_terms(gold_hpo_terms: list[Any]) -> list[tuple[str, str]]:
    """Parse gold HPO terms into (hpo_id, assertion) tuples.

    Handles multiple input formats:
    - Dict with ``id``/``hpo_id`` and ``assertion`` keys
    - Tuple/list of ``(hpo_id, assertion)``
    - Plain string HPO ID (defaults to ``"PRESENT"``)

    Args:
        gold_hpo_terms: List of gold terms in any supported format.

    Returns:
        List of ``(hpo_id, assertion)`` tuples.
    """
    result: list[tuple[str, str]] = []
    for term in gold_hpo_terms:
        if isinstance(term, dict):
            hpo_id = str(term.get("id") or term.get("hpo_id") or "")
            assertion = str(term.get("assertion") or "PRESENT")
        elif isinstance(term, (list, tuple)) and len(term) >= 2:
            hpo_id, assertion = str(term[0]), str(term[1])
        else:
            hpo_id = str(term)
            assertion = "PRESENT"
        result.append((hpo_id, assertion))
    return result
