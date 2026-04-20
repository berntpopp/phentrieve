"""Shared dataset loading helpers for benchmark workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ASSERTION_STATUS_MAP: dict[str, str] = {
    "affirmed": "PRESENT",
    "negated": "ABSENT",
    "uncertain": "UNCERTAIN",
    "normal": "PRESENT",
    "family_history": "FAMILY_HISTORY",
}

LLM_ASSERTION_TO_BENCHMARK: dict[str, str] = {
    "present": "PRESENT",
    "affirmed": "PRESENT",
    "absent": "ABSENT",
    "negated": "ABSENT",
    "uncertain": "UNCERTAIN",
}

PHENOBERT_DATASETS: tuple[str, ...] = ("GSC_plus", "ID_68", "GeneReviews")
RAG_HPO_PAPER_DATASETS: tuple[str, ...] = ("CSC", "GSC")
DIRECTORY_BENCHMARK_DATASETS: tuple[str, ...] = (
    "GSC_plus",
    "ID_68",
    "GeneReviews",
    "CSC",
    "GSC",
)
DEFAULT_PHENOBERT_DATASET = "all"
DEFAULT_SIMPLE_ASSERTION = "PRESENT"
BENCHMARK_TEXT_KEYS: tuple[str, ...] = ("text", "input_text")

DATASET_GOLD_ASSERTION_PROJECTION: dict[str, dict[str, str | None]] = {
    "CSC": {
        "PRESENT": "PRESENT",
        "ABSENT": None,
        "UNCERTAIN": None,
        "FAMILY_HISTORY": None,
    },
    "GSC": {
        "PRESENT": "PRESENT",
        "ABSENT": None,
        "UNCERTAIN": None,
        "FAMILY_HISTORY": None,
    },
}


def load_benchmark_data(
    test_path: Path,
    dataset: str = DEFAULT_PHENOBERT_DATASET,
) -> dict[str, Any]:
    """Load benchmark data from a PhenoBERT directory or simple JSON list file."""
    if test_path.is_dir():
        return load_phenobert_data(test_path, dataset=dataset)
    return load_json_benchmark_data(test_path)


def load_phenobert_data(
    base_dir: Path,
    dataset: str = DEFAULT_PHENOBERT_DATASET,
) -> dict[str, Any]:
    """Load converted PhenoBERT benchmark documents from a directory."""
    dataset_dirs = {
        name: base_dir / name / "annotations" for name in DIRECTORY_BENCHMARK_DATASETS
    }

    if dataset != DEFAULT_PHENOBERT_DATASET:
        if dataset not in dataset_dirs:
            raise ValueError(
                "Unknown dataset: "
                f"{dataset}. Available: {list(DIRECTORY_BENCHMARK_DATASETS)}"
            )
        dataset_dirs = {dataset: dataset_dirs[dataset]}

    documents: list[dict[str, Any]] = []
    total_annotations = 0

    for dataset_name, annotations_dir in dataset_dirs.items():
        if not annotations_dir.exists():
            logger.warning("Dataset directory not found: %s", annotations_dir)
            continue

        for json_file in sorted(annotations_dir.glob("*.json")):
            doc_data = json.loads(json_file.read_text(encoding="utf-8"))
            gold_hpo_terms = _convert_phenobert_annotations(
                doc_data.get("annotations", []),
                dataset_name=dataset_name,
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

    if not documents:
        raise ValueError(
            "No benchmark documents found in PhenoBERT directory "
            f"{base_dir} for dataset {dataset!r}."
        )

    return {
        "metadata": {
            "dataset_name": _directory_dataset_name(dataset),
            "source": _directory_dataset_source(dataset),
            "dataset_namespace": _directory_dataset_source(dataset),
            "total_documents": len(documents),
            "total_annotations": total_annotations,
        },
        "documents": documents,
    }


def load_json_benchmark_data(test_file: Path) -> dict[str, Any]:
    """Load benchmark data from a JSON file."""
    data = json.loads(test_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return _validate_document_benchmark_payload(data, source_name=test_file.stem)
    if not isinstance(data, list):
        raise ValueError("LLM benchmark datasets must be JSON lists of test cases.")

    documents: list[dict[str, Any]] = []
    total_annotations = 0

    for index, raw_case in enumerate(data, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Invalid benchmark case at index {index - 1}.")
        text = _get_case_text(raw_case)
        gold_hpo_terms = _convert_simple_case_annotations(raw_case)
        documents.append(
            {
                "id": str(raw_case.get("description") or f"case_{index}"),
                "text": text,
                "gold_hpo_terms": gold_hpo_terms,
                "source_dataset": "json_list",
            }
        )
        total_annotations += len(gold_hpo_terms)

    return {
        "metadata": {
            "dataset_name": test_file.stem,
            "source": "json_list",
            "total_documents": len(documents),
            "total_annotations": total_annotations,
        },
        "documents": documents,
    }


def _validate_document_benchmark_payload(
    payload: dict[str, Any],
    *,
    source_name: str,
) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Benchmark JSON objects must contain a 'documents' list.")
    if not documents:
        raise ValueError(
            f"No benchmark documents found in JSON dataset {source_name!r}."
        )

    return {
        "metadata": {
            "dataset_name": metadata.get("dataset_name", source_name),
            "source": metadata.get("source", "json_documents"),
            "total_documents": metadata.get("total_documents", len(documents)),
            "total_annotations": metadata.get("total_annotations", 0),
        },
        "documents": documents,
    }


def parse_gold_terms(gold_hpo_terms: list[Any]) -> list[tuple[str, str]]:
    """Parse gold HPO terms into ``(hpo_id, assertion)`` tuples."""
    result: list[tuple[str, str]] = []
    for term in gold_hpo_terms:
        if isinstance(term, dict):
            hpo_id = str(term.get("id") or term.get("hpo_id") or "")
            assertion = str(term.get("assertion") or DEFAULT_SIMPLE_ASSERTION)
        elif isinstance(term, (list, tuple)) and len(term) >= 2:
            hpo_id, assertion = str(term[0]), str(term[1])
        else:
            hpo_id = str(term)
            assertion = DEFAULT_SIMPLE_ASSERTION
        result.append((hpo_id, assertion))
    return result


def _convert_phenobert_annotations(
    annotations: list[dict[str, Any]],
    *,
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    gold_terms: list[dict[str, Any]] = []
    projection = None
    if dataset_name is not None:
        projection = DATASET_GOLD_ASSERTION_PROJECTION.get(dataset_name)
    for annotation in annotations:
        assertion = ASSERTION_STATUS_MAP.get(
            str(annotation.get("assertion_status", "affirmed")),
            DEFAULT_SIMPLE_ASSERTION,
        )
        if projection is not None:
            projected_assertion = projection.get(assertion)
            if projected_assertion is None:
                continue
            assertion = projected_assertion
        gold_terms.append(
            {
                "id": annotation.get("hpo_id", ""),
                "label": annotation.get("label", ""),
                "assertion": assertion,
                "evidence_spans": annotation.get("evidence_spans", []),
            }
        )
    return gold_terms


def _convert_simple_case_annotations(case: dict[str, Any]) -> list[dict[str, Any]]:
    expected_ids = case.get("expected_hpo_ids")
    if not isinstance(expected_ids, list):
        return []
    return [
        {"id": item.strip(), "assertion": DEFAULT_SIMPLE_ASSERTION}
        for item in expected_ids
        if isinstance(item, str) and item.strip()
    ]


def _get_case_text(case: dict[str, Any]) -> str:
    for key in BENCHMARK_TEXT_KEYS:
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Each benchmark case must provide non-empty text.")


def _directory_dataset_source(dataset: str) -> str:
    if dataset in RAG_HPO_PAPER_DATASETS:
        return "rag_hpo_paper"
    return "phenobert"


def _directory_dataset_name(dataset: str) -> str:
    return f"phenobert_{dataset}"
