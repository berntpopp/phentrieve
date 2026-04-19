"""Convert released RAG-HPO paper benchmark data into Phentrieve JSON fixtures."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from shared_utils import ProvenanceTracker

from phentrieve.text_processing.spans import find_span_in_text

CONVERTER_VERSION = "0.1.0"

SOURCE_REPOSITORY_URL = "https://github.com/PoseyPod/RAG-HPO"
DATASET_OUTPUT_NAMES = ("CSC", "GSC")

ASSERTION_STATUS_MAP = {
    "abnormal": "affirmed",
    "affirmed": "affirmed",
    "present": "affirmed",
    "positive": "affirmed",
    "negative": "negated",
    "negated": "negated",
    "absent": "negated",
    "excluded": "negated",
    "family_history": "family_history",
    "family history": "family_history",
    "uncertain": "uncertain",
    "suspected": "uncertain",
    "possible": "uncertain",
}

HPO_ID_PATTERN = re.compile(r"HP:\d{7}")


@dataclass(slots=True)
class ConversionStats:
    """Track conversion output and non-fatal issues."""

    total_documents: int = 0
    total_annotations: int = 0
    warnings: list[str] = field(default_factory=list)
    datasets: dict[str, dict[str, int]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def add_document(self, dataset: str, num_annotations: int) -> None:
        self.total_documents += 1
        self.total_annotations += num_annotations
        dataset_stats = self.datasets.setdefault(
            dataset,
            {"documents": 0, "annotations": 0},
        )
        dataset_stats["documents"] += 1
        dataset_stats["annotations"] += num_annotations

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "summary": {
                "total_documents": self.total_documents,
                "total_annotations": self.total_annotations,
                "warnings": len(self.warnings),
            },
            "datasets": self.datasets,
            "warnings": self.warnings,
        }


class HPOTermLookup:
    """Resolve HPO labels from a TSV export."""

    def __init__(self, hpo_terms_path: Path, hpo_json_path: Path | None = None) -> None:
        self._labels: dict[str, str] = {}
        self._replacements: dict[str, str] = {}
        self._deprecated_ids: set[str] = set()
        with hpo_terms_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                term_id = str(row.get("id", "")).strip()
                label = str(row.get("name", "")).strip()
                if term_id:
                    self._labels[term_id] = label or term_id
        if hpo_json_path is not None:
            self._load_replacements(hpo_json_path)

    def resolve(self, hpo_id: str) -> str:
        return self._labels.get(hpo_id, hpo_id)

    def replacement_for(self, hpo_id: str) -> str | None:
        seen: set[str] = set()
        current = hpo_id
        replacement = self._replacements.get(current)
        while replacement is not None and replacement not in seen:
            seen.add(current)
            current = replacement
            if current not in self._deprecated_ids:
                return current
            replacement = self._replacements.get(current)
        if current != hpo_id and current not in self._deprecated_ids:
            return current
        return None

    def is_deprecated(self, hpo_id: str) -> bool:
        return hpo_id in self._deprecated_ids

    def _load_replacements(self, hpo_json_path: Path) -> None:
        ontology = json.loads(hpo_json_path.read_text(encoding="utf-8"))
        for node in ontology.get("graphs", [{}])[0].get("nodes", []):
            node_id = node.get("id")
            if not (isinstance(node_id, str) and "/HP_" in node_id):
                continue
            current_id = "HP:" + node_id.rsplit("HP_", 1)[1]
            self._labels.setdefault(current_id, str(node.get("lbl") or current_id))
            label = str(node.get("lbl") or current_id)
            if bool(node.get("meta", {}).get("deprecated")) or label.startswith(
                "obsolete "
            ):
                self._deprecated_ids.add(current_id)
            for property_value in node.get("meta", {}).get("basicPropertyValues", []):
                if (
                    property_value.get("pred")
                    == "http://purl.obolibrary.org/obo/IAO_0100001"
                ):
                    replacement_id = str(property_value.get("val", "")).strip()
                    if replacement_id.startswith("HP:"):
                        self._replacements[current_id] = replacement_id


class RagHpoPaperConverter:
    """Convert released CSC/GSC paper data into benchmark fixture JSON."""

    def __init__(
        self,
        hpo_terms_path: Path,
        *,
        hpo_json_path: Path | None = None,
        normalize_obsolete_ids: bool = False,
        drop_obsolete_without_replacement: bool = False,
    ) -> None:
        self._lookup = HPOTermLookup(hpo_terms_path, hpo_json_path=hpo_json_path)
        self._normalize_obsolete_ids = normalize_obsolete_ids
        self._drop_obsolete_without_replacement = drop_obsolete_without_replacement

    def convert(
        self,
        workbook_path: Path,
        test_cases_csv_path: Path,
        output_root: Path,
        dataset: str = "all",
    ) -> dict[str, Any]:
        selected_datasets = self._select_datasets(dataset)
        output_root.mkdir(parents=True, exist_ok=True)

        stats = ConversionStats()
        stats.provenance = {
            "converter_version": CONVERTER_VERSION,
            "conversion_date": _iso_now(),
            "source_repository": SOURCE_REPOSITORY_URL,
            "source_version": ProvenanceTracker.get_git_version(workbook_path.parent),
            "inputs": {
                "workbook": str(workbook_path),
                "test_cases_csv": str(test_cases_csv_path),
            },
        }

        workbook = pd.ExcelFile(workbook_path)
        csc_cases = self._load_csc_cases(test_cases_csv_path)
        csc_annotations = self._load_csc_annotations(workbook)
        csc_cases = self._filter_cases_to_annotations(
            dataset_name="CSC",
            cases=csc_cases,
            annotations_by_case=csc_annotations,
            stats=stats,
        )
        gsc_cases = self._load_gsc_cases(workbook)
        gsc_annotations = self._load_gsc_annotations(workbook)

        for dataset_name in selected_datasets:
            if dataset_name == "CSC":
                documents = self._build_documents(
                    dataset_name="CSC",
                    cases=csc_cases,
                    annotations_by_case=csc_annotations,
                    stats=stats,
                )
            else:
                documents = self._build_documents(
                    dataset_name="GSC",
                    cases=gsc_cases,
                    annotations_by_case=gsc_annotations,
                    stats=stats,
                )

            self._write_documents(output_root, dataset_name, documents)
            for payload in documents:
                stats.add_document(dataset_name, len(payload["annotations"]))

        report = stats.to_dict()
        report_path = output_root / "conversion_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report

    def _select_datasets(self, dataset: str) -> tuple[str, ...]:
        if dataset == "all":
            return DATASET_OUTPUT_NAMES
        if dataset not in DATASET_OUTPUT_NAMES:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. Expected one of: {DATASET_OUTPUT_NAMES}"
            )
        return (dataset,)

    def _load_csc_cases(self, csv_path: Path) -> dict[str, dict[str, str]]:
        cases: dict[str, dict[str, str]] = {}
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_id = _cell_to_str(row.get("Case", ""))
                if not case_id:
                    continue
                cases[case_id] = {
                    "patient_id": case_id,
                    "original_doc_id": case_id,
                    "full_text": _cell_to_str(row.get("clinical_note", "")),
                }
        return cases

    def _load_gsc_cases(self, workbook: pd.ExcelFile) -> dict[str, dict[str, str]]:
        frame = _parse_sheet(workbook, "GSC Input").fillna("")
        cases: dict[str, dict[str, str]] = {}
        for row in frame.to_dict("records"):
            patient_id = _cell_to_str(row.get("patient_id", ""))
            if not patient_id:
                continue
            original_doc_id = _cell_to_str(row.get("ID", "")) or patient_id
            cases[patient_id] = {
                "patient_id": patient_id,
                "original_doc_id": original_doc_id,
                "full_text": _cell_to_str(row.get("clinical_note", "")),
            }
        return cases

    def _load_csc_annotations(
        self, workbook: pd.ExcelFile
    ) -> dict[str, list[dict[str, str]]]:
        frame = _parse_sheet(workbook, "CSC Manual Annotations").fillna("")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in frame.to_dict("records"):
            patient_id = _cell_to_str(row.get("Patient ID", ""))
            if not patient_id:
                continue
            grouped[patient_id].append(
                {
                    "hpo_description": _cell_to_str(row.get("hpo_description", "")),
                    "hpo_term": _cell_to_str(row.get("hpo_term", "")),
                    "category": "Abnormal",
                }
            )
        return grouped

    def _load_gsc_annotations(
        self, workbook: pd.ExcelFile
    ) -> dict[str, list[dict[str, str]]]:
        frame = _parse_sheet(workbook, "GSC Manual Annotations").fillna("")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in frame.to_dict("records"):
            patient_id = _cell_to_str(row.get("Patient ID", ""))
            if not patient_id:
                continue
            grouped[patient_id].append(
                {
                    "hpo_description": _cell_to_str(row.get("hpo_description", "")),
                    "hpo_term": _cell_to_str(row.get("hpo_term", "")),
                    "category": _cell_to_str(row.get("Category", "")),
                }
            )
        return grouped

    def _build_documents(
        self,
        dataset_name: str,
        cases: dict[str, dict[str, str]],
        annotations_by_case: dict[str, list[dict[str, str]]],
        stats: ConversionStats,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []

        for patient_id in sorted(cases, key=_numeric_sort_key):
            case = cases[patient_id]
            full_text = case["full_text"]
            annotations_payload: list[dict[str, Any]] = []
            phrase_search_starts: dict[str, int] = defaultdict(int)
            for annotation in annotations_by_case.get(patient_id, []):
                phrase = annotation["hpo_description"]
                assertion_status = _normalize_assertion_status(annotation["category"])
                span_payloads: list[dict[str, Any]] = []
                if phrase:
                    span = find_span_in_text(
                        phrase,
                        full_text,
                        search_start=phrase_search_starts[phrase],
                    )
                    if span is not None:
                        span_payloads.append(
                            {
                                "start_char": span.start_char,
                                "end_char": span.end_char,
                                "text_snippet": span.text,
                            }
                        )
                        phrase_search_starts[phrase] = span.end_char
                    else:
                        stats.add_warning(
                            f"{dataset_name} case {patient_id}: evidence phrase "
                            f"'{phrase}' not found in source text"
                        )

                for hpo_id in _split_hpo_term_ids(annotation["hpo_term"]):
                    normalized_hpo_id = self._normalize_hpo_id(
                        hpo_id=hpo_id,
                        dataset_name=dataset_name,
                        patient_id=patient_id,
                        stats=stats,
                    )
                    if normalized_hpo_id is None:
                        continue
                    annotations_payload.append(
                        {
                            "hpo_id": normalized_hpo_id,
                            "label": self._lookup.resolve(normalized_hpo_id),
                            "assertion_status": assertion_status,
                            "evidence_spans": span_payloads,
                        }
                    )

            payloads.append(
                {
                    "doc_id": f"{dataset_name}_{patient_id}",
                    "language": "en",
                    "source": "rag_hpo_paper",
                    "full_text": full_text,
                    "metadata": {
                        "dataset": dataset_name,
                        "original_dataset": dataset_name,
                        "original_doc_id": case["original_doc_id"],
                        "text_length_chars": len(full_text),
                        "num_annotations": len(annotations_payload),
                        "conversion_date": _iso_now(),
                    },
                    "annotations": annotations_payload,
                }
            )

        return payloads

    def _filter_cases_to_annotations(
        self,
        *,
        dataset_name: str,
        cases: dict[str, dict[str, str]],
        annotations_by_case: dict[str, list[dict[str, str]]],
        stats: ConversionStats,
    ) -> dict[str, dict[str, str]]:
        annotated_case_ids = set(annotations_by_case)
        filtered_cases: dict[str, dict[str, str]] = {}
        for patient_id, case in cases.items():
            if patient_id not in annotated_case_ids:
                stats.add_warning(
                    f"{dataset_name} case {patient_id}: skipped because no manual "
                    "annotations were released for this case"
                )
                continue
            filtered_cases[patient_id] = case
        return filtered_cases

    def _normalize_hpo_id(
        self,
        *,
        hpo_id: str,
        dataset_name: str,
        patient_id: str,
        stats: ConversionStats,
    ) -> str | None:
        if not self._normalize_obsolete_ids:
            return hpo_id
        replacement_id = self._lookup.replacement_for(hpo_id)
        if replacement_id is None:
            if self._drop_obsolete_without_replacement and self._lookup.is_deprecated(
                hpo_id
            ):
                stats.add_warning(
                    f"{dataset_name} case {patient_id}: dropped obsolete HPO term "
                    f"{hpo_id} because no current replacement was available"
                )
                return None
            return hpo_id
        stats.add_warning(
            f"{dataset_name} case {patient_id}: normalized obsolete HPO term "
            f"{hpo_id} -> {replacement_id}"
        )
        return replacement_id

    def _write_documents(
        self,
        output_root: Path,
        dataset_name: str,
        documents: list[dict[str, Any]],
    ) -> None:
        annotations_dir = output_root / dataset_name / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        for document in documents:
            output_file = annotations_dir / f"{document['doc_id']}.json"
            output_file.write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def _numeric_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, value)


def _normalize_assertion_status(category: str) -> str:
    normalized = category.strip().lower()
    if not normalized:
        return "affirmed"
    return ASSERTION_STATUS_MAP.get(normalized, "affirmed")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _parse_sheet(workbook: pd.ExcelFile, expected_name: str) -> pd.DataFrame:
    sheet_map = {name.strip(): name for name in workbook.sheet_names}
    actual_name = sheet_map.get(expected_name)
    if actual_name is None:
        available = ", ".join(workbook.sheet_names)
        raise ValueError(
            f"Worksheet '{expected_name}' not found. Available sheets: {available}"
        )
    return workbook.parse(actual_name)


def _split_hpo_term_ids(raw_value: str) -> list[str]:
    matches = HPO_ID_PATTERN.findall(raw_value)
    if matches:
        return matches
    normalized = raw_value.strip()
    return [normalized] if normalized else []
