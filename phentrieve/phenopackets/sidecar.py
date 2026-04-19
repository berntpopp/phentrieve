"""Helpers for building and validating phenotype annotation sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from phentrieve.phenopackets.export_models import NormalizedPhenotypeExportRecord

_SCHEMA_FILENAME = "phenotype_annotation_bundle_v1.schema.json"
_SCHEMA_VERSION = "1.0.0"
_ARTIFACT_TYPE = "phenotype_annotation_bundle"


def load_annotation_sidecar_schema() -> dict[str, Any]:
    """Load the checked-in JSON Schema for annotation sidecars."""
    schema_path = Path(__file__).resolve().parent / "schemas" / _SCHEMA_FILENAME
    return json.loads(schema_path.read_text(encoding="utf-8"))


def build_annotation_sidecar(
    *,
    phenopacket_id: str,
    records: list[NormalizedPhenotypeExportRecord],
    generated_by_version: str,
) -> dict[str, Any]:
    """Build a normalized annotation sidecar from export records."""
    annotations: list[dict[str, Any]] = []

    for feature_index, record in enumerate(records):
        annotation: dict[str, Any] = {
            "annotation_id": f"ann-{feature_index + 1:04d}",
            "phenotypic_feature_index": feature_index,
            "hpo_id": record.hpo_id,
            "label": record.label,
            "assertion": record.assertion,
            "spans": [
                {
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "text": span.text,
                }
                for span in record.spans
            ],
            "chunk_refs": list(record.chunk_refs),
            "provenance": {
                "sidecar_linkage_key": record.sidecar_linkage_key,
            },
        }

        if record.confidence is not None:
            annotation["confidence"] = record.confidence
        certainty = getattr(record, "certainty", None)
        if certainty is not None:
            annotation["certainty"] = certainty
        if record.evidence_text is not None:
            annotation["evidence_text"] = record.evidence_text
        if record.source_mode is not None:
            annotation["provenance"]["source_mode"] = record.source_mode
        if record.match_method is not None:
            annotation["provenance"]["match_method"] = record.match_method

        annotations.append(annotation)

    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "generated_by": {
            "tool": "phentrieve",
            "version": generated_by_version,
        },
        "phenopacket_id": phenopacket_id,
        "annotations": annotations,
    }


def validate_annotation_sidecar(sidecar: dict[str, Any]) -> None:
    """Validate a sidecar against the checked-in JSON Schema."""
    jsonschema.validate(instance=sidecar, schema=load_annotation_sidecar_schema())
