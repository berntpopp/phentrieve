"""Helpers for building and validating phenotype annotation sidecars."""

from __future__ import annotations

import importlib.resources
import json
from typing import Any, cast

from phentrieve.phenopackets.export_models import NormalizedPhenotypeExportRecord

_SCHEMA_FILENAME = "phenotype_annotation_bundle_v1.schema.json"
_SCHEMA_VERSION = "1.0.0"
_ARTIFACT_TYPE = "phenotype_annotation_bundle"


def load_annotation_sidecar_schema() -> dict[str, Any]:
    """Load the checked-in JSON Schema for annotation sidecars."""
    schema_text = (
        importlib.resources.files("phentrieve.phenopackets")
        .joinpath("schemas", _SCHEMA_FILENAME)
        .read_text(encoding="utf-8")
    )
    return cast(dict[str, Any], json.loads(schema_text))


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
            "provenance": {},
        }

        if record.confidence is not None:
            annotation["confidence"] = record.confidence
        if record.certainty is not None:
            annotation["certainty"] = record.certainty
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
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise ImportError(
            "Annotation sidecar validation requires jsonschema. "
            "Install phentrieve[phenopackets] to enable this feature."
        ) from exc

    try:
        jsonschema.validate(instance=sidecar, schema=load_annotation_sidecar_schema())
    except jsonschema.ValidationError as exc:
        path_parts = ["$"]
        for part in exc.absolute_path:
            if isinstance(part, int):
                path_parts.append(f"[{part}]")
            else:
                path_parts.append(f".{part}")
        instance_path = "".join(path_parts)
        raise ValueError(
            "Invalid annotation sidecar at "
            f"{instance_path}: {exc.message}. "
            "Check that the sidecar matches schema version "
            f"{_SCHEMA_VERSION} and includes all required fields."
        ) from exc
