import json
from typing import Any

from fastapi import APIRouter

from api.schemas.phenopacket_schemas import (
    ExportPhenotypeRequest,
    ExportTextAttributionRequest,
    PhenopacketExportRequest,
    PhenopacketExportResponse,
)
from phentrieve.phenopackets.utils import export_phenopacket_bundle

router = APIRouter(prefix="/api/v1/phenopackets", tags=["Phenopackets"])


def _map_text_attribution(
    attribution: ExportTextAttributionRequest,
) -> dict[str, Any] | None:
    if (
        attribution.start_char is None
        or attribution.end_char is None
        or attribution.matched_text_in_chunk is None
    ):
        return None

    return {
        "start_char": attribution.start_char,
        "end_char": attribution.end_char,
        "evidence_text": attribution.matched_text_in_chunk,
        "chunk_refs": [attribution.chunk_id],
    }


def _map_phenotype_for_export(phenotype: ExportPhenotypeRequest) -> dict[str, Any]:
    spans = [
        mapped_span
        for attribution in phenotype.text_attributions
        if (mapped_span := _map_text_attribution(attribution)) is not None
    ]

    chunk_refs = list(phenotype.source_chunk_ids)
    if not chunk_refs:
        for span in spans:
            for chunk_ref in span["chunk_refs"]:
                if chunk_ref not in chunk_refs:
                    chunk_refs.append(chunk_ref)

    export_record = phenotype.model_dump(
        exclude_none=True,
        include={
            "hpo_id",
            "label",
            "certainty",
            "confidence",
            "evidence_text",
            "source_mode",
            "match_method",
        },
    )
    export_record.update(
        {
            "hpo_id": phenotype.hpo_id,
            "label": phenotype.label,
            "assertion": phenotype.assertion_status,
            "assertion_status": phenotype.assertion_status,
            "chunk_refs": chunk_refs,
            "spans": spans,
        }
    )
    return export_record


def _build_subject_payload(request: PhenopacketExportRequest) -> dict[str, str] | None:
    if request.subject is None:
        return None

    subject_payload = request.subject.model_dump(
        by_alias=True,
        exclude_none=True,
        include={"id", "sex", "date_of_birth"},
    )
    return subject_payload or None


def _update_sidecar_linked_packet_id(
    bundle: dict[str, Any],
    external_references: list[dict[str, Any]],
    old_id: str,
    new_id: str,
) -> None:
    annotation_sidecar = bundle.get("annotation_sidecar")
    if annotation_sidecar is not None:
        annotation_sidecar["phenopacket_id"] = new_id

    for reference in external_references:
        if reference.get("id") == "phentrieve:annotation_sidecar":
            reference["reference"] = str(reference.get("reference", "")).replace(
                old_id,
                new_id,
            )


def _apply_request_metadata_to_bundle(
    bundle: dict[str, Any], request: PhenopacketExportRequest
) -> dict[str, Any]:
    phenopacket_payload = json.loads(bundle["phenopacket_json"])
    original_packet_id = phenopacket_payload.get("id")

    if request.case_id:
        phenopacket_payload["id"] = request.case_id

    subject_payload = _build_subject_payload(request)
    if subject_payload is not None:
        phenopacket_payload["subject"] = subject_payload

    meta = phenopacket_payload.setdefault("metaData", {})
    external_references = meta.setdefault("externalReferences", [])
    if request.case_label:
        external_references.append(
            {
                "id": "phentrieve:case_label",
                "description": request.case_label,
            }
        )

    if (
        original_packet_id
        and request.case_id
        and bundle.get("annotation_sidecar") is not None
        and original_packet_id != request.case_id
    ):
        _update_sidecar_linked_packet_id(
            bundle,
            external_references,
            old_id=original_packet_id,
            new_id=request.case_id,
        )

    bundle["phenopacket_json"] = json.dumps(phenopacket_payload, indent=2)
    return bundle


@router.post("/export", response_model=PhenopacketExportResponse)
def export_phenopacket(
    request: PhenopacketExportRequest,
) -> PhenopacketExportResponse:
    aggregated_results = [
        _map_phenotype_for_export(phenotype) for phenotype in request.phenotypes
    ]

    bundle = export_phenopacket_bundle(
        aggregated_results=aggregated_results,
        input_text=request.input_text,
        include_annotation_sidecar=request.include_annotation_sidecar,
    )
    bundle = _apply_request_metadata_to_bundle(bundle, request)
    return PhenopacketExportResponse.model_validate(bundle)
