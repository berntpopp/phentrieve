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

    return {
        "hpo_id": phenotype.hpo_id,
        "label": phenotype.label,
        "assertion": phenotype.assertion_status,
        "assertion_status": phenotype.assertion_status,
        "chunk_refs": chunk_refs,
        "spans": spans,
    }


def _apply_request_metadata_to_bundle(
    bundle: dict[str, Any], request: PhenopacketExportRequest
) -> dict[str, Any]:
    phenopacket_payload = json.loads(bundle["phenopacket_json"])
    original_packet_id = phenopacket_payload.get("id")

    if request.case_id:
        phenopacket_payload["id"] = request.case_id

    if request.subject is not None:
        subject_payload = {}
        if request.subject.id is not None:
            subject_payload["id"] = request.subject.id
        if request.subject.sex is not None:
            subject_payload["sex"] = request.subject.sex
        if request.subject.date_of_birth is not None:
            subject_payload["dateOfBirth"] = request.subject.date_of_birth
        if subject_payload:
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
        bundle["annotation_sidecar"]["phenopacket_id"] = request.case_id
        for reference in external_references:
            if reference.get("id") == "phentrieve:annotation_sidecar":
                reference["reference"] = str(reference.get("reference", "")).replace(
                    original_packet_id,
                    request.case_id,
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
