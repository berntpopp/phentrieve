from typing import Any

from fastapi import APIRouter

from api.schemas.phenopacket_schemas import (
    ExportPhenotypeRequest,
    ExportTextAttributionRequest,
    PhenopacketExportRequest,
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


@router.post("/export")
def export_phenopacket(request: PhenopacketExportRequest) -> dict[str, Any]:
    aggregated_results = [
        _map_phenotype_for_export(phenotype) for phenotype in request.phenotypes
    ]

    return export_phenopacket_bundle(
        aggregated_results=aggregated_results,
        input_text=request.input_text,
        include_annotation_sidecar=request.include_annotation_sidecar,
    )
