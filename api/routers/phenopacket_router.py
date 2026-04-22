from typing import Any

from fastapi import APIRouter

from api.schemas.phenopacket_schemas import PhenopacketExportRequest
from phentrieve.phenopackets.utils import export_phenopacket_bundle

router = APIRouter(prefix="/api/v1/phenopackets", tags=["Phenopackets"])


@router.post("/export")
def export_phenopacket(request: PhenopacketExportRequest) -> dict[str, Any]:
    aggregated_results = [
        {
            "hpo_id": phenotype.hpo_id,
            "name": phenotype.label,
            "status": "negated"
            if phenotype.assertion_status == "negated"
            else "affirmed",
            "source_chunk_ids": phenotype.source_chunk_ids,
            "text_attributions": [
                item.model_dump() for item in phenotype.text_attributions
            ],
        }
        for phenotype in request.phenotypes
    ]

    return export_phenopacket_bundle(
        aggregated_results=aggregated_results,
        input_text=request.input_text,
        include_annotation_sidecar=request.include_annotation_sidecar,
    )
