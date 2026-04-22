from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class ExportSubjectRequest(BaseModel):
    id: str | None = None
    sex: str | None = None
    date_of_birth: str | None = Field(
        default=None,
        validation_alias=AliasChoices("date_of_birth", "dateOfBirth"),
    )


class ExportTextAttributionRequest(BaseModel):
    chunk_id: int
    start_char: int | None = None
    end_char: int | None = None
    matched_text_in_chunk: str | None = None


class ExportPhenotypeRequest(BaseModel):
    hpo_id: str
    label: str
    assertion_status: str = "affirmed"
    source_chunk_ids: list[int] = Field(default_factory=list)
    text_attributions: list[ExportTextAttributionRequest] = Field(default_factory=list)
    confidence_band: Literal["high", "medium", "low"] | None = None


class PhenopacketExportRequest(BaseModel):
    case_id: str
    case_label: str | None = None
    input_text: str | None = None
    subject: ExportSubjectRequest | None = None
    include_annotation_sidecar: bool = False
    phenotypes: list[ExportPhenotypeRequest] = Field(default_factory=list)


class AnnotationSidecarGeneratedByResponse(BaseModel):
    tool: str
    version: str


class AnnotationSidecarSpanResponse(BaseModel):
    start_char: int
    end_char: int
    text: str


class AnnotationSidecarAnnotationResponse(BaseModel):
    annotation_id: str
    phenotypic_feature_index: int
    hpo_id: str
    label: str
    assertion: str
    spans: list[AnnotationSidecarSpanResponse] = Field(default_factory=list)
    chunk_refs: list[int] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)
    confidence: float | None = None
    certainty: str | None = None
    evidence_text: str | None = None


class AnnotationSidecarResponse(BaseModel):
    schema_version: str
    artifact_type: str
    generated_by: AnnotationSidecarGeneratedByResponse
    phenopacket_id: str
    annotations: list[AnnotationSidecarAnnotationResponse] = Field(default_factory=list)


class PhenopacketExportResponse(BaseModel):
    phenopacket_json: str
    annotation_sidecar: AnnotationSidecarResponse | None = None
