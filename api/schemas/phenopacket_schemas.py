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
