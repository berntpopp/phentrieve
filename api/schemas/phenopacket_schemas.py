from datetime import UTC, date, datetime, time
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

_NORMALIZED_SUBJECT_SEX = Literal["UNKNOWN_SEX", "FEMALE", "MALE", "OTHER_SEX"]
_SUBJECT_SEX_ALIASES = {
    "0": "UNKNOWN_SEX",
    "unknown_sex": "UNKNOWN_SEX",
    "unknown": "UNKNOWN_SEX",
    "1": "FEMALE",
    "female": "FEMALE",
    "2": "MALE",
    "male": "MALE",
    "3": "OTHER_SEX",
    "other_sex": "OTHER_SEX",
    "other": "OTHER_SEX",
}


class ExportSubjectRequest(BaseModel):
    id: str | None = None
    sex: _NORMALIZED_SUBJECT_SEX | None = None
    date_of_birth: str | None = Field(
        default=None,
        serialization_alias="dateOfBirth",
        validation_alias=AliasChoices("date_of_birth", "dateOfBirth"),
    )

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            if normalized in {"UNKNOWN_SEX", "FEMALE", "MALE", "OTHER_SEX"}:
                return normalized
            alias_match = _SUBJECT_SEX_ALIASES.get(normalized.lower())
            if alias_match is not None:
                return alias_match
        if isinstance(value, int):
            alias_match = _SUBJECT_SEX_ALIASES.get(str(value))
            if alias_match is not None:
                return alias_match

        raise ValueError(
            "subject.sex must be one of UNKNOWN_SEX, FEMALE, MALE, OTHER_SEX."
        )

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def normalize_date_of_birth(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            try:
                parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            except ValueError:
                try:
                    parsed_date = date.fromisoformat(normalized)
                except ValueError as exc:
                    raise ValueError(
                        "subject.dateOfBirth must be a valid ISO 8601 date or datetime."
                    ) from exc
                parsed = datetime.combine(parsed_date, time.min, tzinfo=UTC)
        elif isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, time.min, tzinfo=UTC)
        else:
            raise ValueError(
                "subject.dateOfBirth must be a valid ISO 8601 date or datetime."
            )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = parsed.astimezone(UTC)

        return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ExportTextAttributionRequest(BaseModel):
    chunk_id: int = Field(ge=0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    matched_text_in_chunk: str | None = None

    @model_validator(mode="after")
    def validate_character_span(self) -> "ExportTextAttributionRequest":
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must be greater than or equal to start_char.")
        return self


class ExportPhenotypeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hpo_id: str = Field(pattern=r"^HP:\d{7}$")
    label: str = Field(min_length=1)
    assertion_status: Literal["affirmed", "negated"] = "affirmed"
    certainty: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_text: str | None = None
    source_mode: str | None = None
    match_method: str | None = None
    source_chunk_ids: list[int] = Field(default_factory=list)
    text_attributions: list[ExportTextAttributionRequest] = Field(default_factory=list)

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, value: object) -> str:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
        raise ValueError("label must be a non-empty string.")


class PhenopacketExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
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
