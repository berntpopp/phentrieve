"""This module provides utilities for creating and handling Phenopackets."""

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToJson
from google.protobuf.timestamp_pb2 import Timestamp
from phenopackets import (
    Evidence,
    ExternalReference,
    MetaData,
    OntologyClass,
    Phenopacket,
    PhenotypicFeature,
    Resource,
)

from phentrieve.phenopackets.export_models import NormalizedPhenotypeExportRecord
from phentrieve.phenopackets.sidecar import (
    build_annotation_sidecar,
    validate_annotation_sidecar,
)

logger = logging.getLogger(__name__)

# Maximum length for input text in metadata (prevent excessive payload size)
_MAX_INPUT_TEXT_LENGTH = 1000
VERIFIED_PHENOPACKET_SCHEMA_VERSION = "2.0"
_SIDECAR_EXTERNAL_REFERENCE_ID = "phentrieve:annotation_sidecar"


def _coerce_rank(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_phenotypic_feature(
    record: NormalizedPhenotypeExportRecord,
    *,
    description_parts: list[str] | None = None,
) -> PhenotypicFeature:
    feature_type = OntologyClass(id=record.hpo_id, label=record.label)
    external_reference = None
    if description_parts:
        external_reference = ExternalReference(
            id="phentrieve",
            description=" | ".join(description_parts),
        )
    evidence = [
        Evidence(
            evidence_code=OntologyClass(
                id="ECO:0007636",
                label="computational evidence used in automatic assertion",
            ),
            reference=external_reference,
        )
    ]
    excluded = record.assertion == "negated"
    return PhenotypicFeature(type=feature_type, excluded=excluded, evidence=evidence)


def _get_hpo_version_from_db(db_path: Path | str | None = None) -> str:
    """
    Get HPO version from database metadata.

    Args:
        db_path: Optional path to HPO database. If not provided, uses default.

    Returns:
        HPO version string (e.g., "v2025-03-03") or "unknown" if not found
    """
    try:
        from phentrieve.config import DEFAULT_HPO_DB_FILENAME
        from phentrieve.utils import get_default_data_dir

        candidates: list[Path] = []

        # If caller provided an explicit path, try that first
        if db_path is not None:
            candidates.append(Path(db_path))

        # Primary configured data dir (could be ~/.phentrieve/data or env override)
        candidates.append(get_default_data_dir() / DEFAULT_HPO_DB_FILENAME)

        # Common repo-local data dir (useful when running from project root)
        candidates.append(Path.cwd() / "data" / DEFAULT_HPO_DB_FILENAME)

        # Package-relative data dir (in case code is executed from installed package)
        package_root = Path(__file__).resolve().parents[2]
        candidates.append(package_root / "data" / DEFAULT_HPO_DB_FILENAME)

        # Try candidates in order and pick the first that exists
        db_file: Path | None = None
        for p in candidates:
            try:
                if p and p.exists():
                    db_file = p
                    break
            except Exception:
                continue

        if db_file is None:
            logger.debug(
                "HPO database not found in any known locations: %s", candidates
            )
            return "unknown"

        from phentrieve.data_processing.hpo_database import HPODatabase

        db = HPODatabase(db_file)
        try:
            version = db.get_metadata("hpo_version")
        finally:
            db.close()

        return version or "unknown"
    except Exception as e:
        logger.debug("Failed to retrieve HPO version from database: %s", e)
        return "unknown"


def format_as_phenopacket_v2(
    aggregated_results: list[dict[str, Any]] | None = None,
    chunk_results: list[dict[str, Any]] | None = None,
    phentrieve_version: str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    hpo_version: str | None = None,
    input_text: str | None = None,
) -> str:
    """Format HPO extraction results as a Phenopacket v2 JSON string.

    This function can work with either chunk_results (preferred) or aggregated_results.
    When chunk_results are provided, each chunk's HPO matches are preserved with their
    source text as evidence. When only aggregated_results are provided, the function
    falls back to the simpler format without chunk text evidence.

    Args:
        aggregated_results: A list of dictionaries representing aggregated HPO term
            results. Used as fallback when chunk_results is not provided.
        chunk_results: A list of dictionaries containing chunk-level results with
            matches. Each dict should have 'chunk_idx', 'chunk_text', and 'matches'.
            When provided, this is the preferred data source.
        phentrieve_version: Version of Phentrieve used (e.g., "0.3.0"). If None, retrieved from package.
        embedding_model: Name of embedding model used (e.g., "BAAI/bge-m3").
        reranker_model: Name of reranker model used (e.g., "BAAI/bge-reranker-v2-m3").
        hpo_version: Version of HPO used (e.g., "v2025-03-03"). If None, retrieved from database.
        input_text: Original input text/query for provenance tracking in metadata.

    Returns:
        A JSON string representing the Phenopacket.
    """
    # Retrieve missing metadata from sources
    if phentrieve_version is None:
        try:
            from phentrieve import __version__

            phentrieve_version = __version__
        except ImportError:
            phentrieve_version = "unknown"

    if hpo_version is None:
        hpo_version = _get_hpo_version_from_db()

    # Use chunk_results if provided, otherwise fall back to aggregated_results
    # Note: Explicit length checks prevent empty lists from taking wrong branch
    if chunk_results is not None and len(chunk_results) > 0:
        return _format_from_chunk_results(
            chunk_results,
            phentrieve_version=phentrieve_version,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            hpo_version=hpo_version,
            input_text=input_text,
        )
    elif aggregated_results is not None and len(aggregated_results) > 0:
        return _format_from_aggregated_results(
            aggregated_results,
            phentrieve_version=phentrieve_version,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            hpo_version=hpo_version,
            input_text=input_text,
        )
    else:
        # Return valid empty Phenopacket instead of raw "{}" string
        phenopacket_id = f"phentrieve-phenopacket-{uuid.uuid4()}"
        return _create_phenopacket_json(
            phenopacket_id,
            [],
            phentrieve_version=phentrieve_version,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            hpo_version=hpo_version,
            input_text=input_text,
        )


def export_phenopacket_bundle(
    *,
    aggregated_results: list[dict[str, Any]] | None = None,
    chunk_results: list[dict[str, Any]] | None = None,
    phentrieve_version: str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    hpo_version: str | None = None,
    input_text: str | None = None,
    include_annotation_sidecar: bool = False,
) -> dict[str, Any]:
    """Export a strict Phenopacket JSON string and optional annotation sidecar."""
    resolved_phentrieve_version = phentrieve_version
    if resolved_phentrieve_version is None:
        try:
            from phentrieve import __version__

            resolved_phentrieve_version = __version__
        except ImportError:
            resolved_phentrieve_version = "unknown"

    resolved_hpo_version = hpo_version
    if resolved_hpo_version is None:
        resolved_hpo_version = _get_hpo_version_from_db()

    phenopacket_json = format_as_phenopacket_v2(
        aggregated_results=aggregated_results,
        chunk_results=chunk_results,
        phentrieve_version=resolved_phentrieve_version,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        hpo_version=resolved_hpo_version,
        input_text=input_text,
    )

    annotation_sidecar = None
    if include_annotation_sidecar:
        normalized_records = _normalize_export_records(
            aggregated_results=aggregated_results,
            chunk_results=chunk_results,
        )
        phenopacket_id = json.loads(phenopacket_json)["id"]
        annotation_sidecar = build_annotation_sidecar(
            phenopacket_id=phenopacket_id,
            records=normalized_records,
            generated_by_version=resolved_phentrieve_version,
        )
        validate_annotation_sidecar(annotation_sidecar)
        phenopacket_payload = json.loads(phenopacket_json)
        meta = phenopacket_payload.setdefault("metaData", {})
        external_references = meta.setdefault("externalReferences", [])
        external_references.append(
            {
                "id": _SIDECAR_EXTERNAL_REFERENCE_ID,
                "reference": (
                    f"urn:phentrieve:phenotype-annotation-bundle:"
                    f"{annotation_sidecar['schema_version']}:{phenopacket_id}"
                ),
                "description": "Linked phenotype annotation sidecar emitted with this Phenopacket bundle.",
            }
        )
        phenopacket_json = json.dumps(phenopacket_payload, indent=2)

    return {
        "phenopacket_json": phenopacket_json,
        "annotation_sidecar": annotation_sidecar,
    }


def _format_from_chunk_results(
    chunk_results: list[dict[str, Any]],
    phentrieve_version: str = "unknown",
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    hpo_version: str = "unknown",
    input_text: str | None = None,
) -> str:
    """Format phenopacket from chunk-level results with text evidence.

    Each chunk's HPO matches are included as separate phenotypic features,
    with the source text chunk included in the evidence. No global ranking is
    provided since rankings cannot be compared across different text chunks.

    Args:
        chunk_results: List of chunk results, each containing 'chunk_idx',
            'chunk_text', and 'matches' (list of HPO term matches).
        phentrieve_version: Phentrieve version string.
        embedding_model: Name of embedding model used.
        reranker_model: Name of reranker model used.
        hpo_version: HPO version string.
        input_text: Original input text for metadata.

    Returns:
        A JSON string representing the Phenopacket.
    """
    phenopacket_id = f"phentrieve-phenopacket-{uuid.uuid4()}"
    normalized_records = _normalize_export_records(chunk_results=chunk_results)
    phenotypic_features = []

    for record in normalized_records:
        description_parts = [
            f"Phentrieve retrieval confidence: {(record.confidence or 0.0):.4f}",
        ]
        if record.chunk_refs:
            description_parts.append(f"Chunk: {record.chunk_refs[0] + 1}")
        if record.spans:
            description_parts.append(f"Start: {record.spans[0].start_char}")
            description_parts.append(f"End: {record.spans[0].end_char}")
        if record.assertion:
            description_parts.append(f"Assertion: {record.assertion}")
        if record.evidence_text:
            description_parts.append(f"Source text: {record.evidence_text}")
        phenotypic_features.append(
            _build_phenotypic_feature(record, description_parts=description_parts)
        )

    return _create_phenopacket_json(
        phenopacket_id,
        phenotypic_features,
        phentrieve_version=phentrieve_version,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        hpo_version=hpo_version,
        input_text=input_text,
    )


def _format_from_aggregated_results(
    aggregated_results: list[dict[str, Any]],
    phentrieve_version: str = "unknown",
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    hpo_version: str = "unknown",
    input_text: str | None = None,
) -> str:
    """Format phenopacket from aggregated results (legacy/fallback format).

    Args:
        aggregated_results: A list of dictionaries representing aggregated HPO
            term results with 'id', 'name', 'confidence', and 'rank' fields.
        phentrieve_version: Phentrieve version string.
        embedding_model: Name of embedding model used.
        reranker_model: Name of reranker model used.
        hpo_version: HPO version string.
        input_text: Original input text/query for metadata.

    Returns:
        A JSON string representing the Phenopacket.
    """
    phenopacket_id = f"phentrieve-phenopacket-{uuid.uuid4()}"

    sorted_results = sorted(
        aggregated_results, key=lambda x: _coerce_rank(x.get("rank", 0))
    )
    normalized_results = _normalize_aggregated_results(sorted_results)

    phenotypic_features = []
    for index, (legacy_result, result) in enumerate(
        zip(sorted_results, normalized_results, strict=True),
        start=1,
    ):
        confidence = result.confidence or 0.0
        rank = legacy_result.get("rank", index)

        phenotypic_features.append(
            _build_phenotypic_feature(
                result,
                description_parts=[
                    f"Phentrieve retrieval confidence: {confidence:.4f}, Rank: {rank}"
                ],
            )
        )

    return _create_phenopacket_json(
        phenopacket_id,
        phenotypic_features,
        phentrieve_version=phentrieve_version,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        hpo_version=hpo_version,
        input_text=input_text,
    )


def _normalize_aggregated_results(
    aggregated_results: list[dict[str, Any]],
) -> list[NormalizedPhenotypeExportRecord]:
    return [
        NormalizedPhenotypeExportRecord.from_legacy_dict(result)
        for result in aggregated_results
    ]


def _normalize_export_records(
    *,
    aggregated_results: list[dict[str, Any]] | None = None,
    chunk_results: list[dict[str, Any]] | None = None,
) -> list[NormalizedPhenotypeExportRecord]:
    if chunk_results is not None and len(chunk_results) > 0:
        normalized_records: list[NormalizedPhenotypeExportRecord] = []
        for chunk_result in chunk_results:
            chunk_idx = chunk_result.get("chunk_idx", 0)
            chunk_text = chunk_result.get("chunk_text")
            start_char = chunk_result.get("start_char")
            end_char = chunk_result.get("end_char")
            matches = chunk_result.get("matches", [])

            for match in matches:
                normalized_match = dict(match)
                normalized_match.setdefault("hpo_id", normalized_match.get("id"))
                normalized_match.setdefault(
                    "label",
                    normalized_match.get("name") or normalized_match.get("term_name"),
                )
                normalized_match["evidence_text"] = chunk_text
                normalized_match["chunk_refs"] = [chunk_idx]
                valid_span = (
                    isinstance(start_char, int)
                    and isinstance(end_char, int)
                    and start_char >= 0
                    and end_char >= start_char
                    and isinstance(chunk_text, str)
                    and bool(chunk_text)
                )
                if valid_span:
                    normalized_match["spans"] = [
                        {
                            "text": chunk_text,
                            "start_char": start_char,
                            "end_char": end_char,
                            "chunk_refs": [chunk_idx],
                        }
                    ]
                normalized_records.append(
                    NormalizedPhenotypeExportRecord.from_legacy_dict(normalized_match)
                )

        return normalized_records

    if aggregated_results is not None and len(aggregated_results) > 0:
        sorted_results = sorted(
            aggregated_results, key=lambda x: _coerce_rank(x.get("rank", 0))
        )
        return _normalize_aggregated_results(sorted_results)

    return []


def _create_phenopacket_json(
    phenopacket_id: str,
    phenotypic_features: list[PhenotypicFeature],
    phentrieve_version: str = "unknown",
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    hpo_version: str = "unknown",
    input_text: str | None = None,
) -> str:
    """Create the final Phenopacket JSON from components.

    Args:
        phenopacket_id: Unique identifier for the phenopacket.
        phenotypic_features: List of PhenotypicFeature objects.
        phentrieve_version: Phentrieve version string.
        embedding_model: Name of embedding model used.
        reranker_model: Name of reranker model used.
        hpo_version: HPO version string.
        input_text: Original input text for provenance tracking.

    Returns:
        A JSON string representing the Phenopacket.
    """
    # Create MetaData
    created_timestamp = Timestamp()
    created_timestamp.FromDatetime(datetime.datetime.now(datetime.UTC))

    # Build external references for tool provenance
    external_references = []
    if embedding_model:
        external_references.append(
            ExternalReference(
                id="phentrieve:embedding_model",
                description=embedding_model,
            )
        )
    if reranker_model:
        external_references.append(
            ExternalReference(
                id="phentrieve:reranker_model",
                description=reranker_model,
            )
        )
    if input_text:
        # Truncate long input text to prevent excessive payload size
        truncated = (
            input_text
            if len(input_text) <= _MAX_INPUT_TEXT_LENGTH
            else input_text[:_MAX_INPUT_TEXT_LENGTH] + "..."
        )
        external_references.append(
            ExternalReference(
                id="phentrieve:input_text",
                description=truncated,
            )
        )

    meta_data = MetaData(
        created=created_timestamp,
        created_by=f"phentrieve {phentrieve_version}",
        external_references=external_references if external_references else None,
        resources=[
            Resource(
                id="hp",
                name="human phenotype ontology",
                url="http://purl.obolibrary.org/obo/hp.owl",
                version=hpo_version,
                namespace_prefix="HP",
                iri_prefix="http://purl.obolibrary.org/obo/HP_",
            )
        ],
        phenopacket_schema_version=VERIFIED_PHENOPACKET_SCHEMA_VERSION,
    )

    # Create Phenopacket
    phenopacket = Phenopacket(
        id=phenopacket_id,
        phenotypic_features=phenotypic_features,
        meta_data=meta_data,
    )

    return str(MessageToJson(phenopacket, indent=2))
