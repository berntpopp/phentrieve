"""This module provides utilities for creating and handling Phenopackets."""

import datetime
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

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

logger = logging.getLogger(__name__)


def _get_hpo_version_from_db(db_path: Optional[Path | str] = None) -> str:
    """
    Get HPO version from database metadata.

    Args:
        db_path: Optional path to HPO database. If not provided, uses default.

    Returns:
        HPO version string (e.g., "v2025-03-03") or "unknown" if not found
    """
    try:
        if db_path is None:
            from phentrieve.config import DEFAULT_HPO_DB_FILENAME
            from phentrieve.utils import get_default_data_dir

            db_path = get_default_data_dir() / DEFAULT_HPO_DB_FILENAME

        # Check if database exists
        if isinstance(db_path, str):
            db_path = Path(db_path)

        if not db_path.exists():
            logger.debug(f"HPO database not found at {db_path}")
            return "unknown"

        from phentrieve.data_processing.hpo_database import HPODatabase

        db = HPODatabase(db_path)
        version = db.get_metadata("hpo_version")
        db.close()

        return version or "unknown"
    except Exception as e:
        logger.debug(f"Failed to retrieve HPO version from database: {e}")
        return "unknown"


def format_as_phenopacket_v2(
    aggregated_results: Optional[list[dict[str, Any]]] = None,
    chunk_results: Optional[list[dict[str, Any]]] = None,
    phentrieve_version: Optional[str] = None,
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    hpo_version: Optional[str] = None,
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
        )
    elif aggregated_results is not None and len(aggregated_results) > 0:
        return _format_from_aggregated_results(
            aggregated_results,
            phentrieve_version=phentrieve_version,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            hpo_version=hpo_version,
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
        )


def _format_from_chunk_results(
    chunk_results: list[dict[str, Any]],
    phentrieve_version: str = "unknown",
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    hpo_version: str = "unknown",
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

    Returns:
        A JSON string representing the Phenopacket.
    """
    phenopacket_id = f"phentrieve-phenopacket-{uuid.uuid4()}"
    phenotypic_features = []

    for chunk_result in chunk_results:
        chunk_idx = chunk_result.get("chunk_idx", 0)
        chunk_text = chunk_result.get("chunk_text", "")
        matches = chunk_result.get("matches", [])

        for match in matches:
            hpo_id = match.get("id", "")
            hpo_name = match.get("name", "")
            score = match.get("score", 0.0)
            assertion_status = match.get("assertion_status")

            # Create OntologyClass for the feature type
            feature_type = OntologyClass(id=hpo_id, label=hpo_name)

            # Build description with confidence, chunk info, and source text
            # Note: No rank is provided since rankings are not comparable across chunks
            description_parts = [
                f"Phentrieve retrieval confidence: {score:.4f}",
                f"Chunk: {chunk_idx + 1}",
                f"Source text: {chunk_text}",
            ]
            if assertion_status:
                description_parts.insert(1, f"Assertion: {assertion_status}")

            # Create ExternalReference with evidence details
            external_reference = ExternalReference(
                id="phentrieve",
                description=" | ".join(description_parts),
            )

            # Create Evidence object
            evidence = Evidence(
                evidence_code=OntologyClass(
                    id="ECO:0007636",
                    label="computational evidence used in automatic assertion",
                ),
                reference=external_reference,
            )

            # Determine if the phenotypic feature is excluded (negated)
            # Use case-insensitive comparison for robustness
            excluded = assertion_status is not None and assertion_status.lower() in (
                "negated",
                "absent",
            )

            # Create PhenotypicFeature
            phenotypic_feature = PhenotypicFeature(
                type=feature_type,
                excluded=excluded,
                evidence=[evidence],
            )
            phenotypic_features.append(phenotypic_feature)

    return _create_phenopacket_json(
        phenopacket_id,
        phenotypic_features,
        phentrieve_version=phentrieve_version,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        hpo_version=hpo_version,
    )


def _format_from_aggregated_results(
    aggregated_results: list[dict[str, Any]],
    phentrieve_version: str = "unknown",
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    hpo_version: str = "unknown",
) -> str:
    """Format phenopacket from aggregated results (legacy/fallback format).

    Args:
        aggregated_results: A list of dictionaries representing aggregated HPO
            term results with 'id', 'name', 'confidence', and 'rank' fields.
        phentrieve_version: Phentrieve version string.
        embedding_model: Name of embedding model used.
        reranker_model: Name of reranker model used.
        hpo_version: HPO version string.

    Returns:
        A JSON string representing the Phenopacket.
    """
    phenopacket_id = f"phentrieve-phenopacket-{uuid.uuid4()}"

    # Sort results by rank
    sorted_results = sorted(aggregated_results, key=lambda x: x.get("rank", 0))

    phenotypic_features = []
    for result in sorted_results:
        confidence = result.get("confidence", 0.0)
        rank = result.get("rank")

        # Create OntologyClass for the feature type
        feature_type = OntologyClass(id=result["id"], label=result["name"])

        # Create ExternalReference to store confidence and rank
        external_reference = ExternalReference(
            id="phentrieve",
            description=f"Phentrieve retrieval confidence: {confidence:.4f}, Rank: {rank}",
        )

        # Create Evidence object with an evidence code and the external reference
        evidence = Evidence(
            evidence_code=OntologyClass(
                id="ECO:0007636",
                label="computational evidence used in automatic assertion",
            ),
            reference=external_reference,
        )

        # Create PhenotypicFeature
        phenotypic_feature = PhenotypicFeature(
            type=feature_type,
            evidence=[evidence],
        )
        phenotypic_features.append(phenotypic_feature)

    return _create_phenopacket_json(
        phenopacket_id,
        phenotypic_features,
        phentrieve_version=phentrieve_version,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        hpo_version=hpo_version,
    )


def _create_phenopacket_json(
    phenopacket_id: str,
    phenotypic_features: list[PhenotypicFeature],
    phentrieve_version: str = "unknown",
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    hpo_version: str = "unknown",
) -> str:
    """Create the final Phenopacket JSON from components.

    Args:
        phenopacket_id: Unique identifier for the phenopacket.
        phenotypic_features: List of PhenotypicFeature objects.
        phentrieve_version: Phentrieve version string.
        embedding_model: Name of embedding model used.
        reranker_model: Name of reranker model used.
        hpo_version: HPO version string.

    Returns:
        A JSON string representing the Phenopacket.
    """
    # Create MetaData
    created_timestamp = Timestamp()
    created_timestamp.FromDatetime(datetime.datetime.now(datetime.timezone.utc))

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
        phenopacket_schema_version="2.0.2",
    )

    # Create Phenopacket
    phenopacket = Phenopacket(
        id=phenopacket_id,
        phenotypic_features=phenotypic_features,
        meta_data=meta_data,
    )

    return str(MessageToJson(phenopacket, indent=2))
