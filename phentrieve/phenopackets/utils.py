"""This module provides utilities for creating and handling Phenopackets."""

import datetime
import uuid
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


def format_as_phenopacket_v2(
    aggregated_results: Optional[list[dict[str, Any]]] = None,
    chunk_results: Optional[list[dict[str, Any]]] = None,
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

    Returns:
        A JSON string representing the Phenopacket.
    """
    # Use chunk_results if provided, otherwise fall back to aggregated_results
    # Note: Explicit length checks prevent empty lists from taking wrong branch
    if chunk_results is not None and len(chunk_results) > 0:
        return _format_from_chunk_results(chunk_results)
    elif aggregated_results is not None and len(aggregated_results) > 0:
        return _format_from_aggregated_results(aggregated_results)
    else:
        # Return valid empty Phenopacket instead of raw "{}" string
        phenopacket_id = f"phentrieve-phenopacket-{uuid.uuid4()}"
        return _create_phenopacket_json(phenopacket_id, [])


def _format_from_chunk_results(chunk_results: list[dict[str, Any]]) -> str:
    """Format phenopacket from chunk-level results with text evidence.

    Each chunk's HPO matches are included as separate phenotypic features,
    with the source text chunk included in the evidence. No global ranking is
    provided since rankings cannot be compared across different text chunks.

    Args:
        chunk_results: List of chunk results, each containing 'chunk_idx',
            'chunk_text', and 'matches' (list of HPO term matches).

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

    return _create_phenopacket_json(phenopacket_id, phenotypic_features)


def _format_from_aggregated_results(aggregated_results: list[dict[str, Any]]) -> str:
    """Format phenopacket from aggregated results (legacy/fallback format).

    Args:
        aggregated_results: A list of dictionaries representing aggregated HPO
            term results with 'id', 'name', 'confidence', and 'rank' fields.

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

    return _create_phenopacket_json(phenopacket_id, phenotypic_features)


def _create_phenopacket_json(
    phenopacket_id: str,
    phenotypic_features: list[PhenotypicFeature],
) -> str:
    """Create the final Phenopacket JSON from components.

    Args:
        phenopacket_id: Unique identifier for the phenopacket.
        phenotypic_features: List of PhenotypicFeature objects.

    Returns:
        A JSON string representing the Phenopacket.
    """
    # Create MetaData
    created_timestamp = Timestamp()
    created_timestamp.FromDatetime(datetime.datetime.now(datetime.timezone.utc))

    meta_data = MetaData(
        created=created_timestamp,
        created_by="phentrieve",
        resources=[
            Resource(
                id="hp",
                name="human phenotype ontology",
                url="http://purl.obolibrary.org/obo/hp.owl",
                version="unknown",
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
