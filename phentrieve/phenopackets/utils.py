"""
This module provides utilities for creating and handling Phenopackets.
"""

import datetime
import uuid

from google.protobuf.json_format import MessageToJson
from google.protobuf.timestamp_pb2 import Timestamp

from phenopackets.schema.v2.core.base_pb2 import (
    OntologyClass,
    Evidence,
    ExternalReference,
)
from phenopackets.schema.v2.core.meta_data_pb2 import MetaData, Resource
from phenopackets.schema.v2.core.phenotypic_feature_pb2 import PhenotypicFeature
from phenopackets.schema.v2.phenopackets_pb2 import Phenopacket


def format_as_phenopacket_v2(aggregated_results: list[dict]) -> str:
    """
    Formats the aggregated results as a Phenopacket v2 JSON string.

    Args:
        aggregated_results: A list of dictionaries, where each dictionary
                            represents an aggregated HPO term result.

    Returns:
        A JSON string representing the Phenopacket.
    """
    if not aggregated_results:
        return "{}"

    phenopacket_id = f"phentrieve-phenopacket-{uuid.uuid4()}"

    # Sort results by rank
    aggregated_results.sort(key=lambda x: x.get("rank", 0))

    phenotypic_features = []
    for result in aggregated_results:
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
            evidence_code=OntologyClass(id="ECO:0007636", label="computational evidence used in automatic assertion"),
            reference=external_reference
        )

        # Create PhenotypicFeature
        phenotypic_feature = PhenotypicFeature(
            type=feature_type,
            evidence=[evidence],
        )
        phenotypic_features.append(phenotypic_feature)

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

    return MessageToJson(phenopacket, indent=2)
