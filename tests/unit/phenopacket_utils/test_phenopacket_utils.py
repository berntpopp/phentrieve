import json

import pytest
from google.protobuf.json_format import Parse
from phenopackets import Phenopacket

from phentrieve.phenopackets.export_models import (
    NormalizedPhenotypeExportRecord,
    NormalizedSpan,
)
from phentrieve.phenopackets.utils import (
    _normalize_aggregated_results,
    format_as_phenopacket_v2,
)

pytestmark = pytest.mark.unit


class TestPhenopacketUtils:
    def test_format_as_phenopacket_v2_empty(self):
        """Test with empty aggregated results returns valid Phenopacket."""
        phenopacket_json = format_as_phenopacket_v2(aggregated_results=[])
        phenopacket = json.loads(phenopacket_json)
        # Should return valid Phenopacket structure with no features
        assert "id" in phenopacket
        assert "metaData" in phenopacket
        assert len(phenopacket.get("phenotypicFeatures", [])) == 0

    def test_format_as_phenopacket_v2_empty_both(self):
        """Test with no arguments returns valid Phenopacket."""
        phenopacket_json = format_as_phenopacket_v2()
        phenopacket = json.loads(phenopacket_json)
        # Should return valid Phenopacket structure with no features
        assert "id" in phenopacket
        assert "metaData" in phenopacket
        assert len(phenopacket.get("phenotypicFeatures", [])) == 0

    def test_format_as_phenopacket_v2_basic_aggregated(self):
        """Test basic phenopacket creation from aggregated results."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
            {
                "id": "HP:0001251",
                "name": "Absence seizure",
                "confidence": 0.7,
                "rank": 2,
            },
            {"id": "HP:0002315", "name": "Headache", "confidence": 0.4, "rank": 3},
        ]
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated_results
        )
        phenopacket = json.loads(phenopacket_json)

        assert "id" in phenopacket
        assert "phenotypicFeatures" in phenopacket
        assert "metaData" in phenopacket
        assert len(phenopacket["phenotypicFeatures"]) == 3

    def test_format_as_phenopacket_v2_sorting(self):
        """Test if features are sorted by rank."""
        aggregated_results = [
            {
                "id": "HP:0001251",
                "name": "Absence seizure",
                "confidence": 0.7,
                "rank": 2,
            },
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
            {"id": "HP:0002315", "name": "Headache", "confidence": 0.4, "rank": 3},
        ]
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated_results
        )
        phenopacket = json.loads(phenopacket_json)

        features = phenopacket["phenotypicFeatures"]
        assert features[0]["type"]["id"] == "HP:0001250"
        assert features[1]["type"]["id"] == "HP:0001251"
        assert features[2]["type"]["id"] == "HP:0002315"

    def test_format_as_phenopacket_v2_evidence(self):
        """Test that evidence contains confidence and rank information."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
        ]
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated_results
        )
        phenopacket = json.loads(phenopacket_json)

        features = phenopacket["phenotypicFeatures"]
        assert len(features) == 1

        evidence = features[0]["evidence"]
        assert len(evidence) == 1

        # Check evidence code
        assert evidence[0]["evidenceCode"]["id"] == "ECO:0007636"
        assert "computational evidence" in evidence[0]["evidenceCode"]["label"]

        # Check reference description contains confidence and rank
        description = evidence[0]["reference"]["description"]
        assert "0.9000" in description  # confidence
        assert "Rank: 1" in description

    def test_format_as_phenopacket_v2_chunk_results(self):
        """Test phenopacket creation from chunk results with text evidence."""
        chunk_results = [
            {
                "chunk_idx": 0,
                "chunk_text": "Patient has severe headaches",
                "matches": [
                    {
                        "id": "HP:0002315",
                        "name": "Headache",
                        "score": 0.9,
                        "assertion_status": "affirmed",
                    },
                    {
                        "id": "HP:0012228",
                        "name": "Tension-type headache",
                        "score": 0.7,
                        "assertion_status": "affirmed",
                    },
                ],
            },
            {
                "chunk_idx": 1,
                "chunk_text": "No muscle weakness observed",
                "matches": [
                    {
                        "id": "HP:0001324",
                        "name": "Muscle weakness",
                        "score": 0.8,
                        "assertion_status": "negated",
                    },
                ],
            },
        ]
        phenopacket_json = format_as_phenopacket_v2(chunk_results=chunk_results)
        phenopacket = json.loads(phenopacket_json)

        assert "id" in phenopacket
        assert "phenotypicFeatures" in phenopacket
        assert len(phenopacket["phenotypicFeatures"]) == 3

        # Check first feature (from chunk 0)
        feature1 = phenopacket["phenotypicFeatures"][0]
        assert feature1["type"]["id"] == "HP:0002315"
        assert feature1["type"]["label"] == "Headache"
        assert "excluded" not in feature1  # Not excluded (affirmed)
        description1 = feature1["evidence"][0]["reference"]["description"]
        assert "Patient has severe headaches" in description1
        assert "Chunk: 1" in description1
        assert "Rank:" not in description1  # No rank in chunk-based results

        # Check second feature (from chunk 0)
        feature2 = phenopacket["phenotypicFeatures"][1]
        assert feature2["type"]["id"] == "HP:0012228"
        description2 = feature2["evidence"][0]["reference"]["description"]
        assert "Chunk: 1" in description2

        # Check third feature (from chunk 1, negated)
        feature3 = phenopacket["phenotypicFeatures"][2]
        assert feature3["type"]["id"] == "HP:0001324"
        assert feature3.get("excluded", False) is True  # Should be excluded (negated)
        description3 = feature3["evidence"][0]["reference"]["description"]
        assert "No muscle weakness observed" in description3
        assert "Chunk: 2" in description3

    def test_format_as_phenopacket_v2_metadata(self):
        """Test the metaData field structure with version information."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
        ]
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated_results
        )
        phenopacket = json.loads(phenopacket_json)

        meta = phenopacket["metaData"]
        assert "created" in meta
        # createdBy should now include version
        assert "phentrieve" in meta["createdBy"]
        assert meta["phenopacketSchemaVersion"] == "2.0"

        # Check HPO resource
        resources = meta["resources"]
        assert len(resources) == 1
        assert resources[0]["id"] == "hp"
        assert resources[0]["namespacePrefix"] == "HP"

    def test_format_as_phenopacket_v2_with_metadata_parameters(self):
        """Test phenopacket with embedding model metadata."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
        ]
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated_results,
            phentrieve_version="0.3.0",
            embedding_model="BAAI/bge-m3",
            hpo_version="v2025-03-03",
        )
        phenopacket = json.loads(phenopacket_json)

        meta = phenopacket["metaData"]
        # Check createdBy includes version
        assert meta["createdBy"] == "phentrieve 0.3.0"

        # Check HPO version
        assert meta["resources"][0]["version"] == "v2025-03-03"

        # Check external references for model metadata
        assert "externalReferences" in meta
        ext_refs = meta["externalReferences"]
        assert len(ext_refs) == 1

        # Find embedding reference
        refs_by_id = {ref["id"]: ref["description"] for ref in ext_refs}
        assert refs_by_id["phentrieve:embedding_model"] == "BAAI/bge-m3"

    def test_phenopacket_export_uses_verified_v2_schema_string(self):
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=[
                {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1}
            ]
        )
        phenopacket = json.loads(phenopacket_json)

        assert phenopacket["metaData"]["phenopacketSchemaVersion"] == "2.0"

    def test_phenopacket_export_round_trips_through_protobuf_parser(self):
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=[
                {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1}
            ]
        )

        packet = Phenopacket()
        Parse(phenopacket_json, packet, ignore_unknown_fields=False)

        assert packet.id

    def test_negated_assertion_maps_to_excluded_true(self):
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=[
                {
                    "hpo_id": "HP:0001324",
                    "term_name": "Muscle weakness",
                    "assertion": "negated",
                    "score": 0.8,
                }
            ]
        )
        phenopacket = json.loads(phenopacket_json)

        assert phenopacket["phenotypicFeatures"][0]["excluded"] is True


class TestNormalizedExportModels:
    def test_normalized_span_direct_construction_and_legacy_dict_constructor(self):
        span = NormalizedSpan(
            text="recurrent seizures",
            start_char=10,
            end_char=28,
            chunk_ids=[4],
        )

        assert span.evidence_text == "recurrent seizures"
        assert span.start_char == 10
        assert span.end_char == 28
        assert span.chunk_ids == [4]

        legacy_span = NormalizedSpan.from_legacy_dict(
            {
                "text": "recurrent seizures",
                "start_char": 10,
                "end_char": 28,
                "chunk_ids": [4],
            }
        )

        assert legacy_span == span

    def test_normalized_phenotype_export_record_direct_construction_and_legacy_dict_constructor(
        self,
    ):
        span = NormalizedSpan(
            text="recurrent seizures",
            start_char=10,
            end_char=28,
            chunk_ids=[4],
        )

        record = NormalizedPhenotypeExportRecord(
            hpo_id="HP:0001250",
            label="Seizure",
            assertion="affirmed",
            confidence=0.91,
            spans=[span],
            evidence_text="recurrent seizures",
            chunk_refs=[4],
            source_mode="two_phase",
            match_method="llm_mapping",
        )

        assert record.hpo_id == "HP:0001250"
        assert record.label == "Seizure"
        assert record.assertion == "affirmed"
        assert record.confidence == 0.91
        assert record.spans == [span]
        assert record.evidence_text == "recurrent seizures"
        assert record.chunk_refs == [4]
        assert record.source_mode == "two_phase"
        assert record.match_method == "llm_mapping"
        assert record.sidecar_linkage_key

        identical_record = NormalizedPhenotypeExportRecord(
            hpo_id="HP:0001250",
            label="Seizure",
            assertion="affirmed",
            confidence=0.12,
            spans=[
                NormalizedSpan(
                    text="expanded recurrent seizures",
                    start_char=0,
                    end_char=27,
                    chunk_ids=[4],
                )
            ],
            evidence_text="expanded recurrent seizures",
            chunk_refs=[4],
            source_mode="chunk",
            match_method="different",
        )

        assert identical_record.sidecar_linkage_key == record.sidecar_linkage_key

        aggregated_legacy = NormalizedPhenotypeExportRecord.from_legacy_dict(
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1}
        )

        assert aggregated_legacy.hpo_id == "HP:0001250"
        assert aggregated_legacy.label == "Seizure"
        assert aggregated_legacy.assertion == "affirmed"
        assert aggregated_legacy.confidence == 0.9
        assert aggregated_legacy.spans == []
        assert aggregated_legacy.chunk_refs == []
        assert aggregated_legacy.source_mode == "aggregated"
        assert aggregated_legacy.match_method == "legacy_dict"
        assert aggregated_legacy.sidecar_linkage_key

        chunk_legacy = NormalizedPhenotypeExportRecord.from_legacy_dict(
            {
                "hpo_id": "HP:0001324",
                "term_name": "Muscle weakness",
                "score": 0.8,
                "assertion_status": "negated",
                "evidence_text": "No muscle weakness observed",
                "chunk_refs": [2],
            }
        )

        assert chunk_legacy.hpo_id == "HP:0001324"
        assert chunk_legacy.label == "Muscle weakness"
        assert chunk_legacy.assertion == "negated"
        assert chunk_legacy.confidence == 0.8
        assert chunk_legacy.evidence_text == "No muscle weakness observed"
        assert chunk_legacy.chunk_refs == [2]
        assert chunk_legacy.source_mode == "chunk"
        assert chunk_legacy.match_method == "legacy_dict"
        assert chunk_legacy.sidecar_linkage_key

    def test_normalize_aggregated_results_accepts_legacy_id_name_confidence_keys(
        self,
    ):
        records = _normalize_aggregated_results(
            [
                {
                    "id": "HP:0001250",
                    "name": "Seizure",
                    "confidence": 0.9,
                    "rank": 1,
                }
            ]
        )

        assert records[0].hpo_id == "HP:0001250"
        assert records[0].label == "Seizure"
        assert records[0].confidence == 0.9

    def test_normalize_aggregated_results_accepts_llm_style_hpo_id_term_name_score_keys(
        self,
    ):
        records = _normalize_aggregated_results(
            [
                {
                    "hpo_id": "HP:0001250",
                    "term_name": "Seizure",
                    "score": 0.8,
                    "assertion": "affirmed",
                    "evidence_text": "recurrent seizures",
                }
            ]
        )

        assert records[0].hpo_id == "HP:0001250"
        assert records[0].label == "Seizure"
        assert records[0].assertion == "affirmed"
        assert records[0].evidence_text == "recurrent seizures"
