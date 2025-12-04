import json
import unittest

from phentrieve.phenopackets.utils import format_as_phenopacket_v2


class TestPhenopacketUtils(unittest.TestCase):
    def test_format_as_phenopacket_v2_empty(self):
        """Test with empty aggregated results."""
        phenopacket_json = format_as_phenopacket_v2(aggregated_results=[])
        self.assertEqual(phenopacket_json, "{}")

    def test_format_as_phenopacket_v2_empty_both(self):
        """Test with no arguments."""
        phenopacket_json = format_as_phenopacket_v2()
        self.assertEqual(phenopacket_json, "{}")

    def test_format_as_phenopacket_v2_basic_aggregated(self):
        """Test basic phenopacket creation from aggregated results."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
            {"id": "HP:0001251", "name": "Absence seizure", "confidence": 0.7, "rank": 2},
            {"id": "HP:0002315", "name": "Headache", "confidence": 0.4, "rank": 3},
        ]
        phenopacket_json = format_as_phenopacket_v2(aggregated_results=aggregated_results)
        phenopacket = json.loads(phenopacket_json)

        self.assertIn("id", phenopacket)
        self.assertIn("phenotypicFeatures", phenopacket)
        self.assertIn("metaData", phenopacket)
        self.assertEqual(len(phenopacket["phenotypicFeatures"]), 3)

    def test_format_as_phenopacket_v2_sorting(self):
        """Test if features are sorted by rank."""
        aggregated_results = [
            {"id": "HP:0001251", "name": "Absence seizure", "confidence": 0.7, "rank": 2},
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
            {"id": "HP:0002315", "name": "Headache", "confidence": 0.4, "rank": 3},
        ]
        phenopacket_json = format_as_phenopacket_v2(aggregated_results=aggregated_results)
        phenopacket = json.loads(phenopacket_json)

        features = phenopacket["phenotypicFeatures"]
        self.assertEqual(features[0]["type"]["id"], "HP:0001250")
        self.assertEqual(features[1]["type"]["id"], "HP:0001251")
        self.assertEqual(features[2]["type"]["id"], "HP:0002315")

    def test_format_as_phenopacket_v2_evidence(self):
        """Test that evidence contains confidence and rank information."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
        ]
        phenopacket_json = format_as_phenopacket_v2(aggregated_results=aggregated_results)
        phenopacket = json.loads(phenopacket_json)

        features = phenopacket["phenotypicFeatures"]
        self.assertEqual(len(features), 1)

        evidence = features[0]["evidence"]
        self.assertEqual(len(evidence), 1)

        # Check evidence code
        self.assertEqual(evidence[0]["evidenceCode"]["id"], "ECO:0007636")
        self.assertIn("computational evidence", evidence[0]["evidenceCode"]["label"])

        # Check reference description contains confidence and rank
        description = evidence[0]["reference"]["description"]
        self.assertIn("0.9000", description)  # confidence
        self.assertIn("Rank: 1", description)

    def test_format_as_phenopacket_v2_chunk_results(self):
        """Test phenopacket creation from chunk results with text evidence."""
        chunk_results = [
            {
                "chunk_idx": 0,
                "chunk_text": "Patient has severe headaches",
                "matches": [
                    {"id": "HP:0002315", "name": "Headache", "score": 0.9, "assertion_status": "affirmed"},
                    {"id": "HP:0012228", "name": "Tension-type headache", "score": 0.7, "assertion_status": "affirmed"},
                ],
            },
            {
                "chunk_idx": 1,
                "chunk_text": "No muscle weakness observed",
                "matches": [
                    {"id": "HP:0001324", "name": "Muscle weakness", "score": 0.8, "assertion_status": "negated"},
                ],
            },
        ]
        phenopacket_json = format_as_phenopacket_v2(chunk_results=chunk_results)
        phenopacket = json.loads(phenopacket_json)

        self.assertIn("id", phenopacket)
        self.assertIn("phenotypicFeatures", phenopacket)
        self.assertEqual(len(phenopacket["phenotypicFeatures"]), 3)

        # Check first feature (from chunk 0)
        feature1 = phenopacket["phenotypicFeatures"][0]
        self.assertEqual(feature1["type"]["id"], "HP:0002315")
        self.assertEqual(feature1["type"]["label"], "Headache")
        self.assertNotIn("excluded", feature1)  # Not excluded (affirmed)
        description1 = feature1["evidence"][0]["reference"]["description"]
        self.assertIn("Patient has severe headaches", description1)
        self.assertIn("Chunk: 1", description1)
        self.assertNotIn("Rank:", description1)  # No rank in chunk-based results

        # Check second feature (from chunk 0)
        feature2 = phenopacket["phenotypicFeatures"][1]
        self.assertEqual(feature2["type"]["id"], "HP:0012228")
        description2 = feature2["evidence"][0]["reference"]["description"]
        self.assertIn("Chunk: 1", description2)

        # Check third feature (from chunk 1, negated)
        feature3 = phenopacket["phenotypicFeatures"][2]
        self.assertEqual(feature3["type"]["id"], "HP:0001324")
        self.assertTrue(feature3.get("excluded", False))  # Should be excluded (negated)
        description3 = feature3["evidence"][0]["reference"]["description"]
        self.assertIn("No muscle weakness observed", description3)
        self.assertIn("Chunk: 2", description3)

    def test_format_as_phenopacket_v2_metadata(self):
        """Test the metaData field structure."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
        ]
        phenopacket_json = format_as_phenopacket_v2(aggregated_results=aggregated_results)
        phenopacket = json.loads(phenopacket_json)

        meta = phenopacket["metaData"]
        self.assertIn("created", meta)
        self.assertEqual(meta["createdBy"], "phentrieve")
        self.assertEqual(meta["phenopacketSchemaVersion"], "2.0.2")

        # Check HPO resource
        resources = meta["resources"]
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0]["id"], "hp")
        self.assertEqual(resources[0]["namespacePrefix"], "HP")


if __name__ == "__main__":
    unittest.main()
