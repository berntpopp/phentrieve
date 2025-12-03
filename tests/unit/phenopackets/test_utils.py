
import json
import unittest
from phentrieve.phenopackets.utils import format_as_phenopacket_v2

class TestPhenopacketUtils(unittest.TestCase):

    def test_format_as_phenopacket_v2_empty(self):
        """Test with empty aggregated results."""
        phenopacket_json = format_as_phenopacket_v2([])
        self.assertEqual(phenopacket_json, "{}")

    def test_format_as_phenopacket_v2_basic(self):
        """Test basic phenopacket creation."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
            {"id": "HP:0001251", "name": "Absence seizure", "confidence": 0.7, "rank": 2},
            {"id": "HP:0002315", "name": "Headache", "confidence": 0.4, "rank": 3},
        ]
        phenopacket_json = format_as_phenopacket_v2(aggregated_results)
        phenopacket = json.loads(phenopacket_json)

        self.assertIn("id", phenopacket)
        self.assertIn("phenotypic_features", phenopacket)
        self.assertIn("meta_data", phenopacket)
        self.assertEqual(len(phenopacket["phenotypic_features"]), 3)

    def test_format_as_phenopacket_v2_sorting(self):
        """Test if features are sorted by rank."""
        aggregated_results = [
            {"id": "HP:0001251", "name": "Absence seizure", "confidence": 0.7, "rank": 2},
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
            {"id": "HP:0002315", "name": "Headache", "confidence": 0.4, "rank": 3},
        ]
        phenopacket_json = format_as_phenopacket_v2(aggregated_results)
        phenopacket = json.loads(phenopacket_json)

        features = phenopacket["phenotypic_features"]
        self.assertEqual(features[0]["type"]["id"], "HP:0001250")
        self.assertEqual(features[1]["type"]["id"], "HP:0001251")
        self.assertEqual(features[2]["type"]["id"], "HP:0002315")
        self.assertEqual(features[0]["_phentrieve_meta"]["rank"], 1)
        self.assertEqual(features[1]["_phentrieve_meta"]["rank"], 2)
        self.assertEqual(features[2]["_phentrieve_meta"]["rank"], 3)

    def test_format_as_phenopacket_v2_meta(self):
        """Test the _phentrieve_meta field."""
        aggregated_results = [
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1},
            {"id": "HP:0001251", "name": "Absence seizure", "confidence": 0.7, "rank": 2},
            {"id": "HP:0002315", "name": "Headache", "confidence": 0.4, "rank": 3},
        ]
        phenopacket_json = format_as_phenopacket_v2(aggregated_results)
        phenopacket = json.loads(phenopacket_json)
        
        features = phenopacket["phenotypic_features"]
        
        # Test feature 1 (HIGH)
        meta1 = features[0]["_phentrieve_meta"]
        self.assertEqual(meta1["rank"], 1)
        self.assertEqual(meta1["confidence"], 0.9)
        self.assertEqual(meta1["confidence_level"], "HIGH")
        self.assertAlmostEqual(meta1["abstract_distance"], 0.1)

        # Test feature 2 (MEDIUM)
        meta2 = features[1]["_phentrieve_meta"]
        self.assertEqual(meta2["rank"], 2)
        self.assertEqual(meta2["confidence"], 0.7)
        self.assertEqual(meta2["confidence_level"], "MEDIUM")
        self.assertAlmostEqual(meta2["abstract_distance"], 0.3)

        # Test feature 3 (LOW)
        meta3 = features[2]["_phentrieve_meta"]
        self.assertEqual(meta3["rank"], 3)
        self.assertEqual(meta3["confidence"], 0.4)
        self.assertEqual(meta3["confidence_level"], "LOW")
        self.assertAlmostEqual(meta3["abstract_distance"], 0.6)

if __name__ == '__main__':
    unittest.main()
