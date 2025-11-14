"""
Test data loading module for benchmark evaluation.

This module provides functionality for loading and creating test cases
used in benchmark evaluation of the HPO retrieval system.
"""

import json
import logging
import os
from typing import Any, Optional

from phentrieve.config import DEFAULT_TEST_CASES_SUBDIR
from phentrieve.utils import get_default_data_dir


def load_test_data(test_file: str) -> Optional[list[dict[str, Any]]]:
    """
    Load test cases from a JSON file.

    Expected format:
    [
        {
            "text": "Clinical text in the target language",
            "expected_hpo_ids": ["HP:0000123", "HP:0000456"],
            "description": "Optional description of the case"
        },
        ...
    ]

    Args:
        test_file: Path to the test cases JSON file

    Returns:
        List of test case dictionaries, or None if loading fails
    """
    try:
        with open(test_file, encoding="utf-8") as f:
            test_cases = json.load(f)

        logging.info(f"Loaded {len(test_cases)} test cases from {test_file}")
        return test_cases
    except Exception as e:
        logging.error(f"Error loading test data from {test_file}: {e}")
        return None


def create_sample_test_data(output_file: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Create a sample test dataset if none exists.

    Args:
        output_file: Optional file path to save the sample test data

    Returns:
        List of sample test cases
    """
    # Get the test cases directory from data_dir
    data_dir = get_default_data_dir()
    test_cases_dir = data_dir / DEFAULT_TEST_CASES_SUBDIR

    # Create directory if it doesn't exist
    os.makedirs(test_cases_dir, exist_ok=True)

    if output_file is None:
        output_file = str(test_cases_dir / "sample_test_cases.json")

    # Sample test cases
    sample_test_cases = [
        {
            "description": "Hypertrophic cardiomyopathy with septal hypertrophy",
            "text": "Hypertrophe Kardiomyopathie mit Septumhypertrophie",
            "expected_hpo_ids": ["HP:0001639", "HP:0001712"],
        },
        {
            "description": "Intellectual disability and tremor",
            "text": "Geistige Behinderung und Tremor",
            "expected_hpo_ids": ["HP:0001249", "HP:0001337"],
        },
        {
            "description": "Polydactyly of toes and hands",
            "text": "Polydaktylie der Zehen und Hände",
            "expected_hpo_ids": ["HP:0001829", "HP:0001161"],
        },
        {
            "description": "Down syndrome with heart defect",
            "text": "Down-Syndrom mit Herzfehler",
            "expected_hpo_ids": ["HP:0000598", "HP:0001627"],
        },
        {
            "description": "Epilepsy with seizures",
            "text": "Epilepsie mit Anfällen",
            "expected_hpo_ids": ["HP:0001250", "HP:0001251"],
        },
        {
            "description": "Severe muscular atrophy with hypotonia",
            "text": "Schwere Muskelatrophie mit Hypotonie",
            "expected_hpo_ids": ["HP:0003202", "HP:0001252"],
        },
        {
            "description": "Hearing loss with inner ear malformation",
            "text": "Hörverlust mit Innenohrfehlbildung",
            "expected_hpo_ids": ["HP:0000365", "HP:0011389"],
        },
        {
            "description": "Neutropenia with recurrent infections",
            "text": "Neutropenie mit wiederkehrenden Infektionen",
            "expected_hpo_ids": ["HP:0001875", "HP:0002719"],
        },
        {
            "description": "Microcephaly and developmental delay",
            "text": "Mikrozephalie und Entwicklungsverzögerung",
            "expected_hpo_ids": ["HP:0000252", "HP:0001263"],
        },
    ]

    # Save the sample test cases to a file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_test_cases, f, indent=2, ensure_ascii=False)

    logging.info(f"Created sample test cases file: {output_file}")
    return sample_test_cases
