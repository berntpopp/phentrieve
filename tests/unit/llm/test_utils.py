from __future__ import annotations

from phentrieve.llm.types import AssertionStatus
from phentrieve.llm.utils import (
    extract_hpo_id,
    extract_json,
    normalize_hpo_id,
    parse_assertion,
    token_sort_similarity,
)


def test_extract_json_handles_code_blocks_and_raw_json() -> None:
    code_block = """```json
    {"hpo_id": "HP:0001250"}
    ```"""
    raw = 'prefix {"hpo_id": "HP:0001250"} suffix'
    multiple_blocks = """```json
    not valid
    ```
    ```json
    {"hpo_id": "HP:0001250"}
    ```"""
    multiple_objects = 'prefix {"ignored": true} middle {"hpo_id": "HP:0001250"} suffix'

    assert extract_json(code_block) == {"hpo_id": "HP:0001250"}
    assert extract_json(raw) == {"hpo_id": "HP:0001250"}
    assert extract_json(multiple_blocks) == {"hpo_id": "HP:0001250"}
    assert extract_json(multiple_objects) == {"ignored": True}


def test_extract_hpo_id_and_normalization_helpers() -> None:
    assert extract_hpo_id("recurrent seizures (HP:0001250)") == "HP:0001250"
    assert extract_hpo_id("recurrent seizures (hp-1250)") == "HP:0001250"
    assert extract_hpo_id("recurrent seizures (HP 1250)") == "HP:0001250"
    assert normalize_hpo_id("hp-1250") == "HP:0001250"


def test_token_sort_similarity_ignores_word_order() -> None:
    assert token_sort_similarity("frequent falls", "falls frequent") == 100.0


def test_parse_assertion_maps_synonyms_to_canonical_status() -> None:
    assert parse_assertion("negative") == AssertionStatus.NEGATED
    assert parse_assertion("possible") == AssertionStatus.UNCERTAIN
    assert parse_assertion("present") == AssertionStatus.PRESENT
