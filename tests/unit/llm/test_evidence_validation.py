from __future__ import annotations

import pytest

from phentrieve.llm.evidence_validation import validate_phase1_evidence

pytestmark = pytest.mark.unit


def test_validate_phase1_evidence_drops_unknown_chunk_id() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [99],
                "evidence_text": "recurrent seizures",
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept == []
    assert report.dropped == [
        {
            "phrase": "recurrent seizures",
            "reason": "unknown_chunk_id",
            "chunk_ids": [99],
        }
    ]


def test_validate_phase1_evidence_drops_empty_chunk_ids() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [],
                "evidence_text": "recurrent seizures",
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept == []
    assert report.dropped == [
        {
            "phrase": "recurrent seizures",
            "reason": "empty_chunk_ids",
            "chunk_ids": [],
        }
    ]


def test_validate_phase1_evidence_repairs_missing_evidence_from_phrase() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": None,
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept[0]["evidence_text"] == "recurrent seizures"
    assert report.repairs == [
        {"phrase": "recurrent seizures", "kind": "evidence_text_repair"}
    ]


def test_validate_phase1_evidence_repairs_offsets_from_exact_evidence() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "recurrent seizures",
                "start_char": 999,
                "end_char": 1009,
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept[0]["start_char"] == 12
    assert report.kept[0]["end_char"] == 30
    assert {"phrase": "recurrent seizures", "kind": "offset_repair"} in report.repairs


def test_validate_phase1_evidence_accepts_document_absolute_offsets_as_local() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "recurrent seizures",
                "start_char": 112,
                "end_char": 130,
            }
        ],
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Patient had recurrent seizures.",
                "start_char": 100,
                "end_char": 132,
            }
        ],
    )

    assert report.kept[0]["start_char"] == 12
    assert report.kept[0]["end_char"] == 30
    assert {
        "phrase": "recurrent seizures",
        "kind": "offset_coordinate_repair",
    } in report.repairs


def test_validate_phase1_evidence_downgrades_multichunk_offsets() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [1, 2],
                "evidence_text": "Patient had recurrent seizures",
                "start_char": 0,
                "end_char": 30,
            }
        ],
        grounded_chunks=[
            {"chunk_id": 1, "text": "Patient had recurrent"},
            {"chunk_id": 2, "text": "seizures."},
        ],
    )

    assert report.kept[0]["start_char"] is None
    assert report.kept[0]["end_char"] is None
    assert report.repairs == [
        {"phrase": "recurrent seizures", "kind": "multi_chunk_offset_downgrade"}
    ]


def test_validate_phase1_evidence_downgrades_fuzzy_evidence_to_chunk_level() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizure",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "recurrent seizure",
                "start_char": 12,
                "end_char": 29,
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
        fuzzy_threshold=80.0,
    )

    assert report.kept[0]["start_char"] is None
    assert report.kept[0]["end_char"] is None
    assert report.kept[0]["evidence_text"] == "recurrent seizure"
    assert report.repairs == [
        {"phrase": "recurrent seizure", "kind": "fuzzy_evidence_downgrade"}
    ]


def test_validate_phase1_evidence_accepts_camel_case_list_boundary() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "Ptosis",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "Ptosis",
            },
            {
                "phrase": "Ophthalmoplegia",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "Ophthalmoplegia",
            },
            {
                "phrase": "Corneal ulcers",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "Corneal ulcers",
            },
        ],
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Other findings include the following:PtosisOphthalmoplegiaCorneal ulcers.",
            }
        ],
    )

    assert [item["phrase"] for item in report.kept] == [
        "Ptosis",
        "Ophthalmoplegia",
        "Corneal ulcers",
    ]
    assert report.dropped == []


def test_validate_phase1_evidence_drops_ungrounded_evidence() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "invented ataxia",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "invented ataxia",
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept == []
    assert report.dropped == [
        {
            "phrase": "invented ataxia",
            "reason": "evidence_not_grounded",
            "chunk_ids": [1],
        }
    ]


def test_validate_phase1_evidence_skips_ungrounded_legacy_inputs() -> None:
    original = [
        {
            "phrase": "recurrent seizures",
            "category": "Abnormal",
            "chunk_ids": [],
            "evidence_text": None,
        }
    ]

    report = validate_phase1_evidence(extracted=original, grounded_chunks=[])

    assert report.status == "skipped_no_grounded_chunks"
    assert report.kept == original
    assert report.dropped == []
    assert report.repairs == []
    assert report.kept is not original
