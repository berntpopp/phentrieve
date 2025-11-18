"""
Unit tests for annotation_chunker.py.

Tests the core Voronoi boundary algorithm and chunk generation logic.
"""

import pytest
from annotation_chunker import (
    CHUNKER_VERSION,
    DEFAULT_EXPANSION_RATIOS,
    align_to_word_boundaries,
    compute_voronoi_boundaries,
    generate_chunk_variants,
    validate_annotations,
)

# ============================================================================
# Tests for align_to_word_boundaries
# ============================================================================


def test_align_to_word_boundaries_mid_word():
    """Aligning mid-word should expand to full word."""
    text = "The quick brown fox"
    # Start in middle of "quick" (position 6 = 'u')
    # End in middle of "brown" (position 12 = 'o')
    start, end = align_to_word_boundaries(text, 6, 12)
    assert text[start:end] == "quick brown "


def test_align_to_word_boundaries_already_aligned():
    """Already aligned boundaries should stay the same."""
    text = "The quick brown fox"
    start, end = align_to_word_boundaries(text, 4, 9)  # "quick"
    assert text[start:end] == "quick "


def test_align_to_word_boundaries_with_punctuation():
    """Punctuation should be handled correctly."""
    text = "Hello, world! How are you?"
    start, end = align_to_word_boundaries(text, 3, 10)  # "lo, wo"
    assert text[start:end] == "Hello, world!"


def test_align_to_word_boundaries_sentence_end():
    """Sentence boundaries should be respected."""
    text = "First sentence. Second sentence."
    start, end = align_to_word_boundaries(text, 6, 20)  # "senten... econd s"
    # Should stop at sentence boundary
    extracted = text[start:end]
    assert extracted.startswith("sentence.")
    assert "Second" in extracted


def test_align_to_word_boundaries_start_of_text():
    """Alignment at start of text should work."""
    text = "The quick brown"
    start, end = align_to_word_boundaries(text, 0, 5)
    assert text[start:end] == "The quick "


def test_align_to_word_boundaries_end_of_text():
    """Alignment at end of text should work."""
    text = "The quick brown"
    start, end = align_to_word_boundaries(text, 10, 15)
    assert text[start:end] == "brown"


def test_align_to_word_boundaries_beyond_bounds():
    """Positions beyond text bounds should be clamped."""
    text = "Short"
    start, end = align_to_word_boundaries(text, -5, 100)
    assert text[start:end] == "Short"


# ============================================================================
# Tests for compute_voronoi_boundaries
# ============================================================================


def test_voronoi_boundaries_empty():
    """Empty annotations should return empty boundaries."""
    boundaries = compute_voronoi_boundaries([], text_length=100)
    assert boundaries == []


def test_voronoi_boundaries_single_annotation():
    """Single annotation should get full text as territory."""
    annotations = [{"evidence_spans": [{"start_char": 10, "end_char": 20}]}]
    boundaries = compute_voronoi_boundaries(annotations, text_length=100)
    assert boundaries == [(0, 100)]


def test_voronoi_boundaries_two_annotations():
    """Two annotations should split at midpoint."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 20}]},
        {"evidence_spans": [{"start_char": 50, "end_char": 60}]},
    ]
    boundaries = compute_voronoi_boundaries(annotations, text_length=100)
    # Midpoint between 20 and 50 is (20 + 50) // 2 = 35
    assert boundaries == [(0, 35), (35, 100)]


def test_voronoi_boundaries_three_annotations():
    """Three annotations should create three territories."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 20}]},
        {"evidence_spans": [{"start_char": 40, "end_char": 50}]},
        {"evidence_spans": [{"start_char": 70, "end_char": 80}]},
    ]
    boundaries = compute_voronoi_boundaries(annotations, text_length=100)
    # Midpoints: (20+40)//2=30, (50+70)//2=60
    assert boundaries == [(0, 30), (30, 60), (60, 100)]


def test_voronoi_boundaries_unsorted_annotations():
    """Annotations should be sorted by position automatically."""
    annotations = [
        {"evidence_spans": [{"start_char": 50, "end_char": 60}]},
        {"evidence_spans": [{"start_char": 10, "end_char": 20}]},
    ]
    boundaries = compute_voronoi_boundaries(annotations, text_length=100)
    # Should be sorted: first (10-20), second (50-60)
    # Midpoint: (20+50)//2 = 35
    assert boundaries == [(0, 35), (35, 100)]


def test_voronoi_boundaries_adjacent_annotations():
    """Adjacent annotations (no gap) should split exactly at boundary."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 20}]},
        {
            "evidence_spans": [{"start_char": 20, "end_char": 30}]
        },  # Starts where prev ends
    ]
    boundaries = compute_voronoi_boundaries(annotations, text_length=100)
    # Midpoint: (20+20)//2 = 20
    assert boundaries == [(0, 20), (20, 100)]


# ============================================================================
# Tests for generate_chunk_variants
# ============================================================================


def test_generate_chunk_variants_single_annotation():
    """Generate chunks for single annotation with default ratios."""
    doc = {
        "full_text": "A syndrome of brachydactyly and other features.",
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "label": "Brachydactyly",
                "assertion_status": "affirmed",
                "evidence_spans": [
                    {"start_char": 14, "end_char": 27, "text_snippet": "brachydactyly"}
                ],
            }
        ],
    }

    result = generate_chunk_variants(doc)

    # Check structure
    assert "provenance" in result
    assert "chunks" in result
    assert len(result["chunks"]) == 1

    # Check provenance
    provenance = result["provenance"]
    assert provenance["script"] == "generate_chunking_variants.py"
    assert provenance["script_version"] == CHUNKER_VERSION
    assert "generated_at" in provenance
    assert provenance["parameters"]["strategy"] == "voronoi_midpoint"
    assert provenance["parameters"]["expansion_ratios"] == DEFAULT_EXPANSION_RATIOS

    # Check chunk
    chunk = result["chunks"][0]
    assert chunk["hpo_id"] == "HP:0001156"
    assert chunk["annotation_span"] == [14, 27]
    assert "variants" in chunk

    # Check variants
    assert "0.00" in chunk["variants"]
    assert "0.50" in chunk["variants"]
    assert "1.00" in chunk["variants"]

    # 0.0 should be just the annotation (with word alignment)
    assert chunk["variants"]["0.00"]["text"] == "brachydactyly "
    assert chunk["variants"]["0.00"]["span"] == [14, 28]

    # 1.0 should be full territory (entire text for single annotation)
    assert chunk["variants"]["1.00"]["text"] == doc["full_text"]
    assert chunk["variants"]["1.00"]["span"] == [0, len(doc["full_text"])]


def test_generate_chunk_variants_two_annotations():
    """Generate chunks for two annotations."""
    doc = {
        "full_text": "Patient has brachydactyly and hypotonia.",
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "label": "Brachydactyly",
                "assertion_status": "affirmed",
                "evidence_spans": [
                    {"start_char": 12, "end_char": 25, "text_snippet": "brachydactyly"}
                ],
            },
            {
                "hpo_id": "HP:0001252",
                "label": "Hypotonia",
                "assertion_status": "affirmed",
                "evidence_spans": [
                    {"start_char": 30, "end_char": 39, "text_snippet": "hypotonia"}
                ],
            },
        ],
    }

    result = generate_chunk_variants(doc)

    assert len(result["chunks"]) == 2

    # Check first chunk
    chunk1 = result["chunks"][0]
    assert chunk1["hpo_id"] == "HP:0001156"
    assert chunk1["variants"]["0.00"]["text"] == "brachydactyly "

    # Check second chunk
    chunk2 = result["chunks"][1]
    assert chunk2["hpo_id"] == "HP:0001252"
    assert chunk2["variants"]["0.00"]["text"] == "hypotonia."


def test_generate_chunk_variants_custom_ratios():
    """Custom expansion ratios should be used."""
    doc = {
        "full_text": "Patient has brachydactyly.",
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "evidence_spans": [{"start_char": 12, "end_char": 25}],
            }
        ],
    }

    result = generate_chunk_variants(doc, expansion_ratios=[0.0, 0.25, 0.75, 1.0])

    chunk = result["chunks"][0]
    assert "0.00" in chunk["variants"]
    assert "0.25" in chunk["variants"]
    assert "0.75" in chunk["variants"]
    assert "1.00" in chunk["variants"]
    assert "0.50" not in chunk["variants"]  # Not in custom list


def test_generate_chunk_variants_expansion_calculation():
    """Test exact expansion calculation for 0.5 ratio with word alignment."""
    # Controlled example with known boundaries
    # Text: "AAAA BBBB CCCC" (14 chars)
    # Ann:  ----^^^^^----- (5-10)
    # Territory: 0 to 14 (single annotation)
    # Left space: 5 - 0 = 5
    # Right space: 14 - 10 = 4
    # 0.5 expansion: left=2, right=2
    # Chunk before alignment: 3 to 12 ("A BBBB C")
    # After word alignment: 0 to 14 (whole text)
    doc = {
        "full_text": "AAAA BBBB CCCC",
        "annotations": [
            {
                "hpo_id": "HP:0000001",
                "evidence_spans": [{"start_char": 5, "end_char": 10}],
            }
        ],
    }

    result = generate_chunk_variants(doc, expansion_ratios=[0.5])

    chunk = result["chunks"][0]
    variant = chunk["variants"]["0.50"]

    # Word alignment will expand to full words
    # Starting mid-word will expand to include whole words
    assert variant["span"] == [0, 14]
    assert variant["text"] == "AAAA BBBB CCCC"


def test_generate_chunk_variants_idempotency():
    """Running twice should produce identical chunks (except timestamp)."""
    doc = {
        "full_text": "Test text with annotation.",
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "evidence_spans": [{"start_char": 15, "end_char": 25}],
            }
        ],
    }

    result1 = generate_chunk_variants(doc)
    result2 = generate_chunk_variants(doc)

    # Chunks should be identical
    # Chunks should be identical (word alignment is deterministic)
    assert result1["chunks"] == result2["chunks"]

    # Provenance parameters should be identical (except timestamp)
    assert (
        result1["provenance"]["script_version"]
        == result2["provenance"]["script_version"]
    )
    assert result1["provenance"]["parameters"] == result2["provenance"]["parameters"]


# ============================================================================
# Tests for validate_annotations
# ============================================================================


def test_validate_annotations_valid():
    """Valid annotations should not raise error."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 20}]},
        {"evidence_spans": [{"start_char": 30, "end_char": 40}]},
    ]
    # Should not raise
    validate_annotations(annotations)


def test_validate_annotations_invalid_span_start_equals_end():
    """Span where start == end should raise ValueError."""
    annotations = [{"evidence_spans": [{"start_char": 10, "end_char": 10}]}]

    with pytest.raises(ValueError, match="Invalid span"):
        validate_annotations(annotations)


def test_validate_annotations_invalid_span_start_greater_than_end():
    """Span where start > end should raise ValueError."""
    annotations = [{"evidence_spans": [{"start_char": 20, "end_char": 10}]}]

    with pytest.raises(ValueError, match="Invalid span"):
        validate_annotations(annotations)


def test_validate_annotations_negative_position():
    """Negative positions should raise ValueError."""
    annotations = [{"evidence_spans": [{"start_char": -5, "end_char": 10}]}]

    with pytest.raises(ValueError, match="Negative position"):
        validate_annotations(annotations)


def test_validate_annotations_overlapping():
    """Overlapping annotations should raise ValueError."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 25}]},
        {"evidence_spans": [{"start_char": 20, "end_char": 30}]},  # Overlaps!
    ]

    with pytest.raises(ValueError, match="Overlapping"):
        validate_annotations(annotations)


def test_validate_annotations_touching_ok():
    """Adjacent (touching but not overlapping) annotations are valid."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 20}]},
        {"evidence_spans": [{"start_char": 20, "end_char": 30}]},  # Touches at 20
    ]
    # Should not raise
    validate_annotations(annotations)


def test_validate_annotations_multiple_spans_per_annotation():
    """Each span in multi-span annotations should be validated."""
    annotations = [
        {
            "evidence_spans": [
                {"start_char": 10, "end_char": 20},
                {"start_char": 30, "end_char": 40},
            ]
        }
    ]
    # Should not raise
    validate_annotations(annotations)


def test_validate_annotations_overlap_across_different_annotations():
    """Overlaps between different annotations' spans should be detected."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 25}]},
        {"evidence_spans": [{"start_char": 22, "end_char": 30}]},
    ]

    with pytest.raises(ValueError, match="Overlapping"):
        validate_annotations(annotations)


# ============================================================================
# Integration/Edge Cases
# ============================================================================


def test_real_example_gsc_plus():
    """Test with real example from GSC+ dataset."""
    doc = {
        "doc_id": "GSC+_1003450",
        "language": "en",
        "source": "phenobert",
        "full_text": "A syndrome of brachydactyly (absence of some middle or distal phalanges), "
        "aplastic or hypoplastic nails, symphalangism (ankylois of proximal "
        "interphalangeal joints), synostosis of some carpal and tarsal bones, "
        "craniosynostosis, and dysplastic hip joints is reported in five members "
        "of an Italian family.",
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "label": "Brachydactyly",
                "assertion_status": "affirmed",
                "evidence_spans": [{"start_char": 14, "end_char": 27}],
            },
            {
                "hpo_id": "HP:0009881",
                "label": "Aplasia of the distal phalanges of the hand",
                "assertion_status": "affirmed",
                "evidence_spans": [{"start_char": 29, "end_char": 71}],
            },
        ],
    }

    result = generate_chunk_variants(doc)

    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["hpo_id"] == "HP:0001156"
    assert result["chunks"][1]["hpo_id"] == "HP:0009881"

    # All variants should have text
    for chunk in result["chunks"]:
        for variant_key in ["0.00", "0.50", "1.00"]:
            assert len(chunk["variants"][variant_key]["text"]) > 0


def test_german_example():
    """Test with German clinical text."""
    doc = {
        "full_text": "Kind vorgestellt mit Trinkschwäche und Hypotonie.",
        "annotations": [
            {
                "hpo_id": "HP:0030082",
                "label": "Abnormal drinking behavior",
                "assertion_status": "affirmed",
                "evidence_spans": [{"start_char": 21, "end_char": 34}],
            },
            {
                "hpo_id": "HP:0001252",
                "label": "Hypotonia",
                "assertion_status": "affirmed",
                "evidence_spans": [{"start_char": 39, "end_char": 48}],
            },
        ],
    }

    result = generate_chunk_variants(doc)

    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["variants"]["0.00"]["text"] == "Trinkschwäche "
    assert result["chunks"][1]["variants"]["0.00"]["text"] == "Hypotonie."
