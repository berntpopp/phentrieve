"""
Unit tests for mention dataclasses.

Tests the core data structures for mention-level HPO extraction:
- HPOCandidate
- Mention
- MentionGroup
- DocumentMentions
"""

from phentrieve.text_processing.assertion_representation import (
    AssertionVector,
    affirmed_vector,
    negated_vector,
)
from phentrieve.text_processing.mention import (
    CanonicalAssertion,
    DocumentMentions,
    HPOCandidate,
    Mention,
    MentionGroup,
    map_assertion_to_dataset,
)


class TestHPOCandidate:
    """Tests for HPOCandidate dataclass."""

    def test_basic_creation(self):
        """Test basic HPOCandidate creation."""
        candidate = HPOCandidate(
            hpo_id="HP:0001250",
            label="Seizure",
            score=0.85,
        )
        assert candidate.hpo_id == "HP:0001250"
        assert candidate.label == "Seizure"
        assert candidate.score == 0.85
        assert candidate.refined_score is None
        assert candidate.depth == 0

    def test_effective_score_uses_score_when_no_refined(self):
        """Test effective_score returns score when refined_score is None."""
        candidate = HPOCandidate(
            hpo_id="HP:0001250",
            label="Seizure",
            score=0.85,
        )
        assert candidate.effective_score == 0.85

    def test_effective_score_uses_refined_when_available(self):
        """Test effective_score returns refined_score when set."""
        candidate = HPOCandidate(
            hpo_id="HP:0001250",
            label="Seizure",
            score=0.85,
            refined_score=0.92,
        )
        assert candidate.effective_score == 0.92

    def test_generic_flag_based_on_depth(self):
        """Test that is_generic is set based on depth."""
        shallow = HPOCandidate(
            hpo_id="HP:0000001",
            label="All",
            score=0.9,
            depth=1,
        )
        assert shallow.is_generic is True

        deep = HPOCandidate(
            hpo_id="HP:0001250",
            label="Seizure",
            score=0.9,
            depth=5,
        )
        assert deep.is_generic is False

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        candidate = HPOCandidate(
            hpo_id="HP:0001250",
            label="Seizure",
            score=0.85,
            refined_score=0.92,
            depth=4,
            synonyms=["Epileptic seizure"],
        )

        data = candidate.to_dict()
        restored = HPOCandidate.from_dict(data)

        assert restored.hpo_id == candidate.hpo_id
        assert restored.label == candidate.label
        assert restored.score == candidate.score
        assert restored.refined_score == candidate.refined_score


class TestMention:
    """Tests for Mention dataclass."""

    def test_basic_creation(self):
        """Test basic Mention creation."""
        mention = Mention(
            text="seizures",
            start_char=45,
            end_char=53,
            sentence_idx=2,
        )
        assert mention.text == "seizures"
        assert mention.start_char == 45
        assert mention.end_char == 53
        assert mention.sentence_idx == 2
        assert mention.mention_id  # UUID should be auto-generated

    def test_span_length(self):
        """Test span_length property."""
        mention = Mention(text="seizures", start_char=45, end_char=53)
        assert mention.span_length == 8

    def test_top_candidate(self):
        """Test top_candidate property."""
        mention = Mention(text="seizures", start_char=0, end_char=8)
        mention.hpo_candidates = [
            HPOCandidate(hpo_id="HP:0001250", label="Seizure", score=0.9),
            HPOCandidate(hpo_id="HP:0001251", label="Epilepsy", score=0.7),
        ]

        assert mention.top_candidate is not None
        assert mention.top_candidate.hpo_id == "HP:0001250"

    def test_top_candidate_none_when_empty(self):
        """Test top_candidate returns None when no candidates."""
        mention = Mention(text="seizures", start_char=0, end_char=8)
        assert mention.top_candidate is None

    def test_overlaps_with(self):
        """Test overlap detection between mentions."""
        m1 = Mention(text="severe seizures", start_char=10, end_char=25)
        m2 = Mention(text="seizures", start_char=17, end_char=25)
        m3 = Mention(text="headaches", start_char=30, end_char=39)

        assert m1.overlaps_with(m2) is True
        assert m1.overlaps_with(m3) is False

    def test_canonical_assertion_affirmed_default(self):
        """Test default assertion is affirmed."""
        mention = Mention(text="seizures", start_char=0, end_char=8)
        assert mention.get_canonical_assertion() == "affirmed"

    def test_canonical_assertion_from_vector(self):
        """Test assertion from AssertionVector."""
        mention = Mention(text="seizures", start_char=0, end_char=8)
        mention.assertion = negated_vector(0.9)
        assert mention.get_canonical_assertion() == "negated"

    def test_is_in_family_history(self):
        """Test family history detection."""
        # Via section type
        m1 = Mention(text="epilepsy", start_char=0, end_char=8)
        m1.section_type = "family_history"
        assert m1.is_in_family_history() is True

        # Via assertion vector
        m2 = Mention(text="epilepsy", start_char=0, end_char=8)
        m2.assertion = AssertionVector(family_history=True)
        assert m2.is_in_family_history() is True

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        mention = Mention(
            text="seizures",
            start_char=45,
            end_char=53,
            sentence_idx=2,
            section_type="current_findings",
        )
        mention.assertion = affirmed_vector(0.95)

        data = mention.to_dict()
        restored = Mention.from_dict(data)

        assert restored.text == mention.text
        assert restored.start_char == mention.start_char
        assert restored.assertion is not None

    def test_hash_and_equality(self):
        """Test that mentions can be used in sets/dicts."""
        m1 = Mention(text="seizures", start_char=0, end_char=8)
        m2 = Mention(text="seizures", start_char=0, end_char=8)
        m3 = Mention(text="seizures", start_char=0, end_char=8)
        m3.mention_id = m1.mention_id  # Same ID

        # Different IDs mean different mentions
        assert m1 != m2
        # Same ID means same mention
        assert m1 == m3


class TestMentionGroup:
    """Tests for MentionGroup dataclass."""

    def test_basic_creation(self):
        """Test basic MentionGroup creation."""
        m1 = Mention(text="seizures", start_char=0, end_char=8)
        m1.hpo_candidates = [
            HPOCandidate(hpo_id="HP:0001250", label="Seizure", score=0.9)
        ]
        m2 = Mention(text="convulsions", start_char=20, end_char=31)
        m2.hpo_candidates = [
            HPOCandidate(hpo_id="HP:0001250", label="Seizure", score=0.85)
        ]

        group = MentionGroup(mentions=[m1, m2])

        assert group.num_mentions == 2
        assert group.representative_mention is not None

    def test_all_hpo_ids(self):
        """Test collecting all HPO IDs from group."""
        m1 = Mention(text="seizures", start_char=0, end_char=8)
        m1.hpo_candidates = [
            HPOCandidate(hpo_id="HP:0001250", label="Seizure", score=0.9)
        ]
        m2 = Mention(text="fever", start_char=20, end_char=25)
        m2.hpo_candidates = [
            HPOCandidate(hpo_id="HP:0001945", label="Fever", score=0.8)
        ]

        group = MentionGroup(mentions=[m1, m2])

        assert "HP:0001250" in group.all_hpo_ids
        assert "HP:0001945" in group.all_hpo_ids

    def test_span_range(self):
        """Test span range across all mentions."""
        m1 = Mention(text="seizures", start_char=10, end_char=18)
        m2 = Mention(text="convulsions", start_char=50, end_char=61)

        group = MentionGroup(mentions=[m1, m2])

        assert group.span_range == (10, 61)

    def test_dataset_assertion_mapping(self):
        """Test mapping to dataset-specific assertion."""
        m1 = Mention(text="seizures", start_char=0, end_char=8)
        m1.assertion = negated_vector(0.9)

        group = MentionGroup(
            mentions=[m1],
            final_assertion=m1.assertion,
        )

        assert group.get_canonical_assertion() == "negated"
        assert group.get_dataset_assertion("phenobert") == "ABSENT"


class TestDocumentMentions:
    """Tests for DocumentMentions container."""

    def test_basic_creation(self):
        """Test basic DocumentMentions creation."""
        doc = DocumentMentions(
            doc_id="case_001",
            full_text="Patient presents with seizures and headaches.",
        )
        assert doc.doc_id == "case_001"
        assert doc.num_mentions == 0
        assert doc.num_groups == 0

    def test_add_mentions_and_groups(self):
        """Test adding mentions and groups."""
        doc = DocumentMentions(
            doc_id="case_001",
            full_text="Patient presents with seizures.",
        )

        m1 = Mention(text="seizures", start_char=22, end_char=30)
        m1.hpo_candidates = [
            HPOCandidate(hpo_id="HP:0001250", label="Seizure", score=0.9)
        ]
        doc.add_mention(m1)

        assert doc.num_mentions == 1

        group = MentionGroup(
            mentions=[m1],
            final_hpo=m1.hpo_candidates[0],
        )
        doc.add_group(group)

        assert doc.num_groups == 1

    def test_to_benchmark_format(self):
        """Test conversion to benchmark format."""
        doc = DocumentMentions(
            doc_id="case_001",
            full_text="Patient has seizures and no headaches.",
        )

        m1 = Mention(text="seizures", start_char=12, end_char=20)
        m1.assertion = affirmed_vector()
        m1.hpo_candidates = [
            HPOCandidate(hpo_id="HP:0001250", label="Seizure", score=0.9)
        ]

        m2 = Mention(text="headaches", start_char=28, end_char=37)
        m2.assertion = negated_vector()
        m2.hpo_candidates = [
            HPOCandidate(hpo_id="HP:0002315", label="Headache", score=0.85)
        ]

        doc.add_mention(m1)
        doc.add_mention(m2)

        g1 = MentionGroup(
            mentions=[m1],
            final_hpo=m1.hpo_candidates[0],
            final_assertion=m1.assertion,
        )
        g2 = MentionGroup(
            mentions=[m2],
            final_hpo=m2.hpo_candidates[0],
            final_assertion=m2.assertion,
        )
        doc.add_group(g1)
        doc.add_group(g2)

        result = doc.to_benchmark_format(dataset="phenobert")

        assert ("HP:0001250", "PRESENT") in result
        assert ("HP:0002315", "ABSENT") in result


class TestAssertionMapping:
    """Tests for assertion label mapping."""

    def test_map_affirmed_to_present(self):
        """Test affirmed maps to PRESENT."""
        assert map_assertion_to_dataset("affirmed", "phenobert") == "PRESENT"

    def test_map_negated_to_absent(self):
        """Test negated maps to ABSENT."""
        assert map_assertion_to_dataset("negated", "phenobert") == "ABSENT"

    def test_map_uncertain_to_uncertain(self):
        """Test uncertain maps to UNCERTAIN."""
        assert map_assertion_to_dataset("uncertain", "phenobert") == "UNCERTAIN"

    def test_map_normal_to_present(self):
        """Test normal maps to PRESENT (normal is still a finding)."""
        assert map_assertion_to_dataset("normal", "phenobert") == "PRESENT"

    def test_case_insensitive_mapping(self):
        """Test mapping is case-insensitive."""
        assert map_assertion_to_dataset("AFFIRMED", "phenobert") == "PRESENT"
        assert map_assertion_to_dataset("Negated", "phenobert") == "ABSENT"

    def test_unknown_dataset_uses_phenobert(self):
        """Test unknown dataset falls back to phenobert mapping."""
        assert map_assertion_to_dataset("negated", "unknown_dataset") == "ABSENT"


class TestCanonicalAssertion:
    """Tests for CanonicalAssertion enum."""

    def test_enum_values(self):
        """Test all canonical assertion values exist."""
        assert CanonicalAssertion.AFFIRMED.value == "affirmed"
        assert CanonicalAssertion.NEGATED.value == "negated"
        assert CanonicalAssertion.UNCERTAIN.value == "uncertain"
        assert CanonicalAssertion.NORMAL.value == "normal"
        assert CanonicalAssertion.HISTORICAL.value == "historical"
        assert CanonicalAssertion.FAMILY.value == "family"
