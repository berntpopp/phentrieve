from __future__ import annotations

from phentrieve.llm.pipeline import TwoPhaseLLMPipeline, prepare_retrieval_queries
from phentrieve.llm.pipeline_phase1 import expand_combined_phase1_extractions
from phentrieve.llm.prompts.loader import get_mapping_prompt, get_prompt
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import AnnotationMode


class FakeProvider(LLMProvider):
    def __init__(self, responses: list[dict[str, object]]):
        super().__init__()
        self.responses = list(responses)
        self.structured_calls: list[dict[str, str]] = []

    def complete(self, messages):
        raise AssertionError("unused")

    def run_structured_prompt(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        max_output_tokens=None,
    ):
        self.structured_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        response = self.responses.pop(0)
        self.last_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        parsed = response_model.model_validate(response.get("parsed", response))
        self.last_structured_payload = parsed.model_dump(mode="json")
        return parsed


def test_try_local_match_requires_multiword_substring_boundary():
    pipeline = TwoPhaseLLMPipeline(provider=FakeProvider([]))

    matched = pipeline._try_local_match(
        "scoliosis",
        [
            {
                "hpo_id": "HP:0100884",
                "term_name": "Compensatory scoliosis",
                "score": 0.9,
            }
        ],
    )

    assert matched is None


def test_try_local_match_prefers_exact_token_set_over_broader_subset():
    pipeline = TwoPhaseLLMPipeline(provider=FakeProvider([]))

    matched = pipeline._try_local_match(
        "generalized hypotonia",
        [
            {
                "hpo_id": "HP:0001252",
                "term_name": "Hypotonia",
                "score": 0.82,
            },
            {
                "hpo_id": "HP:0001290",
                "term_name": "Generalized hypotonia",
                "score": 0.9,
            },
        ],
    )

    assert matched is not None
    assert matched["hpo_id"] == "HP:0001290"


def test_try_local_match_prefers_matched_surface_over_canonical_label():
    pipeline = TwoPhaseLLMPipeline(provider=FakeProvider([]))

    matched = pipeline._try_local_match(
        "difficulty in walking",
        [
            {
                "hpo_id": "HP:0001288",
                "term_name": "Gait disturbance",
                "matched_text": "Difficulty walking",
                "matched_component": "synonym",
                "score": 0.91,
            },
            {
                "hpo_id": "HP:0002317",
                "term_name": "Unsteady gait",
                "score": 0.89,
            },
        ],
    )

    assert matched is not None
    assert matched["hpo_id"] == "HP:0001288"


def test_resolve_with_mapping_prompt_normalizes_phrase_before_llm_call():
    provider = FakeProvider(
        [{"parsed": {"phrase": "frequent falls", "hpo_id": "HP:0002355"}}]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider)
    mapping_prompt = get_mapping_prompt("en")

    (
        resolved_terms,
        prompt_tokens,
        completion_tokens,
        _token_usage,
        _request_count,
        _local_fallback_count,
        _mapping_trace,
    ) = pipeline._resolve_with_mapping_prompt(
        unresolved=[
            {
                "phrase": "Frequent-Falls",
                "category": "Abnormal",
                "grounded_context": {
                    "primary_chunk_text": "The child has frequent falls while walking."
                },
                "candidates": [
                    {
                        "hpo_id": "HP:0002355",
                        "term_name": "Gait disturbance",
                        "matched_text": "Difficulty walking",
                        "matched_component": "synonym",
                    }
                ],
            }
        ],
        mapping_prompt=mapping_prompt,
    )

    assert resolved_terms
    assert prompt_tokens == 10
    assert completion_tokens == 5
    assert '"phrase": "frequent falls"' in provider.structured_calls[0]["user_prompt"]
    assert (
        '"matched_text": "Difficulty walking"'
        in provider.structured_calls[0]["user_prompt"]
    )


def test_prepare_retrieval_queries_strips_unit_suffixes_without_losing_core_phrase():
    queries = prepare_retrieval_queries("serum creatinine 11.2 mg/dL")

    assert "serum creatinine 11.2 mg/dL" in queries
    assert "serum creatinine 11.2" in queries
    assert queries[-1] == "serum creatinine 11.2"

    marker_queries = prepare_retrieval_queries("inflammatory marker 152 mg/l")
    assert "inflammatory marker 152 mg/l" in marker_queries
    assert "inflammatory marker 152" in marker_queries


def test_prepare_retrieval_queries_keeps_location_and_morphology_modifiers():
    queries = prepare_retrieval_queries("swelling in the right lower leg")

    assert queries[0] == "swelling in the right lower leg"
    assert any("right lower leg" in query for query in queries)
    assert "swelling" not in queries[1:]


def test_prepare_retrieval_queries_does_not_invent_hand_written_paraphrases():
    queries = prepare_retrieval_queries("tongue biting")

    assert "self biting" not in queries
    assert "self-biting" not in queries


def test_prepare_retrieval_queries_expands_known_abbreviations_after_original():
    queries = prepare_retrieval_queries("XLID")

    assert queries[0] == "XLID"
    assert "X-linked intellectual disability" in queries


def test_prepare_retrieval_queries_adds_conservative_lab_canonical_variant():
    queries = prepare_retrieval_queries("lactate dehydrogenase was markedly elevated")

    assert queries[0] == "lactate dehydrogenase was markedly elevated"
    assert "elevated lactate dehydrogenase" in queries


def test_prepare_retrieval_queries_adds_conservative_low_output_variant():
    queries = prepare_retrieval_queries("urine output remained low")

    assert queries[0] == "urine output remained low"
    assert "low urine output" in queries


def test_phase1_prompt_says_phase2_handles_retrieval_variants():
    prompt = get_prompt(AnnotationMode.TWO_PHASE, "en")
    system = prompt.render_system_prompt()

    assert "Phase 2 will compute retrieval variants" in system
    assert "faithful extraction" in system
    assert "short normalized clinical phrase" not in system


def test_expand_combined_phase1_extractions_splits_only_source_substring_mentions():
    expanded = expand_combined_phase1_extractions(
        [
            {
                "phrase": "hypertonia/spasticity of the extremities",
                "category": "abnormal",
                "chunk_ids": [15],
                "evidence_text": "hypertonia/spasticity of the extremities",
            },
            {
                "phrase": "pontine cerebellar hypoplasia",
                "category": "abnormal",
                "chunk_ids": [44, 45],
                "evidence_text": "pontine cerebellar hypoplasia",
            },
        ]
    )

    phrases = [item["phrase"] for item in expanded]

    assert phrases == [
        "hypertonia",
        "spasticity of the extremities",
        "pontine cerebellar hypoplasia",
    ]
    assert all(item["category"] == "abnormal" for item in expanded)
    assert expanded[0]["chunk_ids"] == [15]
    assert expanded[2]["chunk_ids"] == [44, 45]


def test_expand_combined_phase1_extractions_keeps_abbreviations_source_faithful():
    expanded = expand_combined_phase1_extractions(
        [
            {
                "phrase": "XLID",
                "category": "abnormal",
                "chunk_ids": [29],
                "evidence_text": "XLID",
            }
        ]
    )

    assert expanded == [
        {
            "phrase": "XLID",
            "category": "abnormal",
            "chunk_ids": [29],
            "evidence_text": "XLID",
        }
    ]


def test_expand_combined_phase1_extractions_leaves_non_combined_phrases_unchanged():
    original = [
        {
            "phrase": "severe global developmental delay",
            "category": "abnormal",
            "chunk_ids": [3],
            "evidence_text": "severe global developmental delay",
        },
        {
            "phrase": "respiratory failure",
            "category": "abnormal",
            "chunk_ids": [4],
            "evidence_text": "respiratory failure",
        },
    ]

    assert expand_combined_phase1_extractions(original) == original
