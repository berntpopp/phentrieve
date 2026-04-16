from __future__ import annotations

from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.prompts.loader import get_mapping_prompt
from phentrieve.llm.provider import LLMProvider


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
        return response_model.model_validate(response.get("parsed", response))


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
                        "term_name": "Difficulty walking",
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
