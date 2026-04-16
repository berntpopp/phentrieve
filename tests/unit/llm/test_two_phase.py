from __future__ import annotations

import json

from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.prompts.loader import get_mapping_prompt
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import LLMResponse


class FakeProvider(LLMProvider):
    def __init__(self, responses: list[str]):
        super().__init__()
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return LLMResponse(
            content=self.responses.pop(0),
            model="gemini-2.5-flash",
            provider="gemini",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def run_structured_prompt(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        max_output_tokens=None,
    ):
        raise NotImplementedError


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
    provider = FakeProvider([json.dumps({"hpo_id": "HP:0002355"})])
    pipeline = TwoPhaseLLMPipeline(provider=provider)
    mapping_prompt = get_mapping_prompt("en")

    resolved_terms, prompt_tokens, completion_tokens = (
        pipeline._resolve_with_mapping_prompt(
            unresolved=[
                {
                    "phrase": "Frequent-Falls",
                    "category": "Abnormal",
                    "original_sentence": "The child has frequent falls while walking.",
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
    )

    assert resolved_terms
    assert prompt_tokens == 10
    assert completion_tokens == 5
    assert '"phrase": "frequent falls"' in provider.calls[0][-1]["content"]
