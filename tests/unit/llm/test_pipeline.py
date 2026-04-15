import pytest
from pydantic import ValidationError

from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.prompts.loader import load_prompt_template
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    LLMExtractionResult,
    LLMMeta,
    LLMPhenotype,
    LLMPipelineConfig,
)


class FakeProvider(LLMProvider):
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def run_structured_prompt(
        self, *, system_prompt: str, user_prompt: str, response_model
    ):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
            }
        )
        return response_model.model_validate(self.payload)


def test_load_prompt_template_reads_packaged_template():
    content = load_prompt_template("two_phase_system.txt")
    assert "HPO" in content


def test_llm_extraction_result_requires_meta():
    with pytest.raises(ValidationError):
        LLMExtractionResult.model_validate(
            {
                "terms": [
                    {
                        "term_id": "HP:0001250",
                        "label": "Seizure",
                        "evidence": "Patient had recurrent seizures",
                        "assertion": "present",
                    }
                ]
            }
        )


def test_two_phase_pipeline_normalizes_structured_terms():
    system_prompt = load_prompt_template("two_phase_system.txt")
    user_prompt = load_prompt_template("two_phase_user.txt").format(
        text="Patient had recurrent seizures."
    )
    provider = FakeProvider(
        {
            "terms": [
                {
                    "term_id": "HP:0001250",
                    "label": "Seizure",
                    "evidence": "Patient had recurrent seizures",
                    "assertion": "present",
                }
            ],
            "meta": {
                "llm_model": "gpt-5.4-mini",
                "llm_mode": "two_phase",
                "prompt_version": "v1",
                "token_input": None,
                "token_output": None,
            },
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider)

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        config=LLMPipelineConfig(model="gpt-5.4-mini", mode="two_phase"),
    )

    assert provider.calls == [
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_model": LLMExtractionResult,
        }
    ]
    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0001250",
            label="Seizure",
            evidence="Patient had recurrent seizures",
            assertion="present",
        )
    ]
    assert result.meta == LLMMeta(
        llm_model="gpt-5.4-mini",
        llm_mode="two_phase",
        prompt_version="v1",
    )
