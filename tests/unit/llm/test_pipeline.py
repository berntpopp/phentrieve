from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.prompts.loader import load_prompt_template
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import LLMPhenotype, LLMPipelineConfig


class FakeProvider(LLMProvider):
    def __init__(self, payload):
        self.payload = payload

    def run_structured_prompt(
        self, *, system_prompt: str, user_prompt: str, response_model
    ):
        return response_model.model_validate(self.payload)


def test_load_prompt_template_reads_packaged_template():
    content = load_prompt_template("two_phase_system.txt")
    assert "HPO" in content


def test_two_phase_pipeline_normalizes_structured_terms():
    provider = FakeProvider(
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
    pipeline = TwoPhaseLLMPipeline(provider=provider)

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        config=LLMPipelineConfig(model="gpt-5.4-mini", mode="two_phase"),
    )

    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0001250",
            label="Seizure",
            evidence="Patient had recurrent seizures",
            assertion="present",
        )
    ]
    assert result.meta.llm_mode == "two_phase"
