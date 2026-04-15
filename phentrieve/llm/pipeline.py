from phentrieve.llm.prompts.loader import load_prompt_template
from phentrieve.llm.types import (
    LLMExtractionResult,
    LLMMeta,
    LLMPipelineConfig,
    LLMTermsResult,
)


class TwoPhaseLLMPipeline:
    def __init__(self, provider):
        self.provider = provider

    def run(self, *, text: str, config: LLMPipelineConfig):
        system_prompt = load_prompt_template("two_phase_system.txt")
        user_prompt = load_prompt_template("two_phase_user.txt").format(text=text)
        parsed = self.provider.run_structured_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=LLMTermsResult,
        )
        return LLMExtractionResult(
            terms=parsed.terms,
            meta=LLMMeta(
                llm_model=config.model,
                llm_mode=config.mode,
            ),
        )
