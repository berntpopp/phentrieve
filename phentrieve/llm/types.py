from pydantic import BaseModel, Field


class LLMPhenotype(BaseModel):
    term_id: str
    label: str
    evidence: str | None = None
    assertion: str = "present"


class LLMMeta(BaseModel):
    llm_model: str
    llm_mode: str
    prompt_version: str = "v1"
    token_input: int | None = None
    token_output: int | None = None


class LLMPipelineConfig(BaseModel):
    model: str
    mode: str = "two_phase"
    language: str | None = None


class LLMExtractionResult(BaseModel):
    terms: list[LLMPhenotype] = Field(default_factory=list)
    meta: LLMMeta


class LLMTermsResult(BaseModel):
    terms: list[LLMPhenotype] = Field(default_factory=list)
