"""Findings of another person who is not a relative (experiencer ``other``).

They are resolved through their own retrieval batch, like family findings, and
returned in ``other_person_findings``; they never enter ``terms``. Non-phenotype
metadata keeps experiencer ``proband`` with category ``Other`` and stays dropped.
"""

from __future__ import annotations

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.types import LLMPipelineConfig
from tests.unit.llm.test_pipeline import (
    FakeProvider,
    PhraseKeyedToolExecutor,
    grounded_phenotype,
)

TEXT = "The patient has asthma. His neighbour has a cataract. Onset in infancy."


def _run():
    neighbour = grounded_phenotype("cataract", "Other")
    neighbour["experiencer"] = "other"
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype("asthma", "Abnormal"),
                        neighbour,
                        grounded_phenotype("onset in infancy", "Other"),
                    ]
                }
            }
        ]
    )
    tool_executor = PhraseKeyedToolExecutor(
        {
            "asthma": [{"hpo_id": "HP:0002099", "term_name": "asthma", "score": 0.95}],
            "cataract": [
                {"hpo_id": "HP:0000518", "term_name": "cataract", "score": 0.95}
            ],
            "onset in infancy": [
                {"hpo_id": "HP:0003593", "term_name": "onset in infancy", "score": 0.95}
            ],
        }
    )
    pipeline = pipeline_module.TwoPhaseLLMPipeline(
        provider=provider, tool_executor=tool_executor
    )
    result = pipeline.run(
        text=TEXT,
        grounded_chunks=[{"chunk_id": 1, "text": TEXT}],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )
    return result, tool_executor


def test_other_person_finding_is_returned_separately():
    result, _ = _run()

    assert [t.term_id for t in result.other_person_findings] == ["HP:0000518"]
    assert result.other_person_findings[0].experiencer == "other"
    assert [t.term_id for t in result.terms] == ["HP:0002099"]
    assert result.family_history_findings == []


def test_other_person_finding_is_retrieved_in_its_own_batch():
    _, tool_executor = _run()

    assert [q["phrases"] for q in tool_executor.queries] == [["asthma"], ["cataract"]]
