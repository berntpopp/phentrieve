from __future__ import annotations

from typing import Any

from phentrieve.cli.text_interactive import interactive_text_mode


def test_interactive_text_mode_passes_assertion_preference(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    prompt_inputs = iter(["Patient has seizures.", "q"])

    class FakePipeline:
        def __init__(self, *, assertion_config: dict[str, Any], **kwargs: Any) -> None:
            captured["assertion_config"] = assertion_config

        def process(self, raw_text: str) -> list[dict[str, str]]:
            return [{"text": raw_text, "status": "present"}]

    class FakeRetriever:
        pass

    monkeypatch.setattr(
        "phentrieve.cli.text_interactive.resolve_chunking_pipeline_config",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_interactive.TextProcessingPipeline",
        FakePipeline,
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_interactive.DenseRetriever.from_model_name",
        lambda **kwargs: FakeRetriever(),
    )
    monkeypatch.setattr(
        "phentrieve.embeddings.load_embedding_model",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_interactive.orchestrate_hpo_extraction",
        lambda **kwargs: ([], []),
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_interactive._display_interactive_text_results",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "rich.prompt.Prompt.ask",
        lambda *args, **kwargs: next(prompt_inputs),
    )
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: False,
    )

    interactive_text_mode(assertion_preference="keyword")

    assert captured["assertion_config"]["preference"] == "keyword"
