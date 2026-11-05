"""#400: list negation and family experiencer in the standard pipeline.

Chunking splits "No A or B" into one chunk per finding, and only the chunk next
to the cue saw it. A relative named in the sentence ("His brother has ...") was
ignored. All sentences are invented.
"""

import warnings

import pytest

from phentrieve.config import DEFAULT_ASSERTION_CONFIG
from phentrieve.text_processing.assertion_detection import (
    AssertionStatus,
    get_spacy_model,
    pseudo_cue_spans,
)
from phentrieve.text_processing.full_text_service import run_standard_backend
from phentrieve.text_processing.pipeline import (
    FAMILY_EXPERIENCER,
    TextProcessingPipeline,
    _propagate_list_negation,
    _SentenceContext,
)

pytestmark = pytest.mark.skipif(
    get_spacy_model("en") is None or get_spacy_model("de") is None,
    reason="needs the en and de spaCy models",
)

# Model-free chunking that still splits lists at commas and conjunctions.
_CFG = [
    {"type": "paragraph"},
    {"type": "sentence"},
    {"type": "fine_grained_punctuation"},
    {"type": "conjunction"},
    {"type": "final_chunk_cleaner"},
]


def _pipeline(language: str) -> TextProcessingPipeline:
    warnings.filterwarnings("ignore")
    return TextProcessingPipeline(
        language=language,
        chunking_pipeline_config=_CFG,
        assertion_config=dict(DEFAULT_ASSERTION_CONFIG),
    )


@pytest.fixture(scope="module")
def en() -> TextProcessingPipeline:
    return _pipeline("en")


@pytest.fixture(scope="module")
def de() -> TextProcessingPipeline:
    return _pipeline("de")


def _statuses(pipeline: TextProcessingPipeline, text: str) -> dict[str, str]:
    return {
        chunk["text"].lower(): chunk["status"].value
        for chunk in pipeline.process(text, include_positions=True)
    }


def _status_of(statuses: dict[str, str], needle: str) -> str:
    hits = [status for chunk, status in statuses.items() if needle in chunk]
    assert len(hits) == 1, statuses
    return hits[0]


@pytest.mark.parametrize(
    ("text", "negated"),
    [
        ("No evidence of hearing loss or cataract.", ["cataract"]),
        ("No fever, rash or vomiting.", ["rash", "vomiting"]),
        ("She does not have ptosis or strabismus.", ["strabismus"]),
    ],
)
def test_negation_reaches_every_list_item(en, text, negated) -> None:
    statuses = _statuses(en, text)
    for item in negated:
        assert _status_of(statuses, item) == "negated", statuses


def test_german_list_is_negated(de) -> None:
    statuses = _statuses(de, "Keine Krampfanfaelle, Ataxie oder Hoerminderung.")

    assert set(statuses.values()) == {"negated"}, statuses


@pytest.mark.parametrize(
    ("text", "affirmed"),
    [
        ("No fever, but cough.", "cough"),
        ("No fever. Cough and rash.", "rash"),
        ("No fever and the patient is coughing.", "coughing"),
        ("Seizures were not observed, ataxia present.", "ataxia"),
        ("Not only seizures but also ataxia.", "ataxia"),
        ("Not only seizures but also ataxia.", "seizures"),
    ],
)
def test_negation_stops_at_scope_terminators(en, text, affirmed) -> None:
    statuses = _statuses(en, text)

    assert _status_of(statuses, affirmed) == "affirmed", statuses


def test_negation_crosses_chunks_split_inside_the_scope() -> None:
    """The issue's split: "evidence" | "hearing loss" | "cataract"."""
    text = "No evidence of hearing loss or cataract."
    chunks = [
        {"text": t, "status": s, "assertion_details": {}}
        for t, s in (
            ("evidence", AssertionStatus.NEGATED),
            ("hearing loss", AssertionStatus.AFFIRMED),
            ("cataract", AssertionStatus.AFFIRMED),
        )
    ]
    spans = [(3, 11), (15, 27), (31, 39)]
    context = _SentenceContext(text, "en", pseudo_cue_spans(text, "en"))

    _propagate_list_negation(context, chunks, spans)

    assert [c["status"] for c in chunks] == [AssertionStatus.NEGATED] * 3
    assert chunks[2]["assertion_details"]["propagated_negation"] is True


def _experiencer(pipeline: TextProcessingPipeline, text: str, needle: str):
    for chunk in pipeline.process(text, include_positions=True):
        if needle in chunk["text"].lower():
            return chunk["assertion_details"].get("experiencer")
    raise AssertionError(f"no chunk with {needle!r}")


@pytest.mark.parametrize(
    ("text", "needle", "expected"),
    [
        ("His brother has scoliosis.", "scoliosis", FAMILY_EXPERIENCER),
        (
            "Her mother had cataract and hearing loss.",
            "hearing loss",
            FAMILY_EXPERIENCER,
        ),
        ("Family history of deafness.", "deafness", FAMILY_EXPERIENCER),
        ("Mother reports that he has seizures.", "seizures", None),
        ("The patient has scoliosis.", "scoliosis", None),
        ("Her sister is healthy. She has scoliosis.", "scoliosis", None),
    ],
)
def test_family_experiencer(en, text, needle, expected) -> None:
    assert _experiencer(en, text, needle) == expected


def test_family_experiencer_german(de) -> None:
    assert _experiencer(de, "Der Vater hat eine Skoliose.", "skoliose") == (
        FAMILY_EXPERIENCER
    )


def test_standard_backend_lists_family_findings_apart(en, mocker) -> None:
    """A relative's finding is a family-history finding, not the patient's."""
    hits = {
        "scoliosis": ("HP:0002650", "Scoliosis"),
        "ataxia": ("HP:0001251", "Ataxia"),
    }

    def fake_query(*, text_chunks, **_kwargs):
        results = []
        for text in text_chunks:
            found = [hit for key, hit in hits.items() if key in text.lower()]
            results.append(
                {
                    "ids": [[hpo_id for hpo_id, _ in found]],
                    "metadatas": [[{"id": i, "label": n} for i, n in found]],
                    "similarities": [[0.9] * len(found)],
                }
            )
        return results

    mocker.patch(
        "phentrieve.text_processing.hpo_extraction_orchestrator.query_chunk_candidates",
        side_effect=fake_query,
    )

    result = run_standard_backend(
        text="His brother has scoliosis. The patient has ataxia.",
        text_pipeline=en,
        retriever=mocker.Mock(),
    )

    assert [t["id"] for t in result["aggregated_hpo_terms"]] == ["HP:0001251"]
    family = result["family_history_findings"]
    assert [t["id"] for t in family] == ["HP:0002650"]
    assert family[0]["experiencer"] == FAMILY_EXPERIENCER
