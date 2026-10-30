"""C1 regression: deterministic assertion detection must see leading negation cues.

The final-chunk cleaner strips leading cue words ("no", "does not have") from the
chunk text, and the chunk's character offset jumps past the cue. Before the fix,
assertion detection ran on the cue-stripped chunk and reported AFFIRMED for
explicitly negated findings -- the worst failure class for an annotation tool.

The fix runs assertion detection over the chunk plus its preceding within-sentence
context (so leading cues are restored), bounded by the previous chunk and the
current sentence so a following concept's cue is not pulled in.
"""

import warnings

import pytest

from phentrieve.config import DEFAULT_ASSERTION_CONFIG
from phentrieve.text_processing.assertion_detection import pseudo_cue_spans
from phentrieve.text_processing.pipeline import (
    TextProcessingPipeline,
    _splits_pseudo_cue,
)

# Model-free chunking config that still runs the final-chunk cleaner (which strips
# leading cue words) and conjunction/punctuation splitting -- reproduces C1 without
# needing a SentenceTransformer model.
_CFG = [
    {"type": "paragraph"},
    {"type": "sentence"},
    {"type": "fine_grained_punctuation"},
    {"type": "conjunction"},
    {"type": "final_chunk_cleaner"},
]


@pytest.fixture(scope="module")
def pipeline() -> TextProcessingPipeline:
    warnings.filterwarnings("ignore")
    return TextProcessingPipeline(
        language="en",
        chunking_pipeline_config=_CFG,
        assertion_config=dict(DEFAULT_ASSERTION_CONFIG),
    )


def _status(chunk: dict) -> str:
    s = chunk["status"]
    return s.value if hasattr(s, "value") else s


@pytest.mark.parametrize(
    ("chunk_span", "expected"),
    [
        ((0, 15), True),  # "Ataxia cannot be"
        ((16, 25), True),  # "ruled out"
        ((0, 25), False),  # whole phrase: the detector judged it correctly
        ((27, 35), False),  # another sentence
    ],
)
def test_chunk_holding_part_of_a_pseudo_cue_is_detected(chunk_span, expected) -> None:
    """Semantic chunking can split "cannot be ruled out" (#459)."""
    text = "Ataxia cannot be ruled out. No tremor."
    spans = pseudo_cue_spans(text, "en")

    assert _splits_pseudo_cue(chunk_span, spans) is expected


def _status_for(pipeline: TextProcessingPipeline, text: str, needle: str) -> str:
    chunks = pipeline.process(text, include_positions=True)
    hits = [c for c in chunks if needle.lower() in c["text"].lower()]
    assert hits, f"no chunk containing {needle!r} in {[c['text'] for c in chunks]}"
    return _status(hits[0])


@pytest.mark.parametrize(
    "text,needle",
    [
        ("There is no nystagmus.", "nystagmus"),
        ("She does not have ataxia.", "ataxia"),
        ("There is no nystagmus. She does not have ataxia.", "nystagmus"),
        ("There is no nystagmus. She does not have ataxia.", "ataxia"),
        ("The patient denies headache.", "headache"),
        ("Patient has seizures but no fever.", "fever"),
    ],
)
def test_leading_negation_is_detected(pipeline, text, needle):
    """no X / not X / does not have X / denies X / "but no X" -> negated."""
    assert _status_for(pipeline, text, needle) == "negated"


@pytest.mark.parametrize(
    "text,needle",
    [
        ("The patient had seizures.", "seizure"),
        ("Patient has seizures but no fever.", "seizure"),
        ("Bilateral hearing loss was noted.", "hearing"),
    ],
)
def test_affirmed_findings_are_not_over_negated(pipeline, text, needle):
    """A present finding with a negation cue elsewhere in the sentence stays affirmed."""
    assert _status_for(pipeline, text, needle) == "affirmed"
