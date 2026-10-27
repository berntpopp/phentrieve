"""Batched encode / spaCy paths must match the per-item paths exactly (#410)."""

import hashlib
from typing import Any

import numpy as np
import pytest
import spacy

from phentrieve.text_processing import assertion_detection
from phentrieve.text_processing.assertion_detection import (
    AssertionDetector,
    AssertionStatus,
    CombinedAssertionDetector,
    DependencyAssertionDetector,
)
from phentrieve.text_processing.chunkers import SlidingWindowSemanticSplitter
from phentrieve.text_processing.pipeline import TextProcessingPipeline

pytestmark = pytest.mark.unit

TEXTS = [
    "Patient has severe headaches. No fever was observed.",
    "Mother reports seizures since age two and denies any loss of consciousness.",
    "The heart is normal and kidneys are unremarkable.",
    "Short.",
    "Hypotonia, developmental delay, and recurrent infections were noted. "
    "There is no evidence of hearing loss. Echocardiogram showed a ventricular "
    "septal defect; MRI brain was normal. Father has hypertension.",
    "Possible macrocephaly without hydrocephalus or seizures in the family.",
]

needs_en_spacy = pytest.mark.skipif(
    not spacy.util.is_package("en_core_web_sm"),
    reason="spaCy model en_core_web_sm not installed",
)


class HashingModel:
    """Deterministic stand-in for a SentenceTransformer (bag of hashed words)."""

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        self.encode_calls.append(list(texts))
        rows = []
        for text in texts:
            vector = np.zeros(64)
            for word in text.lower().split():
                digest = hashlib.sha512(word.encode()).digest()
                vector += np.frombuffer(digest, dtype=np.uint8) - 127.5
            rows.append(vector)
        return np.array(rows, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return 64


class TestSlidingWindowBatchedEncode:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "window_size_tokens": 7,
                "step_size_tokens": 1,
                "splitting_threshold": 0.5,
            },
            {
                "window_size_tokens": 3,
                "step_size_tokens": 1,
                "splitting_threshold": 0.9,
            },
            {
                "window_size_tokens": 4,
                "step_size_tokens": 2,
                "splitting_threshold": 0.95,
                "min_split_segment_length_words": 2,
            },
        ],
    )
    def test_chunk_matches_per_segment_splitting(self, params):
        model = HashingModel()
        splitter = SlidingWindowSemanticSplitter(language="en", model=model, **params)
        segments = [*TEXTS, "", "   "]

        expected = [
            split
            for segment in segments
            if segment.strip()
            for split in splitter._split_one_segment_by_sliding_window(segment)
        ]
        model.encode_calls.clear()

        assert splitter.chunk(segments) == expected

    def test_chunk_encodes_all_windows_in_one_call(self):
        model = HashingModel()
        splitter = SlidingWindowSemanticSplitter(
            language="en",
            model=model,
            window_size_tokens=4,
            step_size_tokens=2,
            splitting_threshold=0.95,
            min_split_segment_length_words=2,
        )

        result = splitter.chunk(TEXTS)

        assert len(model.encode_calls) == 1
        # The fixture must actually exercise splitting, or equality is trivial.
        assert len(result) > len(TEXTS)

    def test_chunk_skips_encode_when_no_segment_needs_it(self):
        model = HashingModel()
        splitter = SlidingWindowSemanticSplitter(language="en", model=model)

        assert splitter.chunk(["Short.", "", "Too short too"]) == [
            "Short.",
            "Too short too",
        ]
        assert model.encode_calls == []


@needs_en_spacy
class TestAssertionDetectBatch:
    @pytest.mark.parametrize("preference", ["dependency", "keyword", "any_negative"])
    def test_combined_detect_batch_matches_detect(self, preference):
        detector = CombinedAssertionDetector(language="en", preference=preference)
        inputs = [*TEXTS, "", "   "]

        assert detector.detect_batch(inputs) == [detector.detect(t) for t in inputs]

    def test_dependency_detect_batch_uses_one_pipe_call(self, monkeypatch):
        nlp = assertion_detection.get_spacy_model("en")
        calls = {"call": 0, "pipe": 0}

        class CountingNlp:
            def __call__(self, text):
                calls["call"] += 1
                return nlp(text)

            def pipe(self, texts, **kwargs):
                calls["pipe"] += 1
                return nlp.pipe(texts, **kwargs)

        detector = DependencyAssertionDetector(language="en")
        expected = [detector.detect(t) for t in TEXTS]
        monkeypatch.setattr(
            assertion_detection, "get_spacy_model", lambda lang: CountingNlp()
        )

        assert detector.detect_batch([*TEXTS, ""]) == [
            *expected,
            detector.detect(""),
        ]
        assert calls == {"call": 0, "pipe": 1}

    def test_detect_doc_matches_detect(self):
        detector = DependencyAssertionDetector(language="en")
        nlp = assertion_detection.get_spacy_model("en")
        for text in TEXTS:
            assert detector.detect_doc(text, nlp(text)) == detector.detect(text)


class RecordingDetector(AssertionDetector):
    """Status depends on the input text, so a mis-ordered batch would show."""

    def __init__(self) -> None:
        super().__init__(language="en")
        self.batches: list[list[str]] = []

    def detect(self, text_chunk: str) -> tuple[AssertionStatus, dict[str, Any]]:
        status = (
            AssertionStatus.NEGATED
            if "no " in text_chunk.lower()
            else AssertionStatus.AFFIRMED
        )
        return status, {"input": text_chunk}

    def detect_batch(self, text_chunks):
        self.batches.append(list(text_chunks))
        return super().detect_batch(text_chunks)


def test_pipeline_runs_assertion_detection_once_per_document():
    pipeline = TextProcessingPipeline(
        language="en",
        chunking_pipeline_config=[{"type": "paragraph"}, {"type": "sentence"}],
        assertion_config={"disable": True},
    )
    detector = RecordingDetector()
    pipeline.assertion_detector = detector
    text = "Patient has severe headaches. No fever was observed.\n\nMother is well."

    chunks = pipeline.process(text, include_positions=True)

    assert len(detector.batches) == 1
    assert len(detector.batches[0]) == len(chunks) == 3
    for chunk, assertion_input in zip(chunks, detector.batches[0], strict=True):
        assert (chunk["status"], chunk["assertion_details"]) == detector.detect(
            assertion_input
        )
    assert [c["status"] for c in chunks] == [
        AssertionStatus.AFFIRMED,
        AssertionStatus.NEGATED,
        AssertionStatus.AFFIRMED,
    ]
