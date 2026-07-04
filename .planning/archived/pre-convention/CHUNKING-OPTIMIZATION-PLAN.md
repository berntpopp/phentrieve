# Chunking Optimization and Benchmarking Framework

**Status:** Active
**Date:** 2025-01-18
**Related Issues:** [#17](https://github.com/berntpopp/phentrieve/issues/17), [#25](https://github.com/berntpopp/phentrieve/issues/25)
**Priority:** High
**Estimated Effort:** 5-7 weeks
**Design Principles:** KISS, DRY, SOLID, Iterative Delivery
**Software Status:** Alpha (no backward compatibility constraints)

---

## Executive Summary

This plan proposes **focused, pragmatic improvements** to Phentrieve's text chunking and benchmarking infrastructure, following rigorous software engineering principles.

### Key Goals

1. **Enhanced Chunking:** Unified chunker with adaptive sizing, parallelization, and medical abbreviation support
2. **Benchmarking Framework:** Modular components for systematic strategy evaluation
3. **Span-Based Metrics:** Approximate matching for evidence span evaluation

### Design Principles Applied

- ✅ **KISS (Keep It Simple):** 3 enhanced classes instead of 14 new ones
- ✅ **DRY (Don't Repeat Yourself):** Shared utilities for common operations
- ✅ **SOLID:** Single responsibility, dependency injection, extensibility
- ✅ **Iterative:** Deliver value incrementally, profile before optimizing
- ✅ **Alpha Software:** Clean slate, no legacy baggage

### Expected Outcomes

- **Performance:** 2-4x speedup for batch processing (after profiling validates need)
- **Quality:** 10-15% F1 improvement with adaptive chunking
- **Maintainability:** Reduced complexity, better test coverage
- **User Experience:** Simplified configuration, clear design

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Research Foundations](#2-research-foundations)
3. [Proposed Improvements](#3-proposed-improvements)
4. [Benchmarking Framework](#4-benchmarking-framework)
5. [Span-Based Evaluation](#5-span-based-evaluation)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Technical Specifications](#7-technical-specifications)
8. [References](#8-references)

---

## 1. Current State Analysis

### 1.1 Existing Architecture

**Location:** `phentrieve/text_processing/`

**Chunker Types (7):**
1. `NoOpChunker` - Pass-through
2. `ParagraphChunker` - Split on double newlines
3. `SentenceChunker` - pysbd-based segmentation
4. `FineGrainedPunctuationChunker` - Punctuation splitting
5. `ConjunctionChunker` - Split on conjunctions
6. `SlidingWindowSemanticSplitter` - Cosine similarity-based
7. `FinalChunkCleaner` - Cleanup low-value words

**Predefined Strategies (7):**
- `simple`, `semantic`, `detailed`, `sliding_window` (default), `sliding_window_cleaned`, `sliding_window_punct_cleaned`, `sliding_window_punct_conj_cleaned`

**Multilingual:** EN, DE, FR, ES, NL

**Test Coverage:** 157 tests, 13% statement coverage

### 1.2 Key Limitations

**Performance:**
- ❌ No parallelization for batch processing
- ❌ O(n²) similarity comparisons in sliding window
- ❌ Model loaded per pipeline instance

**Algorithmic:**
- ❌ Fixed window size (no adaptation to text density)
- ❌ No chunk overlap option for context preservation
- ❌ Medical abbreviations (pt., dx.) cause incorrect splits

**Benchmarking:**
- ❌ Only 1 full-text annotation example
- ❌ No span-based metrics (only term-level evaluation)
- ❌ No systematic chunking strategy comparison

### 1.3 Refactoring Approach

✅ **Clean refactoring** - Replace old strategies with improved versions
✅ **Simplify configuration** - Reduce from 7 strategies to 3 clear options
✅ **No legacy support** - Fresh start for alpha software

---

## 2. Research Foundations

### 2.1 Industry Benchmarks

**NVIDIA 2024 Chunking Study:**
- Page-level chunking: 0.648 accuracy, most consistent
- Query type matters: Factoid (256-512 tokens), Analytical (1024+ tokens)

**Anthropic Contextual Retrieval (2024):**
- Adding document context to chunks: 10-20% improvement
- Simple approach: Prepend first sentence as context

**LangChain SemanticChunker:**
- Embedding-based semantic boundaries
- Auto-calculated thresholds from distance distribution

**Key Insight:** Semantic chunking with adaptive sizing outperforms fixed-size approaches.

### 2.2 Evaluation Standards

**SemEval 2013 NER Evaluation:**
- **Strict:** Exact boundary + correct label
- **Exact:** Exact boundary, any label
- **Partial:** Overlap ≥ threshold
- **Type:** Any overlap + correct label

**Partial Match Weighting:**
```
Precision = (COR + 0.5 × PAR) / ACT
Recall = (COR + 0.5 × PAR) / POS
```

**Intersection over Union (IoU):** Standard for span overlap (threshold: 0.5)

### 2.3 Parallelization Patterns

**Python Multiprocessing:**
- ~3-5x speedup for batch document processing
- ~2-3x for within-document parallelization
- Critical: Load model once per worker, not per document

**Sentence-Transformers Built-in:**
```python
model.encode_multi_process(
    sentences,
    pool_size=4,
    batch_size=32
)
```

**Lesson:** Use library features before building custom solutions.

---

## 3. Proposed Improvements

### 3.1 Overview - Simplified Design

**Replace existing strategies with 3 enhanced components:**

1. **`EnhancedSlidingWindowSplitter`** - Unified semantic chunker with all optimizations
2. **`MedicalPunctuationChunker`** - Extends existing with abbreviation support
3. **`ChunkPostProcessor`** - Optional overlap addition

**Key Principle:** Configuration over proliferation of classes.

### 3.2 Component 1: Enhanced Sliding Window Splitter

**File:** `phentrieve/text_processing/chunkers.py` (add to existing)

```python
"""
Enhanced semantic chunker with adaptive sizing and parallelization.

Replaces the need for multiple separate classes by using configuration.
"""

from typing import List, Dict, Any, Optional
from multiprocessing import cpu_count
from sentence_transformers import SentenceTransformer
import numpy as np

class EnhancedSlidingWindowSplitter(TextChunker):
    """
    Unified sliding window semantic splitter with optional enhancements.

    Features:
    - Adaptive window sizing based on text density
    - Parallel embedding generation for long texts
    - Configurable context enhancement
    - All features via constructor params (KISS principle)
    """

    def __init__(
        self,
        model: SentenceTransformer,
        language: str,
        # Core sliding window params
        window_size_tokens: int = 7,
        step_size_tokens: int = 1,
        splitting_threshold: float = 0.5,
        min_split_segment_length_words: int = 3,
        # Enhancement: Adaptive window sizing
        adaptive_window: bool = False,
        min_window_size: int = 3,
        max_window_size: int = 10,
        # Enhancement: Parallelization (for long texts)
        use_parallel: bool = False,
        parallel_threshold_windows: int = 200,
        batch_size: int = 32,
        # Enhancement: Context preservation
        add_document_context: bool = False,
        context_length_chars: int = 200,
    ):
        """
        Initialize enhanced sliding window splitter.

        Args:
            model: SentenceTransformer for embeddings
            language: Language code (en, de, fr, es, nl)
            window_size_tokens: Base window size (used if not adaptive)
            step_size_tokens: Sliding window step
            splitting_threshold: Cosine similarity threshold for splits
            min_split_segment_length_words: Minimum chunk size
            adaptive_window: Enable adaptive window sizing
            min_window_size: Min window (if adaptive)
            max_window_size: Max window (if adaptive)
            use_parallel: Enable parallel embedding generation
            parallel_threshold_windows: Use parallel if windows > threshold
            batch_size: Batch size for embedding generation
            add_document_context: Prepend document context to chunks
            context_length_chars: Length of context to prepend
        """
        super().__init__(language)
        self.model = model
        self.window_size_tokens = window_size_tokens
        self.step_size_tokens = step_size_tokens
        self.splitting_threshold = splitting_threshold
        self.min_split_segment_length_words = min_split_segment_length_words

        # Enhancement flags
        self.adaptive_window = adaptive_window
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.use_parallel = use_parallel
        self.parallel_threshold_windows = parallel_threshold_windows
        self.batch_size = batch_size
        self.add_document_context = add_document_context
        self.context_length_chars = context_length_chars

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Chunk text segments using enhanced sliding window.

        Args:
            text_segments: List of text segments to chunk

        Returns:
            List of chunks
        """
        results = []

        # Generate document context once (if enabled)
        doc_context = self._generate_context(text_segments) if self.add_document_context else None

        for segment in text_segments:
            tokens = segment.split()

            # Skip short segments
            if len(tokens) < self.window_size_tokens * 2:
                results.append(segment)
                continue

            # Determine window size (adaptive or fixed)
            if self.adaptive_window:
                window_size = self._calculate_adaptive_window_size(tokens)
            else:
                window_size = self.window_size_tokens

            # Create sliding windows
            windows = self._create_sliding_windows(tokens, window_size)
            window_texts = [" ".join(window) for window in windows]

            # Generate embeddings (parallel or sequential)
            embeddings = self._generate_embeddings(window_texts)

            # Find split points based on similarity
            split_indices = self._find_split_points(embeddings, window_size)

            # Create chunks from split indices
            chunks = self._split_by_indices(segment, tokens, split_indices)

            # Add document context if enabled
            if doc_context:
                chunks = [f"{doc_context}\n\n{chunk}" for chunk in chunks]

            results.extend(chunks)

        return results

    def _calculate_adaptive_window_size(self, tokens: List[str]) -> int:
        """
        Calculate window size based on text density.

        Heuristic:
        - Short tokens (avg < 5 chars): Larger window
        - Long tokens (avg > 8 chars): Smaller window
        """
        avg_token_length = sum(len(t) for t in tokens) / len(tokens)

        if avg_token_length < 5:
            return self.max_window_size
        elif avg_token_length > 8:
            return self.min_window_size
        else:
            # Linear interpolation
            ratio = (avg_token_length - 5) / 3
            size = self.max_window_size - ratio * (self.max_window_size - self.min_window_size)
            return int(max(self.min_window_size, min(self.max_window_size, size)))

    def _create_sliding_windows(
        self,
        tokens: List[str],
        window_size: int,
    ) -> List[List[str]]:
        """Create sliding windows over tokens (DRY utility)."""
        windows = []
        for i in range(0, max(1, len(tokens) - window_size + 1), self.step_size_tokens):
            window = tokens[i:i + window_size]
            if len(window) == window_size:  # Only full windows
                windows.append(window)
        return windows

    def _generate_embeddings(self, window_texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with optional parallelization.

        Uses built-in sentence-transformers parallelization.
        """
        # Use parallel encoding for long texts
        if self.use_parallel and len(window_texts) > self.parallel_threshold_windows:
            embeddings = self.model.encode_multi_process(
                window_texts,
                pool_size=max(1, cpu_count() - 1),
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        else:
            # Sequential encoding for short texts
            embeddings = self.model.encode(
                window_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        return embeddings

    def _find_split_points(
        self,
        embeddings: np.ndarray,
        window_size: int,
    ) -> List[int]:
        """Find split points based on embedding similarities."""
        from sklearn.metrics.pairwise import cosine_similarity

        split_indices = []

        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0, 0]

            if similarity < self.splitting_threshold:
                # Map window index to token index
                token_idx = i * self.step_size_tokens + window_size
                split_indices.append(token_idx)

        return split_indices

    def _split_by_indices(
        self,
        original_text: str,
        tokens: List[str],
        split_indices: List[int],
    ) -> List[str]:
        """Split text by token indices."""
        if not split_indices:
            return [original_text]

        chunks = []
        start = 0

        for idx in split_indices:
            chunk_tokens = tokens[start:idx]
            if len(chunk_tokens) >= self.min_split_segment_length_words:
                chunks.append(" ".join(chunk_tokens))
            start = idx

        # Add remaining tokens
        if start < len(tokens):
            remaining_tokens = tokens[start:]
            if len(remaining_tokens) >= self.min_split_segment_length_words:
                chunks.append(" ".join(remaining_tokens))

        return chunks

    def _generate_context(self, text_segments: List[str]) -> str:
        """
        Generate document context (simple approach).

        Uses first N characters as context (Anthropic-inspired).
        """
        full_text = " ".join(text_segments)

        if len(full_text) <= self.context_length_chars:
            return full_text

        # Take first N chars, break at word boundary
        context = full_text[:self.context_length_chars]
        last_space = context.rfind(" ")

        if last_space > self.context_length_chars * 0.8:
            context = context[:last_space]

        return context + "..."
```

**Key Design Decisions:**

1. ✅ **Single class** instead of 4 separate classes
2. ✅ **Configuration flags** instead of inheritance hierarchy
3. ✅ **Built-in parallelization** from sentence-transformers
4. ✅ **Simple context** (first N chars) instead of LLM-based
5. ✅ **Extends existing** `TextChunker` base class

### 3.3 Component 2: Medical-Aware Punctuation Chunker

**File:** `phentrieve/text_processing/chunkers.py` (add to existing)

```python
class MedicalPunctuationChunker(FineGrainedPunctuationChunker):
    """
    Extends FineGrainedPunctuationChunker with medical abbreviation support.

    Prevents incorrect splits at medical abbreviations like:
    - pt., pts., dx., hx., fx., rx., tx.
    - b.i.d., t.i.d., q.d., p.r.n.
    - mg., mcg., mL., cc., kg.
    """

    def __init__(
        self,
        language: str,
        include_medical_abbreviations: bool = True,
        additional_abbreviations: Optional[List[str]] = None,
    ):
        """
        Initialize medical-aware punctuation chunker.

        Args:
            language: Language code
            include_medical_abbreviations: Load standard medical abbreviations
            additional_abbreviations: User-provided abbreviations to protect
        """
        super().__init__(language)

        if include_medical_abbreviations:
            medical_abbrevs = self._load_medical_abbreviations(language)
            self.protected_patterns.extend(medical_abbrevs)

        if additional_abbreviations:
            self.protected_patterns.extend(additional_abbreviations)

    def _load_medical_abbreviations(self, language: str) -> List[str]:
        """
        Load medical abbreviations from resources.

        Falls back to English if language not available.
        """
        try:
            from phentrieve.text_processing.default_lang_resources import load_language_resource

            medical_abbrevs = load_language_resource(
                "medical_abbreviations.json",
                language,
                "abbreviations",
            )

            if medical_abbrevs:
                return medical_abbrevs
        except (FileNotFoundError, KeyError):
            pass

        # Fallback to English abbreviations
        return self._get_default_medical_abbreviations()

    @staticmethod
    def _get_default_medical_abbreviations() -> List[str]:
        """Get default medical abbreviations (English)."""
        return [
            # Patient/diagnostic
            "pt.", "pts.", "dx.", "hx.", "fx.", "rx.", "tx.",
            "s/p", "w/", "c/o", "r/o",
            # Physical exam
            "HEENT", "CV", "resp.", "GI", "GU", "neuro.",
            # Dosing
            "b.i.d.", "t.i.d.", "q.d.", "q.o.d.", "p.r.n.",
            "q.i.d.", "q.h.", "q.4h.", "q.6h.",
            # Units
            "mg.", "mcg.", "mL.", "cc.", "kg.", "lb.",
            "cm.", "mm.", "ft.", "in.",
            # Common
            "Dr.", "Prof.", "vs.", "etc.", "i.e.", "e.g.",
        ]
```

**Medical Abbreviations Resource File:**

**Create:** `phentrieve/text_processing/default_lang_resources/medical_abbreviations.json`

```json
{
  "en": {
    "abbreviations": [
      "pt.", "pts.", "dx.", "hx.", "fx.", "rx.", "tx.",
      "s/p", "w/", "c/o", "r/o",
      "HEENT", "CV", "resp.", "GI", "GU", "neuro.",
      "b.i.d.", "t.i.d.", "q.d.", "q.o.d.", "p.r.n.",
      "mg.", "mcg.", "mL.", "cc.", "kg.", "lb.",
      "Dr.", "Prof.", "vs.", "etc.", "i.e.", "e.g."
    ]
  },
  "de": {
    "abbreviations": [
      "Pat.", "Diagn.", "Anamnese", "Rez.", "Ther.",
      "z.B.", "d.h.", "u.a.", "v.a.", "ggf.",
      "mg.", "kg.", "cm.", "ml.",
      "Dr.", "Prof."
    ]
  },
  "fr": {
    "abbreviations": [
      "pt.", "diag.", "rx.", "tx.",
      "p.ex.", "c.-à-d.", "etc.",
      "mg.", "kg.", "cm.", "ml.",
      "Dr.", "Prof."
    ]
  }
}
```

### 3.4 Component 3: Chunk Post-Processor

**File:** `phentrieve/text_processing/chunkers.py` (add to existing)

```python
class ChunkPostProcessor(TextChunker):
    """
    Post-processing transformations for chunks.

    Features:
    - Add overlap between consecutive chunks for context preservation
    - Configurable overlap size
    """

    def __init__(
        self,
        language: str,
        overlap_words: int = 3,
    ):
        """
        Initialize chunk post-processor.

        Args:
            language: Language code
            overlap_words: Number of words to overlap between chunks
        """
        super().__init__(language)
        self.overlap_words = overlap_words

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Add overlap to chunks.

        Args:
            text_segments: Input chunks

        Returns:
            Chunks with overlap added
        """
        if len(text_segments) <= 1:
            return text_segments

        overlapped = []

        for i, segment in enumerate(text_segments):
            tokens = segment.split()

            # Add suffix from previous chunk
            if i > 0:
                prev_tokens = text_segments[i - 1].split()
                suffix = prev_tokens[-self.overlap_words:]
                tokens = suffix + tokens

            # Add prefix from next chunk
            if i < len(text_segments) - 1:
                next_tokens = text_segments[i + 1].split()
                prefix = next_tokens[:self.overlap_words]
                tokens = tokens + prefix

            overlapped.append(" ".join(tokens))

        return overlapped
```

### 3.5 Simplified Strategy Configuration

**File:** `phentrieve/config.py` (replace old strategies)

```python
"""
Simplified chunking strategies for alpha release.

Old strategies removed - clean slate approach.
"""

# Strategy 1: FAST - Minimal chunking for speed
FAST_CONFIG = [
    {"type": "paragraph"},
    {"type": "sentence"},
]

# Strategy 2: BALANCED - Good default for most use cases
BALANCED_CONFIG = [
    {"type": "paragraph"},
    {"type": "sentence"},
    {
        "type": "enhanced_sliding_window",
        "config": {
            "window_size_tokens": 7,
            "adaptive_window": True,
            "use_parallel": True,
            "parallel_threshold_windows": 200,
        }
    },
]

# Strategy 3: PRECISE - Maximum quality for clinical text
PRECISE_CONFIG = [
    {"type": "paragraph"},
    {"type": "sentence"},
    {"type": "medical_punctuation"},
    {
        "type": "enhanced_sliding_window",
        "config": {
            "window_size_tokens": 5,
            "adaptive_window": True,
            "splitting_threshold": 0.6,
            "use_parallel": True,
        }
    },
    {"type": "chunk_overlap", "config": {"overlap_words": 3}},
    {"type": "final_cleanup"},
]

# Map strategy names to configurations
CHUNKING_STRATEGIES = {
    "fast": FAST_CONFIG,
    "balanced": BALANCED_CONFIG,
    "precise": PRECISE_CONFIG,
}

def get_chunking_strategy(name: str) -> List[Dict[str, Any]]:
    """
    Get chunking strategy by name.

    Args:
        name: Strategy name (fast/balanced/precise)

    Returns:
        Pipeline configuration

    Raises:
        ValueError: If strategy not found
    """
    if name not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy: {name}. "
            f"Available: {list(CHUNKING_STRATEGIES.keys())}"
        )
    return CHUNKING_STRATEGIES[name]
```

**Usage Example:**

```python
# Simple and clear - 3 options instead of 7
pipeline_config = get_chunking_strategy("balanced")  # Default
pipeline_config = get_chunking_strategy("fast")      # Speed priority
pipeline_config = get_chunking_strategy("precise")   # Quality priority
```

---

## 4. Benchmarking Framework

### 4.1 Architecture - SOLID Design

**Problem:** Original plan had 300-line God class doing everything.

**Solution:** Split into 5 focused components following Single Responsibility Principle.

```
BenchmarkOrchestrator (30 lines)
    ↓ uses
DatasetLoader (60 lines)
PipelineRunner (50 lines)
MetricsCalculator (80 lines)
ResultsAggregator (40 lines)
BenchmarkReporter (60 lines)

Total: ~320 lines across 6 classes vs 300 lines in 1 class
Benefits: Testable, extensible, maintainable
```

### 4.2 Component Interfaces

**File:** `phentrieve/evaluation/benchmarking/interfaces.py` (NEW)

```python
"""
Protocol interfaces for benchmarking components.

Enables dependency injection and mocking for tests.
"""

from typing import Protocol, List, Dict, Any
from pathlib import Path

class DatasetLoader(Protocol):
    """Interface for loading benchmark datasets."""

    def load_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Load dataset by ID."""
        ...

class PipelineRunner(Protocol):
    """Interface for running chunking + retrieval pipeline."""

    def process_document(
        self,
        text: str,
        strategy: str,
        language: str,
    ) -> Dict[str, Any]:
        """Process document with given strategy."""
        ...

class MetricsCalculator(Protocol):
    """Interface for calculating evaluation metrics."""

    def calculate_metrics(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        full_text: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate all metrics for document."""
        ...
```

### 4.3 Implementation - Dataset Loader

**File:** `phentrieve/evaluation/benchmarking/dataset_loader.py` (NEW)

```python
"""
Dataset loading for benchmarking.

Handles:
- Dataset catalog management
- JSON file loading
- Dataset validation
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class BenchmarkDatasetLoader:
    """
    Load benchmark datasets from standard directory structure.

    Directory structure:
        data/benchmark_datasets/
        ├── dataset_catalog.json
        └── {dataset_id}/
            ├── metadata.json
            └── annotations/
                └── *.json
    """

    def __init__(self, datasets_root: Path):
        """
        Initialize dataset loader.

        Args:
            datasets_root: Root directory for benchmark datasets
        """
        self.datasets_root = Path(datasets_root)
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> Dict[str, Any]:
        """Load dataset catalog."""
        catalog_path = self.datasets_root / "dataset_catalog.json"

        if not catalog_path.exists():
            logger.warning(f"Dataset catalog not found: {catalog_path}")
            return {"datasets": []}

        with open(catalog_path) as f:
            return json.load(f)

    def load_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Load dataset by ID.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dict with 'metadata' and 'documents' keys

        Raises:
            FileNotFoundError: If dataset not found
        """
        dataset_dir = self.datasets_root / dataset_id

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

        # Load metadata
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load documents
        annotations_dir = dataset_dir / "annotations"
        documents = []

        for doc_file in sorted(annotations_dir.glob("*.json")):
            with open(doc_file) as f:
                doc = json.load(f)
                documents.append(doc)

        logger.info(f"Loaded dataset '{dataset_id}': {len(documents)} documents")

        return {
            "dataset_id": dataset_id,
            "metadata": metadata,
            "documents": documents,
        }

    def list_available_datasets(self) -> List[str]:
        """List all available dataset IDs."""
        return [ds["id"] for ds in self.catalog.get("datasets", [])]
```

### 4.4 Implementation - Metrics Calculator

**File:** `phentrieve/evaluation/benchmarking/metrics_calculator.py` (NEW)

```python
"""
Unified metrics calculation for benchmarking.

Calculates:
- Term-based metrics (P/R/F1)
- Span-based metrics (with approximate matching)
- Chunking quality metrics
"""

from typing import Dict, Any, List
import numpy as np
from phentrieve.evaluation.span_metrics import SpanBasedEvaluator
from phentrieve.evaluation.retrieval_metrics import calculate_semantically_aware_set_based_prf1

class BenchmarkMetricsCalculator:
    """
    Calculate all evaluation metrics.

    Single Responsibility: Metric calculation only.
    """

    def __init__(
        self,
        enable_span_metrics: bool = True,
        span_overlap_threshold: float = 0.5,
        span_match_mode: str = "partial",
    ):
        """
        Initialize metrics calculator.

        Args:
            enable_span_metrics: Whether to calculate span-based metrics
            span_overlap_threshold: IoU threshold for partial matches
            span_match_mode: Matching mode (strict/exact/partial/type)
        """
        self.enable_span_metrics = enable_span_metrics

        if enable_span_metrics:
            self.span_evaluator = SpanBasedEvaluator(
                overlap_threshold=span_overlap_threshold,
                match_mode=span_match_mode,
            )

    def calculate_metrics(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        full_text: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for a document.

        Args:
            predicted: Predicted HPO annotations
            ground_truth: Gold standard annotations
            full_text: Original document text
            chunks: Chunking results

        Returns:
            Dict with 'term_metrics', 'span_metrics', 'chunking_quality'
        """
        metrics = {}

        # 1. Term-based metrics (existing)
        metrics["term_metrics"] = calculate_semantically_aware_set_based_prf1(
            extracted_annotations=predicted,
            ground_truth_annotations=ground_truth,
        )

        # 2. Span-based metrics (NEW)
        if self.enable_span_metrics and ground_truth:
            metrics["span_metrics"] = self.span_evaluator.evaluate(
                predicted_annotations=predicted,
                gold_annotations=ground_truth,
                full_text=full_text,
            )

        # 3. Chunking quality metrics
        metrics["chunking_quality"] = self._calculate_chunking_quality(chunks, full_text)

        return metrics

    @staticmethod
    def _calculate_chunking_quality(
        chunks: List[Dict[str, Any]],
        full_text: str,
    ) -> Dict[str, float]:
        """Calculate chunking quality metrics."""
        chunk_lengths_words = [len(c["text"].split()) for c in chunks]
        chunk_lengths_chars = [len(c["text"]) for c in chunks]

        return {
            "num_chunks": len(chunks),
            "avg_chunk_length_words": float(np.mean(chunk_lengths_words)) if chunks else 0.0,
            "std_chunk_length_words": float(np.std(chunk_lengths_words)) if chunks else 0.0,
            "min_chunk_length_words": int(min(chunk_lengths_words)) if chunks else 0,
            "max_chunk_length_words": int(max(chunk_lengths_words)) if chunks else 0,
            "median_chunk_length_words": float(np.median(chunk_lengths_words)) if chunks else 0.0,
            "coverage_ratio": sum(chunk_lengths_chars) / len(full_text) if full_text else 1.0,
        }
```

### 4.5 Implementation - Benchmark Orchestrator

**File:** `phentrieve/evaluation/benchmarking/benchmark.py` (NEW)

```python
"""
Main benchmark orchestrator using dependency injection.

Coordinates dataset loading, pipeline execution, metrics calculation,
aggregation, and reporting.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for chunking benchmark."""

    dataset_ids: List[str]
    chunking_strategies: List[str]
    output_dir: Path
    enable_span_metrics: bool = True
    span_overlap_threshold: float = 0.5
    save_per_document_results: bool = False

class ChunkingBenchmark:
    """
    Orchestrate chunking benchmark execution.

    Uses dependency injection for all components (testable, extensible).
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        dataset_loader: "BenchmarkDatasetLoader",
        pipeline_runner: "BenchmarkPipelineRunner",
        metrics_calculator: "BenchmarkMetricsCalculator",
        results_aggregator: "BenchmarkResultsAggregator",
        reporter: "BenchmarkReporter",
    ):
        """
        Initialize benchmark with injected dependencies.

        Args:
            config: Benchmark configuration
            dataset_loader: Component for loading datasets
            pipeline_runner: Component for running pipelines
            metrics_calculator: Component for calculating metrics
            results_aggregator: Component for aggregating results
            reporter: Component for reporting/visualization
        """
        self.config = config
        self.dataset_loader = dataset_loader
        self.pipeline_runner = pipeline_runner
        self.metrics_calculator = metrics_calculator
        self.results_aggregator = results_aggregator
        self.reporter = reporter

    def run(self) -> Dict[str, Any]:
        """
        Run complete benchmark.

        Returns:
            Aggregated benchmark results
        """
        logger.info("Starting chunking benchmark")
        logger.info(f"Datasets: {self.config.dataset_ids}")
        logger.info(f"Strategies: {self.config.chunking_strategies}")

        all_results = {
            "config": {
                "datasets": self.config.dataset_ids,
                "strategies": self.config.chunking_strategies,
                "span_metrics_enabled": self.config.enable_span_metrics,
            },
            "datasets": {},
        }

        # Process each dataset
        for dataset_id in self.config.dataset_ids:
            logger.info(f"Benchmarking dataset: {dataset_id}")

            # Load dataset
            dataset = self.dataset_loader.load_dataset(dataset_id)

            dataset_results = {
                "metadata": dataset["metadata"],
                "strategies": {},
            }

            # Evaluate each strategy
            for strategy in self.config.chunking_strategies:
                logger.info(f"  Evaluating strategy: {strategy}")

                strategy_results = self._evaluate_strategy(
                    strategy=strategy,
                    dataset=dataset,
                )

                dataset_results["strategies"][strategy] = strategy_results

            all_results["datasets"][dataset_id] = dataset_results

        # Aggregate results across datasets
        all_results["aggregated"] = self.results_aggregator.aggregate(all_results)

        # Generate reports
        self.reporter.generate_reports(all_results, self.config.output_dir)

        logger.info("Benchmark complete")
        return all_results

    def _evaluate_strategy(
        self,
        strategy: str,
        dataset: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate single strategy on dataset."""
        documents = dataset["documents"]
        language = dataset["metadata"].get("language", "en")

        doc_results = []
        total_time = 0.0

        for doc in documents:
            start_time = time.time()

            # Run pipeline
            pipeline_result = self.pipeline_runner.process_document(
                text=doc["full_text"],
                strategy=strategy,
                language=language,
            )

            elapsed = time.time() - start_time
            total_time += elapsed

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(
                predicted=pipeline_result["predicted_annotations"],
                ground_truth=doc.get("annotations", []),
                full_text=doc["full_text"],
                chunks=pipeline_result["chunks"],
            )

            doc_result = {
                "doc_id": doc.get("doc_id", "unknown"),
                "metrics": metrics,
                "processing_time_sec": elapsed,
            }

            doc_results.append(doc_result)

        # Aggregate document results
        aggregated_metrics = self._aggregate_document_results(doc_results)

        return {
            "aggregated_metrics": aggregated_metrics,
            "total_processing_time_sec": total_time,
            "avg_processing_time_per_doc_sec": total_time / len(documents) if documents else 0,
            "num_documents": len(documents),
            "per_document_results": doc_results if self.config.save_per_document_results else [],
        }

    @staticmethod
    def _aggregate_document_results(doc_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across documents."""
        if not doc_results:
            return {}

        # Extract term metrics
        term_precisions = [d["metrics"]["term_metrics"]["precision"] for d in doc_results]
        term_recalls = [d["metrics"]["term_metrics"]["recall"] for d in doc_results]
        term_f1s = [d["metrics"]["term_metrics"]["f1_score"] for d in doc_results]

        # Extract chunking quality
        num_chunks = [d["metrics"]["chunking_quality"]["num_chunks"] for d in doc_results]
        avg_chunk_lengths = [d["metrics"]["chunking_quality"]["avg_chunk_length_words"] for d in doc_results]

        aggregated = {
            # Term metrics
            "avg_precision": float(np.mean(term_precisions)),
            "avg_recall": float(np.mean(term_recalls)),
            "avg_f1_score": float(np.mean(term_f1s)),
            "std_f1_score": float(np.std(term_f1s)),

            # Chunking quality
            "avg_num_chunks_per_doc": float(np.mean(num_chunks)),
            "avg_chunk_length_words": float(np.mean(avg_chunk_lengths)),
        }

        # Add span metrics if available
        if "span_metrics" in doc_results[0]["metrics"]:
            span_f1s = [d["metrics"]["span_metrics"]["span_f1_score"] for d in doc_results]
            aggregated["avg_span_f1_score"] = float(np.mean(span_f1s))
            aggregated["std_span_f1_score"] = float(np.std(span_f1s))

        return aggregated
```

**Key Design Decisions:**

1. ✅ **Dependency Injection:** All components passed to constructor (testable)
2. ✅ **Single Responsibility:** Each component has one job
3. ✅ **Small Methods:** Each method < 50 lines
4. ✅ **Clear Interfaces:** Protocol-based contracts
5. ✅ **Logging:** Structured logging throughout

### 4.6 CLI Integration

**File:** `phentrieve/cli/benchmark_commands.py` (extend existing)

```python
@benchmark_app.command("chunking")
def benchmark_chunking(
    datasets: List[str] = typer.Option(
        None,
        "--dataset", "-d",
        help="Dataset IDs to benchmark (repeat for multiple, or use 'all')",
    ),
    strategies: List[str] = typer.Option(
        None,
        "--strategy", "-s",
        help="Chunking strategies to evaluate (repeat for multiple, or use 'all')",
    ),
    output_dir: Path = typer.Option(
        Path("results/chunking_benchmarks"),
        "--output", "-o",
        help="Output directory for results",
    ),
    enable_span_metrics: bool = typer.Option(
        True,
        "--span-metrics/--no-span-metrics",
        help="Enable span-based evaluation metrics",
    ),
    save_details: bool = typer.Option(
        False,
        "--save-details",
        help="Save per-document results",
    ),
):
    """
    Benchmark chunking strategies on annotated datasets.

    Examples:
        # Benchmark specific dataset and strategies
        phentrieve benchmark chunking -d GSC_plus -s balanced -s precise

        # Benchmark all available
        phentrieve benchmark chunking -d all -s all

        # With detailed results
        phentrieve benchmark chunking -d GSC_plus -s balanced --save-details
    """
    from phentrieve.evaluation.benchmarking import (
        BenchmarkConfig,
        ChunkingBenchmark,
        BenchmarkDatasetLoader,
        BenchmarkPipelineRunner,
        BenchmarkMetricsCalculator,
        BenchmarkResultsAggregator,
        BenchmarkReporter,
    )
    from phentrieve.config import get_config

    config_obj = get_config()

    # Resolve "all" keywords
    loader = BenchmarkDatasetLoader(config_obj.data_root_dir / "benchmark_datasets")

    if datasets and "all" in datasets:
        datasets = loader.list_available_datasets()
    elif not datasets:
        typer.echo("Error: Must specify at least one dataset or use 'all'")
        raise typer.Exit(1)

    if strategies and "all" in strategies:
        strategies = ["fast", "balanced", "precise"]
    elif not strategies:
        strategies = ["balanced"]

    # Create config
    benchmark_config = BenchmarkConfig(
        dataset_ids=list(datasets),
        chunking_strategies=list(strategies),
        output_dir=output_dir,
        enable_span_metrics=enable_span_metrics,
        save_per_document_results=save_details,
    )

    # Initialize components (dependency injection)
    from phentrieve.retrieval import DenseRetriever

    retriever = DenseRetriever(
        collection_name=config_obj.default_model,
        model_name=config_obj.default_model,
    )

    pipeline_runner = BenchmarkPipelineRunner(retriever=retriever)
    metrics_calculator = BenchmarkMetricsCalculator(
        enable_span_metrics=enable_span_metrics,
        span_overlap_threshold=0.5,
    )
    results_aggregator = BenchmarkResultsAggregator()
    reporter = BenchmarkReporter()

    # Run benchmark
    benchmark = ChunkingBenchmark(
        config=benchmark_config,
        dataset_loader=loader,
        pipeline_runner=pipeline_runner,
        metrics_calculator=metrics_calculator,
        results_aggregator=results_aggregator,
        reporter=reporter,
    )

    typer.echo("Starting benchmark...")
    results = benchmark.run()

    typer.echo(f"\nResults saved to: {output_dir}")
    typer.echo("\nSummary:")
    for strategy, metrics in results["aggregated"].items():
        typer.echo(f"  {strategy}: F1={metrics['overall_f1']:.3f}")
```

---

## 5. Span-Based Evaluation

### 5.1 Design - DRY Compliant

**Shared Utilities:** Extract common annotation parsing logic.

**File:** `phentrieve/evaluation/annotation_utils.py` (NEW)

```python
"""
Shared utilities for annotation handling (DRY principle).

Provides:
- Normalized annotation format (dataclasses)
- Parsing from various formats
- Text span utilities
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Span:
    """Normalized text span with character offsets."""

    start_char: int
    end_char: int
    text: str
    hpo_id: str = ""
    confidence: float = 1.0

    def overlaps_with(self, other: "Span") -> float:
        """
        Calculate Intersection over Union (IoU) with another span.

        Returns:
            IoU ratio (0.0 to 1.0)
        """
        intersection_start = max(self.start_char, other.start_char)
        intersection_end = min(self.end_char, other.end_char)

        if intersection_end <= intersection_start:
            return 0.0

        intersection_length = intersection_end - intersection_start
        union_start = min(self.start_char, other.start_char)
        union_end = max(self.end_char, other.end_char)
        union_length = union_end - union_start

        return intersection_length / union_length if union_length > 0 else 0.0

    def exact_boundary_match(self, other: "Span") -> bool:
        """Check if boundaries exactly match."""
        return (
            self.start_char == other.start_char and
            self.end_char == other.end_char
        )

    def type_match(self, other: "Span") -> bool:
        """Check if HPO IDs match."""
        return self.hpo_id == other.hpo_id

@dataclass
class Annotation:
    """Normalized HPO annotation format."""

    hpo_id: str
    label: str
    evidence_spans: List[Span] = field(default_factory=list)
    assertion_status: str = "affirmed"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """
        Parse annotation from various formats (DRY utility).

        Handles field name variations:
        - hpo_id or id
        - label or name
        - evidence_spans or text_attributions
        """
        # Normalize HPO ID
        hpo_id = data.get("hpo_id") or data.get("id", "")

        # Normalize label
        label = data.get("label") or data.get("name", "")

        # Normalize evidence spans
        raw_spans = (
            data.get("evidence_spans") or
            data.get("text_attributions") or
            []
        )

        spans = []
        for span_data in raw_spans:
            span = Span(
                start_char=span_data.get("start_char") or span_data.get("start", 0),
                end_char=span_data.get("end_char") or span_data.get("end", 0),
                text=span_data.get("text_snippet") or span_data.get("text", ""),
                hpo_id=hpo_id,
                confidence=span_data.get("confidence") or span_data.get("score", 1.0),
            )
            spans.append(span)

        return cls(
            hpo_id=hpo_id,
            label=label,
            evidence_spans=spans,
            assertion_status=data.get("assertion_status", "affirmed"),
        )

def parse_annotations(raw_annotations: List[Dict[str, Any]]) -> List[Annotation]:
    """
    Parse list of annotations from dicts (DRY utility).

    Args:
        raw_annotations: List of annotation dicts

    Returns:
        List of normalized Annotation objects
    """
    return [Annotation.from_dict(ann) for ann in raw_annotations]
```

### 5.2 Span-Based Evaluator

**File:** `phentrieve/evaluation/span_metrics.py` (NEW)

```python
"""
Span-based evaluation metrics for HPO extraction.

Implements SemEval 2013-style evaluation with configurable matching modes.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from phentrieve.evaluation.annotation_utils import Annotation, Span, parse_annotations

class MatchType(Enum):
    """Type of span match following SemEval 2013 standard."""
    STRICT = ("strict", 1.0)   # Exact boundary + correct HPO term
    EXACT = ("exact", 0.75)    # Exact boundary, any term
    TYPE = ("type", 0.75)      # Any overlap + correct HPO term
    PARTIAL = ("partial", 0.5) # Overlap >= threshold
    NONE = ("none", 0.0)       # No match

    def __init__(self, mode_name: str, weight: float):
        self.mode_name = mode_name
        self.weight = weight

@dataclass
class SpanMatch:
    """Result of matching predicted span to gold span."""
    match_type: MatchType
    predicted_span: Span
    gold_span: Optional[Span]
    overlap_ratio: float
    char_start_diff: int
    char_end_diff: int

class SpanBasedEvaluator:
    """
    Evaluate HPO extraction with span-based metrics.

    Features:
    - Configurable overlap threshold (IoU)
    - Multiple matching modes (strict/exact/partial/type)
    - SemEval 2013-style partial match weighting
    """

    def __init__(
        self,
        overlap_threshold: float = 0.5,
        match_mode: str = "partial",
        partial_match_weight: float = 0.5,
    ):
        """
        Initialize span-based evaluator.

        Args:
            overlap_threshold: Minimum IoU for partial match (0.0-1.0)
            match_mode: Matching mode (strict/exact/partial/type)
            partial_match_weight: Weight for partial matches in P/R (default: 0.5 per SemEval)
        """
        self.overlap_threshold = overlap_threshold
        self.match_mode = match_mode
        self.partial_match_weight = partial_match_weight

    def evaluate(
        self,
        predicted_annotations: List[Dict[str, Any]],
        gold_annotations: List[Dict[str, Any]],
        full_text: str,
    ) -> Dict[str, Any]:
        """
        Evaluate predicted annotations against gold standard.

        Args:
            predicted_annotations: Extracted HPO terms with text attributions
            gold_annotations: Ground truth HPO terms with evidence spans
            full_text: Original document text (for validation)

        Returns:
            Dictionary with span-based metrics
        """
        # Parse annotations (DRY utility)
        predicted = parse_annotations(predicted_annotations)
        gold = parse_annotations(gold_annotations)

        # Flatten spans from annotations
        predicted_spans = self._extract_all_spans(predicted)
        gold_spans = self._extract_all_spans(gold)

        # Match predicted to gold spans
        matches = self._match_spans(predicted_spans, gold_spans)

        # Calculate metrics
        metrics = self._calculate_metrics(matches, len(predicted_spans), len(gold_spans))

        # Add match details (optional, for debugging)
        metrics["num_predicted_spans"] = len(predicted_spans)
        metrics["num_gold_spans"] = len(gold_spans)
        metrics["match_type_counts"] = self._count_match_types(matches)

        return metrics

    @staticmethod
    def _extract_all_spans(annotations: List[Annotation]) -> List[Span]:
        """Extract all evidence spans from annotations."""
        spans = []
        for annotation in annotations:
            for span in annotation.evidence_spans:
                # Ensure HPO ID is set
                if not span.hpo_id:
                    span.hpo_id = annotation.hpo_id
                spans.append(span)
        return spans

    def _match_spans(
        self,
        predicted_spans: List[Span],
        gold_spans: List[Span],
    ) -> List[SpanMatch]:
        """
        Match predicted spans to gold spans (greedy algorithm).

        Strategy:
        1. Calculate overlap matrix (all predicted vs all gold)
        2. Greedy matching: assign each predicted to best gold match
        3. Record match type and overlap ratio
        """
        matches = []
        matched_gold_indices = set()

        for pred_span in predicted_spans:
            best_match_type = MatchType.NONE
            best_overlap = 0.0
            best_gold_idx = None

            # Find best match among gold spans
            for gold_idx, gold_span in enumerate(gold_spans):
                if gold_idx in matched_gold_indices:
                    continue

                # Calculate overlap
                overlap_ratio = pred_span.overlaps_with(gold_span)

                # Determine match type
                match_type = self._determine_match_type(pred_span, gold_span, overlap_ratio)

                # Update best match
                if match_type != MatchType.NONE and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_match_type = match_type
                    best_gold_idx = gold_idx

            # Record match
            if best_match_type != MatchType.NONE and best_gold_idx is not None:
                matched_gold_indices.add(best_gold_idx)
                gold_span = gold_spans[best_gold_idx]

                matches.append(SpanMatch(
                    match_type=best_match_type,
                    predicted_span=pred_span,
                    gold_span=gold_span,
                    overlap_ratio=best_overlap,
                    char_start_diff=abs(pred_span.start_char - gold_span.start_char),
                    char_end_diff=abs(pred_span.end_char - gold_span.end_char),
                ))
            else:
                # No match found
                matches.append(SpanMatch(
                    match_type=MatchType.NONE,
                    predicted_span=pred_span,
                    gold_span=None,
                    overlap_ratio=0.0,
                    char_start_diff=-1,
                    char_end_diff=-1,
                ))

        return matches

    def _determine_match_type(
        self,
        pred_span: Span,
        gold_span: Span,
        overlap_ratio: float,
    ) -> MatchType:
        """Determine match type based on overlap and HPO term (SemEval 2013)."""
        exact_boundary = pred_span.exact_boundary_match(gold_span)
        hpo_match = pred_span.type_match(gold_span)
        partial_overlap = overlap_ratio >= self.overlap_threshold

        # SemEval hierarchy
        if exact_boundary and hpo_match:
            return MatchType.STRICT
        elif exact_boundary:
            return MatchType.EXACT
        elif partial_overlap and hpo_match:
            return MatchType.TYPE
        elif partial_overlap:
            return MatchType.PARTIAL
        else:
            return MatchType.NONE

    def _calculate_metrics(
        self,
        matches: List[SpanMatch],
        num_predicted: int,
        num_gold: int,
    ) -> Dict[str, float]:
        """Calculate P/R/F1 from matches with SemEval weighting."""
        # Count matches by type
        match_counts = {match_type: 0 for match_type in MatchType}
        for match in matches:
            match_counts[match.match_type] += 1

        # Calculate weighted correct count based on match mode
        correct_count = self._calculate_correct_count(match_counts)

        # Precision, Recall, F1
        precision = correct_count / num_predicted if num_predicted > 0 else 0.0
        recall = correct_count / num_gold if num_gold > 0 else 0.0

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        # Boundary error statistics
        boundary_errors = [
            (m.char_start_diff + m.char_end_diff) / 2
            for m in matches
            if m.match_type != MatchType.NONE
        ]

        return {
            "span_precision": precision,
            "span_recall": recall,
            "span_f1_score": f1_score,
            "avg_boundary_error_chars": float(np.mean(boundary_errors)) if boundary_errors else 0.0,
            "max_boundary_error_chars": float(max(boundary_errors)) if boundary_errors else 0.0,
        }

    def _calculate_correct_count(self, match_counts: Dict[MatchType, int]) -> float:
        """Calculate weighted correct count based on match mode."""
        if self.match_mode == "strict":
            return float(match_counts[MatchType.STRICT])
        elif self.match_mode == "exact":
            return float(match_counts[MatchType.STRICT] + match_counts[MatchType.EXACT])
        elif self.match_mode == "type":
            return float(match_counts[MatchType.STRICT] + match_counts[MatchType.TYPE])
        elif self.match_mode == "partial":
            # SemEval weighting: partial matches count as 0.5
            return float(
                match_counts[MatchType.STRICT] +
                match_counts[MatchType.TYPE] +
                (match_counts[MatchType.EXACT] + match_counts[MatchType.PARTIAL]) * self.partial_match_weight
            )
        else:
            return float(match_counts[MatchType.STRICT])

    @staticmethod
    def _count_match_types(matches: List[SpanMatch]) -> Dict[str, int]:
        """Count matches by type (for reporting)."""
        counts = {}
        for match_type in MatchType:
            counts[match_type.mode_name] = sum(
                1 for m in matches if m.match_type == match_type
            )
        return counts
```

**Key Design Decisions:**

1. ✅ **DRY:** Uses shared `annotation_utils` for parsing
2. ✅ **Dataclasses:** Clean `Span` and `Annotation` objects with methods
3. ✅ **Enum:** Match types with built-in weights
4. ✅ **SemEval Standard:** Follows established evaluation protocol
5. ✅ **Configurable:** Threshold and mode easily adjusted

---

## 6. Implementation Roadmap

### Phase 1: Core Enhancements (2 weeks)

**Week 1: Profile & Enhanced Chunker**

**CRITICAL: Profile before optimizing!**

```bash
# Day 1: Establish baseline
python -m cProfile -o baseline.prof phentrieve text process corpus.txt
python -m pstats baseline.prof
> sort cumtime
> stats 30

# Identify actual bottlenecks (not assumed ones)
# Options: ChromaDB queries? Model loading? Chunking? I/O?
```

- [ ] Run profiling on representative workload
- [ ] Identify top 3 bottlenecks
- [ ] Document findings in `plan/PROFILING-RESULTS.md`
- [ ] Implement `EnhancedSlidingWindowSplitter`
- [ ] Unit tests for adaptive windowing
- [ ] Unit tests for parallel embedding generation
- [ ] Performance benchmarks (before/after)

**Week 2: Medical Abbreviations & Post-Processing**

- [ ] Create `medical_abbreviations.json` (EN, DE, FR)
- [ ] Implement `MedicalPunctuationChunker`
- [ ] Unit tests for abbreviation handling
- [ ] Implement `ChunkPostProcessor` (overlap)
- [ ] Unit tests for overlap addition
- [ ] Integration tests for full pipeline
- [ ] Update CLI to support new strategies
- [ ] Update configuration (`fast`, `balanced`, `precise`)

**Deliverables:**
- 3 enhanced chunker components
- Profiling report showing actual bottlenecks
- Performance measurements (speedup if parallelization needed)
- Simplified configuration (3 strategies)

---

### Phase 2: Benchmarking Framework (2 weeks)

**Week 1: Core Components**

- [ ] Implement `annotation_utils.py` (DRY utilities)
- [ ] Implement `SpanBasedEvaluator`
- [ ] Unit tests for span matching (all match modes)
- [ ] Unit tests for IoU calculation
- [ ] Implement `BenchmarkDatasetLoader`
- [ ] Implement `BenchmarkMetricsCalculator`
- [ ] Unit tests for metrics calculation

**Week 2: Orchestration & CLI**

- [ ] Implement `BenchmarkPipelineRunner`
- [ ] Implement `BenchmarkResultsAggregator`
- [ ] Implement `BenchmarkReporter` (JSON + basic plots)
- [ ] Implement `ChunkingBenchmark` orchestrator
- [ ] Add `phentrieve benchmark chunking` CLI command
- [ ] Integration tests on toy dataset
- [ ] Documentation for CLI usage

**Deliverables:**
- 6 focused components (SRP-compliant)
- Span-based evaluation with approximate matching
- Automated chunking strategy comparison
- CLI for running benchmarks

---

### Phase 3: Dataset Preparation (1 week) - OPTIONAL

**Consider deferring or simplifying:**

- Use existing BiolarkGSC+ / ID-68 datasets (already annotated)
- Convert to Phentrieve format using script from PhenoBERT conversion plan
- Manual validation of 20-30 documents

**If PubMed extraction is implemented:**

- [ ] Implement `PubMedExtractor` (split into API client + parser)
- [ ] CLI command for extraction
- [ ] Extract 20-30 case reports
- [ ] Manual review workflow
- [ ] Documentation

**Deliverables:**
- 20-30 gold standard annotated documents
- Reproducible extraction pipeline (if PubMed)

---

### Phase 4: Polish & Documentation (1-2 weeks)

**Week 1: Documentation & Testing**

- [ ] API documentation (Sphinx)
- [ ] Architecture Decision Records (ADRs) for key decisions
- [ ] Example notebooks
- [ ] Performance tuning guide
- [ ] Update CLAUDE.md with new commands
- [ ] Achieve >80% coverage for new code

**Week 2: Testing & Refinement**

- [ ] Property-based tests (Hypothesis) for chunkers
- [ ] Integration tests for full workflow
- [ ] Performance regression tests
- [ ] Code review and refinement
- [ ] Production-ready polish

**Deliverables:**
- Comprehensive documentation
- High test coverage
- Production-ready code
- Clean, maintainable codebase

---

### Timeline Summary

```
Phase 1: Weeks 1-2   ████████░░░░░░░░░░  Core (profile first!)
Phase 2: Weeks 3-4   ░░░░░░░░████████░░  Benchmarking
Phase 3: Week 5      ░░░░░░░░░░░░░░░░██  Datasets (optional)
Phase 4: Weeks 6-7   ░░░░░░░░░░░░░░██░░  Polish & Docs

Total: 5-7 weeks
```

**Milestones:**
- **Week 2:** Enhanced chunkers with measured performance improvements
- **Week 4:** Full benchmarking framework operational
- **Week 5:** Dataset ready (or using existing)
- **Week 7:** Production release with documentation

---

## 7. Technical Specifications

### 7.1 Dependencies

**Minimal New Dependencies:**

```toml
# pyproject.toml

[project]
dependencies = [
    # Existing dependencies...
    # No new required dependencies! Use what we have.
]

[project.optional-dependencies]
benchmarking = [
    "matplotlib>=3.7.0",  # For visualizations
    "seaborn>=0.12.0",    # For statistical plots
]

# PubMed extraction (if Phase 3 implemented)
pubmed = [
    "biopython>=1.81",
]
```

**Principle:** Minimize dependencies. Use built-in libraries and existing dependencies.

### 7.2 Configuration

**Simplified:** `phentrieve.yaml`

```yaml
# Chunking configuration
chunking:
  default_strategy: "balanced"  # fast/balanced/precise

  # Enhancement options (applied to compatible strategies)
  enhancements:
    adaptive_window: true          # Enable adaptive window sizing
    use_parallel: true             # Enable parallelization
    parallel_threshold: 200        # Use parallel if windows > threshold
    add_context: false             # Add document context to chunks
    overlap_words: 0               # Chunk overlap (0 = disabled)

# Benchmarking configuration
benchmarking:
  datasets_dir: "data/benchmark_datasets"
  results_dir: "results/benchmarks"

  span_metrics:
    enabled: true
    overlap_threshold: 0.5         # IoU threshold for partial matches
    match_mode: "partial"          # strict/exact/partial/type
    partial_match_weight: 0.5      # SemEval standard
```

**Simplified:** 10 params total with clear defaults.

### 7.3 API Changes

**New Modules:**

```
phentrieve/
├── text_processing/
│   ├── chunkers.py                         # UPDATED: Add 3 classes
│   └── default_lang_resources/
│       └── medical_abbreviations.json      # NEW
├── evaluation/
│   ├── annotation_utils.py                 # NEW: DRY utilities
│   ├── span_metrics.py                     # NEW: Span evaluation
│   └── benchmarking/                       # NEW: Benchmark components
│       ├── __init__.py
│       ├── interfaces.py
│       ├── dataset_loader.py
│       ├── pipeline_runner.py
│       ├── metrics_calculator.py
│       ├── results_aggregator.py
│       ├── reporter.py
│       └── benchmark.py
└── config.py                               # UPDATED: Simplified strategies
```

**New CLI Commands:**

```bash
# Benchmarking
phentrieve benchmark chunking --dataset GSC_plus --strategy balanced

# Text processing (simplified strategies)
phentrieve text chunk --strategy fast input.txt       # Speed
phentrieve text chunk --strategy balanced input.txt   # Default
phentrieve text chunk --strategy precise input.txt    # Quality
```

### 7.4 Performance Targets

**Realistic Targets (based on research, validated by profiling):**

| Metric | Baseline | Target | Condition |
|--------|----------|--------|-----------|
| **Batch Processing** | 2-5 sec/doc | 0.5-1.5 sec/doc | 10+ documents, parallelization enabled |
| **Single Document** | 2-5 sec | 2-4 sec | No change expected (overhead ~10%) |
| **Long Documents** | O(n²) | O(n log n) | >1000 words, parallel embeddings |
| **Medical Texts** | ~15% missed terms | ~10% missed | Abbreviation support |
| **Adaptive Chunking** | F1=0.65 | F1=0.72-0.75 | On benchmarks (10-15% improvement) |

**Measurement Plan:**

```python
# Before/after comparison
old_chunker = SlidingWindowSemanticSplitter(...)
new_chunker = EnhancedSlidingWindowSplitter(adaptive_window=True, use_parallel=True)

# Measure on same dataset
results_old = benchmark(old_chunker, dataset)
results_new = benchmark(new_chunker, dataset)

# Report improvements
speedup = results_old["time"] / results_new["time"]
f1_improvement = results_new["f1"] - results_old["f1"]
```

---

## 8. References

### Research Papers

1. **LangChain Semantic Chunking (2024)**
   - https://python.langchain.com/docs/modules/data_connection/document_transformers/
   - Embedding-based semantic boundaries

2. **Anthropic Contextual Retrieval (2024)**
   - https://www.anthropic.com/news/contextual-retrieval
   - 10-20% improvement with context prepending

3. **NVIDIA Chunking Benchmark (2024)**
   - Seven strategies across five datasets
   - Page-level: 0.648 accuracy

4. **SemEval 2013 NER Evaluation**
   - https://aclanthology.org/S13-2056/
   - Standard for span-based evaluation

5. **RAG-HPO Embedding Approach (PubMed: 39720417)**
   - 31% improvement over PhenoTagger (F1=0.64 → 0.70)

### Tools & Libraries

1. **nervaluate** - https://github.com/MantisAI/nervaluate (SemEval implementation reference)
2. **Sentence-Transformers** - https://www.sbert.net/ (built-in parallelization)
3. **spaCy** - https://explosion.ai/blog/multithreading-with-cython (NLP patterns)

### Datasets

1. **BiolarkGSC+** - 228 clinical notes with HPO terms
2. **ID-68** - 68 intellectual disability clinical notes
3. **LIRICAL** - 5,485 cases with HPO terms and diagnoses

---

## Appendix A: Design Decisions

### A.1 Why 3 Strategies Instead of 7?

**Problem:** 7 confusing strategy names with unclear differences.

**Solution:** 3 clear, purpose-driven strategies.

**Rationale:**
- ✅ **Clarity:** `fast`, `balanced`, `precise` - self-explanatory
- ✅ **User Experience:** Easy to choose based on priority (speed vs quality)
- ✅ **Maintenance:** Fewer strategies = less testing burden
- ✅ **Alpha Freedom:** No legacy constraints, clean start

**Mapping:**
- `simple` + `semantic` → `fast`
- `sliding_window` + variations → `balanced`
- `detailed` + enhancements → `precise`

### A.2 Why Alpha Simplification Matters

**Advantages of Alpha Status:**
1. ✅ **No breaking changes** - Nothing to break!
2. ✅ **Clean architecture** - Start with best practices
3. ✅ **No technical debt** - No legacy support burden
4. ✅ **Rapid iteration** - Change fast based on feedback
5. ✅ **Simple codebase** - Easier for contributors

**Trade-offs:**
- ❌ Less caution about changing APIs
- ❌ No migration guides needed
- ❌ Can refactor aggressively

**Decision:** Embrace alpha status, build it right from the start.

### A.3 Why Split Benchmark Runner?

**Problem:** 300-line God class.

**Solution:** 6 focused components with dependency injection.

**Rationale:**
- ✅ **SRP:** Each class has one responsibility
- ✅ **Testability:** Can test components in isolation
- ✅ **Extensibility:** Easy to swap implementations
- ✅ **DIP:** Depend on abstractions (protocols)

**Trade-off:** More classes, more boilerplate.
**Decision:** Maintainability > brevity. Quality code is readable code.

---

## Appendix B: Testing Strategy

### B.1 Unit Tests

**Coverage Target:** >80% for new code

**Key Test Cases:**

```python
# Enhanced sliding window splitter
def test_adaptive_window_sizing():
    """Adaptive window adjusts based on token length."""
    short_tokens = ["a", "b", "c", "d", "e"] * 20  # avg 1 char
    long_tokens = ["abcdefgh", "ijklmnop"] * 20    # avg 8 chars

    splitter = EnhancedSlidingWindowSplitter(
        model=mock_model,
        language="en",
        adaptive_window=True,
        min_window_size=3,
        max_window_size=10,
    )

    # Short tokens → larger window
    window_size_short = splitter._calculate_adaptive_window_size(short_tokens)
    assert window_size_short == 10

    # Long tokens → smaller window
    window_size_long = splitter._calculate_adaptive_window_size(long_tokens)
    assert window_size_long == 3
```

### B.2 Property-Based Tests

**Use Hypothesis to cover edge cases:**

```python
from hypothesis import given, strategies as st

@given(
    text=st.text(min_size=50, max_size=1000),
    window_size=st.integers(min_value=3, max_value=10),
)
def test_sliding_window_coverage(text, window_size):
    """Property: All input tokens should appear in some chunk."""
    splitter = EnhancedSlidingWindowSplitter(
        model=mock_model,
        language="en",
        window_size_tokens=window_size,
        adaptive_window=False,
    )

    chunks = splitter.chunk([text])

    # Reconstruct tokens from chunks
    input_tokens = set(text.split())
    chunk_tokens = set()
    for chunk in chunks:
        chunk_tokens.update(chunk.split())

    # All input tokens should appear in chunks
    assert input_tokens <= chunk_tokens
```

---

**END OF PLAN**
