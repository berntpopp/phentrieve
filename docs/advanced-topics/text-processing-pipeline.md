# Text Processing Pipeline

Phentrieve utilizes a sophisticated, configurable pipeline to transform raw clinical text into analyzable semantic units. This document explains the architecture, implementation details, and customization options.

## Architecture Overview

The pipeline operates in sequential stages defined in your configuration. The default strategy (`sliding_window_punct_conj_cleaned`) follows this flow:

1. **Normalization**: Line endings and whitespace are standardized
2. **Paragraph/Sentence Splitting**: Text is broken down structurally
3. **Fine-Grained Splitting**: Punctuation and Conjunction splitting
4. **Semantic Windowing**: AI-driven semantic boundary detection
5. **Cleaning**: Removal of non-semantic artifacts

## Chunking Strategies

Phentrieve implements multiple chunking strategies, each building on the previous one to create increasingly granular semantic units.

### The Sliding Window Semantic Splitter

**Class:** `SlidingWindowSemanticSplitter`

This is the core innovation in Phentrieve's processing. Instead of arbitrary splits, it uses vector embeddings to detect semantic shifts in the text.

#### How It Works

1. **Tokenization**: The text segment is tokenized using simple whitespace splitting
2. **Windowing**: A sliding window (configurable size, default=7 tokens, step=1 token) moves across the text
3. **Embedding**: Each window is embedded using a fast SBERT model
4. **Coherence Check**: Cosine similarity is calculated between adjacent windows
5. **Splitting**: If similarity drops below the `splitting_threshold` (default 0.5), a split point is marked
6. **Negation Merging**: A heuristic pass re-merges splits that accidentally separated a negation term (e.g., "no") from its subject using language-specific resources

#### Configuration Parameters

```yaml
chunking_pipeline:
  - type: sliding_window
    config:
      window_size_tokens: 7        # Size of each embedding window
      step_size_tokens: 1          # Step size (1 = maximum overlap)
      splitting_threshold: 0.5     # Cosine similarity threshold
      min_split_segment_length_words: 3  # Minimum words per segment
```

**Performance Characteristics:**
- **Accuracy**: Highest semantic precision
- **Speed**: Slower due to embedding computation (~100-500ms per segment with GPU)
- **Memory**: Moderate (model must be loaded)

**When to Use:**
- Complex sentences with multiple phenotypic mentions
- Text without clear sentence boundaries
- When semantic precision is paramount

### Structural Chunkers

#### FineGrainedPunctuationChunker

**Purpose:** Splits on commas, semicolons, and colons while preserving special cases.

**Intelligent Handling:**
- **Decimal Numbers**: Preserves "1.5", "98.6", etc.
- **Abbreviations**: Preserves "Dr.", "vs.", "Ph.D.", "ie.", "eg.", etc.
- **Initials**: Preserves "A.B." style initials

**Implementation:**
```python
# Splits on: . , : ; ? !
# But preserves:
abbreviations = [r"\bDr\.", r"\bMs\.", r"\bMr\.", r"\bMrs\.", r"\bPh\.D\.",
                 r"\bed\.", r"\bp\.", r"\bie\.", r"\beg\.", r"\bcf\.",
                 r"\bvs\.", r"\bSt\.", r"\bJr\.", r"\bSr\.", r"[A-Z]\.[A-Z]\."]
```

**Example:**
```
Input:  "Patient has arachnodactyly, i.e., long fingers; heart rate is 98.6 bpm."
Output: ["Patient has arachnodactyly", "i.e., long fingers", "heart rate is 98.6 bpm"]
```

#### ConjunctionChunker

**Purpose:** Splits before coordinating conjunctions while keeping the conjunction with the following chunk.

**Language Support:**
- **English**: "and", "but", "or", "nor", "for", "yet", "so"
- **German**: "und", "aber", "oder", "denn", "sondern"
- **Other languages**: Loaded from `coordinating_conjunctions.json`

**Splitting Logic:**
```python
# Pattern: Split before " conjunction " (case-insensitive, word boundaries)
split_pattern = r"\s+(?=(\b(?:and|but|or|...)\b\s+))"
```

**Example:**
```
Input:  "Patient has seizures but no developmental delay"
Output: ["Patient has seizures", "but no developmental delay"]
```

### Final Chunk Cleaner

**Class:** `FinalChunkCleaner`

Post-processes chunks to remove "low semantic value" content.

#### Cleaning Operations

1. **Leading Stopword Removal**: Strips "the", "a", "an", "with", etc. from the beginning
2. **Trailing Stopword Removal**: Strips conjunctions and articles from the end
3. **Low-Value Chunk Filtering**: Removes chunks consisting entirely of stop words
4. **Length Filtering**: Removes chunks shorter than `min_cleaned_chunk_length_chars` (default: 1)

#### Multi-Pass Cleaning

The cleaner performs up to `max_cleanup_passes` (default: 3) to handle nested cases:

```
Pass 1: "the and the patient" → "and the patient"
Pass 2: "and the patient" → "the patient"
Pass 3: "the patient" → "patient" (final)
```

#### Language Resources

Stopwords are loaded from JSON resources per language:
- `leading_cleanup_words.json`: Words to remove from start
- `trailing_cleanup_words.json`: Words to remove from end
- `low_value_words.json`: Words indicating low semantic content

**Example:**
```
Input:  ["the patient", "with seizures", "and", "a small head"]
Output: ["patient", "seizures", "small head"]
```

## Assertion Detection

The `CombinedAssertionDetector` determines if a phenotype is:

- **Affirmed**: "Patient has..."
- **Negated**: "No sign of..."
- **Normal**: "Heart sounds are normal"
- **Uncertain**: "Possible evidence of..."

### Detection Architecture

#### Priority Logic (Highest to Lowest)

1. **Dependency Parsing (spaCy)**
   - Uses grammatical dependency trees
   - Finds exact scope of negation terms
   - Highest accuracy but slower

2. **Keyword Patterns**
   - Fallback to window-based keyword matching
   - Used if dependency parsing is inconclusive
   - Used for languages without spaCy models

#### Dependency-Based Detection

**How It Works:**
1. Parse sentence into dependency tree using spaCy
2. Find negation words (e.g., "no", "without", "denies")
3. Traverse dependency tree to find negation scope
4. Check if chunk text falls within negation scope

**Advantages:**
- Grammatically accurate
- Handles complex sentence structures
- Language-specific patterns

**Example:**
```
Text: "Patient has no history of seizures but does have tremors"
Tree: "Patient" ← SUBJ ← "has" → OBJ → "history" → PREP → "of" → POBJ → "seizures"
                                                       ↑ NEG ← "no"
Result:
  - "seizures" → NEGATED (in scope of "no")
  - "tremors" → AFFIRMED (outside scope)
```

#### Keyword-Based Detection (Fallback)

**Pattern Matching:**
- Negation window: Looks for negation keywords within N words before the chunk
- Uncertainty markers: "possible", "maybe", "uncertain", "suspected"
- Normal markers: "normal", "unremarkable", "within normal limits"

**Language Resources:**
- `negation_keywords.json`: Language-specific negation terms
- `uncertainty_keywords.json`: Uncertainty markers
- `normal_keywords.json`: Normalcy indicators

## Complete Pipeline Configuration

### Default Strategy: `sliding_window_punct_conj_cleaned`

```yaml
chunking_pipeline:
  # 1. Normalize whitespace and line endings
  - type: paragraph        # Split on double newlines

  # 2. Sentence boundaries
  - type: sentence         # Split on sentence boundaries (spaCy-based)

  # 3. Fine-grained structural splitting
  - type: fine_grained_punctuation  # Split on commas, semicolons, etc.
  - type: conjunction                # Split before coordinating conjunctions

  # 4. Semantic splitting
  - type: sliding_window
    config:
      window_size_tokens: 7
      step_size_tokens: 1
      splitting_threshold: 0.5
      min_split_segment_length_words: 3

  # 5. Cleanup non-semantic elements
  - type: final_chunk_cleaner
    config:
      min_cleaned_chunk_length_chars: 1
      filter_short_low_value_chunks_max_words: 2
      max_cleanup_passes: 3
```

### Alternative Strategies

#### `simple` - Basic Structural Splitting
```yaml
chunking_pipeline:
  - type: paragraph
  - type: sentence
```
**Use Case:** Well-structured clinical notes with clear sentence boundaries

#### `sliding_window` - Pure Semantic
```yaml
chunking_pipeline:
  - type: paragraph
  - type: sliding_window
    config:
      window_size_tokens: 10  # Larger windows for less aggressive splitting
      splitting_threshold: 0.4  # Lower threshold = more splits
```
**Use Case:** Text without punctuation, voice transcriptions

## CLI Override Parameters

You can override pipeline configuration via CLI:

```bash
# Override sliding window parameters
phentrieve text process "..." \
  --strategy sliding_window_punct_conj_cleaned \
  --window-size 10 \
  --step-size 2 \
  --threshold 0.4 \
  --min-segment 5

# Use different strategy
phentrieve text process "..." --strategy simple
```

**Important:** CLI overrides apply to ALL stages in the pipeline that use those parameters, not just the named strategy.

## Performance Considerations

### Speed Comparison (per 1000-word document)

| Strategy | CPU Time | GPU Time | Memory |
|----------|----------|----------|--------|
| `simple` | ~50ms | ~50ms | Low |
| `fine_grained_punctuation` | ~100ms | ~100ms | Low |
| `conjunction` | ~150ms | ~150ms | Low |
| `sliding_window` | ~2000ms | ~300ms | High |
| `sliding_window_punct_conj_cleaned` | ~2500ms | ~400ms | High |

### Optimization Tips

1. **Use GPU**: Provides 5-10x speedup for semantic splitting
2. **Increase Window Step**: Larger `step_size_tokens` = faster but less precise
3. **Increase Threshold**: Higher `splitting_threshold` = fewer splits = faster
4. **Batch Processing**: Process multiple documents in parallel

## Custom Pipeline Creation

For advanced use cases, create custom pipelines programmatically:

```python
from phentrieve.text_processing.chunkers import (
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowSemanticSplitter,
    FinalChunkCleaner
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.embeddings import get_model

# Load embedding model for semantic splitting
model = get_model("FremyCompany/BioLORD-2023-M")

# Create custom chunker chain
chunkers = [
    ParagraphChunker(language="en"),
    SentenceChunker(language="en"),
    SlidingWindowSemanticSplitter(
        language="en",
        model=model,
        window_size_tokens=10,
        splitting_threshold=0.3  # More aggressive splitting
    ),
    FinalChunkCleaner(
        language="en",
        min_cleaned_chunk_length_chars=5
    )
]

# Create pipeline
pipeline = TextProcessingPipeline(
    chunkers=chunkers,
    language="en"
)

# Process text
chunks = pipeline.chunk_text("Your clinical text here...")
```

## Language Support

The pipeline adapts to different languages via:

1. **spaCy Models**: Language-specific sentence splitting and dependency parsing
2. **Resource Files**: Language-specific stopwords, conjunctions, negation keywords
3. **Model Selection**: Use language-specific or multilingual embedding models

**Supported Languages:**
- English (en)
- German (de)
- Spanish (es)
- French (fr)
- Dutch (nl)

To add a new language:
1. Install spaCy model: `python -m spacy download {lang}_core_web_sm`
2. Add language resources to `phentrieve/text_processing/resources/`
3. Configure in `phentrieve.yaml`

!!! tip "GPU Acceleration"
    The semantic sliding window chunker benefits significantly from GPU acceleration. On a modern GPU, processing time can be reduced from ~2.5s to ~0.4s per 1000-word document.

!!! warning "Model Loading"
    The `SlidingWindowSemanticSplitter` requires a SentenceTransformer model to be loaded into memory (~400MB). For processing large batches, consider reusing the same model instance across documents.
