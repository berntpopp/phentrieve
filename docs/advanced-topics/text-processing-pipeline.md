# Text Processing Pipeline

Phentrieve's text processing pipeline is a sophisticated system for extracting HPO terms from clinical text. This page explains its architecture and customization options in detail.

## Pipeline Architecture

The text processing pipeline consists of several components that work together:

1. **Text Chunkers**: Divide input text into manageable chunks
2. **Assertion Detectors**: Determine the status of phenotypic mentions
3. **HPO Term Retrievers**: Find relevant HPO terms for each chunk
4. **Result Processors**: Aggregate and filter the results

### Chunking Strategies

Based on our project memories, Phentrieve implements four different chunking strategies:

1. **Simple**: `paragraph → sentence`
   - Basic chunking that splits text into paragraphs and then sentences
   - Good for well-structured clinical notes with clear sentence boundaries

2. **Semantic**: `paragraph → sentence → semantic splitting`
   - More advanced chunking that further splits sentences based on semantic similarity
   - Uses the `SlidingWindowSemanticSplitter` to create semantically coherent chunks
   - Good for complex sentences with multiple phenotypic mentions

3. **Detailed**: `paragraph → sentence → fine-grained punctuation → semantic splitting`
   - The most fine-grained chunking strategy
   - First splits by punctuation, then applies semantic splitting
   - Best for dense clinical text with many phenotypic mentions

4. **Sliding Window**: `customizable semantic sliding window`
   - Most configurable strategy with parameters for window size, step size, etc.
   - Good for texts without clear sentence boundaries or when you want precise control

All strategies now use the `SlidingWindowSemanticSplitter` component, providing consistent behavior across strategies. Command-line parameters such as `--window-size`, `--step-size`, `--threshold`, and `--min-segment` override the configuration for all strategies, not just "sliding_window".

### Assertion Detection

The assertion detection system determines whether a phenotypic mention is affirmed, negated, uncertain, or described as normal. It uses both:

1. **Keyword-based Detection**: Identifies negation and uncertainty based on specific keywords and patterns
2. **Dependency-based Detection**: Uses syntactic dependency parsing to more accurately determine the scope of negation

The system implements a priority-based logic:
1. Dependency-based negation has highest priority
2. Context-specific keywords have second priority
3. General negation/uncertainty keywords have lowest priority

## Customization Options

### Chunking Parameters

You can customize the chunking behavior with these parameters:

- `--window-size`: Size of the sliding window (number of tokens)
- `--step-size`: Step size for sliding window progression
- `--threshold`: Semantic similarity threshold for chunk boundaries
- `--min-segment`: Minimum segment length to be considered a valid chunk

### Assertion Detection Configuration

Assertion detection can be customized through configuration files:

- Keyword lists for negation, uncertainty, and normalcy
- Language-specific patterns
- Priority rules for different detection methods

### Custom Processor Chains

For advanced use cases, you can create custom processor chains by combining different components:

```python
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.text_processing.chunkers import SemanticChunker
from phentrieve.text_processing.assertion import DependencyAssertionDetector

# Create a custom pipeline
pipeline = TextProcessingPipeline(
    chunker=SemanticChunker(
        window_size=128,
        step_size=64,
        threshold=0.25
    ),
    assertion_detector=DependencyAssertionDetector(),
    # Additional components...
)

# Process text with the custom pipeline
results = pipeline.process("Patient clinical text here...")
```

## Performance Considerations

The choice of chunking strategy affects both accuracy and performance:

- **Simple**: Fastest but may miss nuanced phenotypic mentions
- **Semantic**: Good balance between speed and accuracy
- **Detailed**: Most accurate but slowest
- **Sliding Window**: Performance depends on configuration parameters

!!! tip "GPU Acceleration"
    According to our memories, Phentrieve supports GPU acceleration with CUDA when available. This can significantly improve the performance of the text processing pipeline, especially when using large embedding models.
