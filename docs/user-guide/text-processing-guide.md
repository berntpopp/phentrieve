# Text Processing Guide

Phentrieve includes robust text processing capabilities for extracting HPO terms from clinical text. This guide explains how to use these features and customize the text processing pipeline.

## Text Processing Overview

The text processing pipeline in Phentrieve follows these steps:

1. **Text Chunking**: Divides the input text into manageable chunks
2. **Embedding Generation**: Converts each chunk into a vector representation
3. **HPO Term Retrieval**: Finds relevant HPO terms for each chunk
4. **Assertion Detection**: Determines the status of each term (affirmed, negated, etc.)
5. **Evidence Aggregation**: Combines evidence from multiple chunks for the same HPO term
6. **Result Filtering**: Filters results based on confidence thresholds

## Chunking Strategies

Phentrieve provides multiple text chunking strategies that can be combined in a pipeline:

### Simple Chunking
Divides text into paragraphs, then sentences.

```bash
phentrieve text process --strategy simple "Patient text here..."
```

### Semantic Chunking
More advanced chunking that divides text into semantic units:

1. Divides text into paragraphs
2. Splits paragraphs into sentences
3. Uses semantic similarity to further split sentences into meaningful chunks

```bash
phentrieve text process --strategy semantic "Patient text here..."
```

### Detailed Chunking
Even more fine-grained chunking:

1. Divides text into paragraphs
2. Splits paragraphs into sentences
3. Uses punctuation to create fine-grained segments
4. Applies semantic splitting to those segments

```bash
phentrieve text process --strategy detailed "Patient text here..."
```

### Sliding Window Chunking
Customizable semantic sliding window approach:

```bash
phentrieve text process --strategy sliding_window --window-size 128 --step-size 64 "Patient text here..."
```

!!! note "Command-line Parameters"
    The parameters `--window-size`, `--step-size`, `--threshold`, and `--min-segment` override the configuration for all strategies, not just "sliding_window".

## Assertion Detection

Phentrieve can detect the assertion status of each identified HPO term:

- **Affirmed**: The phenotype is positively mentioned (default)
- **Negated**: The phenotype is explicitly negated (e.g., "no microcephaly", "denies seizures")
- **Normal**: The finding is described as normal or within normal limits
- **Uncertain**: The phenotype is mentioned with uncertainty

Assertion detection uses both keyword-based and dependency-based approaches with a priority-based logic:

1. Dependency-based negation has highest priority
2. Context-specific keywords have second priority
3. General negation/uncertainty keywords have lowest priority

## Processing Clinical Text

### Basic Usage

```bash
# Process text directly
phentrieve text process "The patient exhibits microcephaly and frequent seizures."

# Process a text file
phentrieve text process --input-file clinical_notes.txt --output-file results.json
```

### Filtering Options

```bash
# Set minimum confidence threshold
phentrieve text process --min-confidence 0.4 "Patient text here..."

# Return only the highest-scoring HPO term for each chunk
phentrieve text process --top-term-per-chunk "Patient text here..."

# Specify language for better chunking and assertion detection
phentrieve text process --language de "Der Patient zeigt Mikrozephalie."
```

### Output Formats

```bash
# Output as JSON (default)
phentrieve text process --output-format json "Patient text here..."

# Output as CSV
phentrieve text process --output-format csv "Patient text here..."
```

## Example Output

```json
{
  "input_text": "The patient exhibits microcephaly and frequent seizures.",
  "processed_chunks": [
    {
      "text": "The patient exhibits microcephaly",
      "hpo_terms": [
        {
          "id": "HP:0000252",
          "name": "Microcephaly",
          "similarity": 0.85,
          "assertion": "affirmed"
        }
      ]
    },
    {
      "text": "frequent seizures",
      "hpo_terms": [
        {
          "id": "HP:0001250",
          "name": "Seizures",
          "similarity": 0.78,
          "assertion": "affirmed"
        }
      ]
    }
  ],
  "aggregated_results": [
    {
      "id": "HP:0000252",
      "name": "Microcephaly",
      "confidence": 0.85,
      "evidence_count": 1,
      "assertion": "affirmed"
    },
    {
      "id": "HP:0001250",
      "name": "Seizures",
      "confidence": 0.78,
      "evidence_count": 1,
      "assertion": "affirmed"
    }
  ]
}
```
