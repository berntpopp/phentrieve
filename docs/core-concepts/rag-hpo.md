# RAG-HPO Approach

Phentrieve uses a Retrieval-Augmented Generation (RAG) approach specifically adapted for mapping clinical text to Human Phenotype Ontology (HPO) terms.

## What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that combines:

1. A retrieval system that finds relevant documents from a knowledge base
2. A generation system that uses both the query and retrieved documents to produce a response

In the context of Phentrieve, we adapt this approach to retrieve relevant HPO terms for a given clinical description.

## The Phentrieve RAG-HPO Architecture

### 1. Setup Phase (One-time)

- **HPO Data Acquisition**: Downloads the official HPO data in JSON format from JAX
- **HPO Term Extraction**: Processes the HPO data, extracting HPO terms with a focus on phenotypic abnormalities
- **Document Creation**: For each HPO term, creates a comprehensive document containing:
  - HPO ID (e.g., HP:0000123)
  - Primary label/name in English
  - Definition
  - Synonyms
- **Vector Embedding**: Converts each HPO term document into a vector representation using a multilingual embedding model
- **Index Creation**: Stores these vectors in a vector database (ChromaDB) for efficient retrieval

### 2. Query Phase (Runtime)

- **Query Embedding**: Converts the clinical text (in any supported language) into a vector representation using the same embedding model
- **Semantic Search**: Finds HPO term vectors that are most similar to the query vector
- **Re-ranking** (Optional): Uses a cross-encoder model to re-rank the results for improved precision
- **Result Processing**: Formats and returns the matched HPO terms with their similarity scores

## Advantages of this Approach

- **Direct semantic matching**: No error-prone intermediate translation step
- **Language-independent**: The model understands the meaning across languages
- **Scalable**: New HPO terms can be easily added to the knowledge base
- **Flexible**: Multiple embedding models can be used and compared
- **Extensible**: Additional processing steps (like assertion detection) can be integrated

## Technical Implementation

The RAG-HPO approach in Phentrieve is implemented using:

- **ChromaDB**: A vector database for storing and retrieving HPO term embeddings
- **HuggingFace Transformers**: For loading and using state-of-the-art multilingual embedding models
- **Sentence Transformers**: For efficient generation of text embeddings
- **FastAPI**: For exposing the functionality through a REST API
- **Vue.js**: For the web frontend interface

## Performance Considerations

The performance of the RAG-HPO approach depends on several factors:

- **Embedding Model**: Different models have different strengths for various languages and domains
- **Vector Database**: ChromaDB provides efficient similarity search capabilities
- **Hardware Acceleration**: GPU support significantly improves processing speed
- **Re-ranking**: Improves precision but adds computational overhead

!!! note "GPU Acceleration"
    According to our memory, Phentrieve supports GPU acceleration with CUDA when available and gracefully falls back to CPU when unavailable.
