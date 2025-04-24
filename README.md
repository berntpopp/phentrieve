# German HPO RAG Prototype

This project implements a multilingual Retrieval-Augmented Generation (RAG) system for mapping German clinical text to Human Phenotype Ontology (HPO) terms without requiring translation.

## Core Idea

We use the `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` model to create a shared semantic space between German input text and English HPO terms. The system:

1. **Indexes HPO terms**: Creates embeddings for English HPO descriptions (labels, definitions, synonyms)
2. **Processes German queries**: Embeds German sentences using the same model
3. **Retrieves matches**: Finds the most semantically similar HPO terms for German input
4. **Returns results**: Provides HPO IDs and names without needing translation

## Advantages

- **Direct German processing**: No error-prone full-text translation
- **Semantic matching**: Finds relevant terms even without exact synonym matches
- **Offline usage**: All components (model, index, HPO data) are local after setup
- **Simpler architecture**: Avoids challenges of mapping coordinates from translated text

## Limitations

- **No coordinate information**: Identifies relevant HPO terms but not their precise positions in text
- **Embedding quality dependent**: Performance relies on how well the model handles clinical language
- **Requires HPO filtering**: Need to carefully select relevant HPO terms

## Setup and Usage

### Installation

```bash
pip install -r requirements.txt
```

### Prepare the HPO Index (Run Once)

```bash
python download_hpo.py  # Downloads hp.json if needed
python setup_hpo_index.py  # Parses HPO, creates index (takes time!)
```

Note: The first run will download the model (~1.1 GB) and generate embeddings, which can be time-intensive.

### Run the CLI Tool

```bash
python german_hpo_rag.py
```

Then enter German clinical descriptions when prompted.

## File Structure

- `download_hpo.py`: Downloads HPO data from official source
- `setup_hpo_index.py`: Parses HPO data and builds the ChromaDB index
- `german_hpo_rag.py`: CLI tool for querying with German text
- `requirements.txt`: Project dependencies