# Frequently Asked Questions

## General Questions

### What is Phentrieve?

Phentrieve is a comprehensive system for mapping clinical text in multiple languages to Human Phenotype Ontology (HPO) terms using a Retrieval-Augmented Generation (RAG) approach. It supports multilingual text processing, benchmarking across various embedding models, and provides flexible interfaces through a Python package, API, and web frontend.

### What languages does Phentrieve support?

Phentrieve supports multiple languages through its multilingual embedding models. The exact language coverage depends on the specific model being used. Domain-specific models like BioLORD support major languages for biomedical text, while general multilingual models like BGE-M3 support dozens of languages.

### Do I need to translate my clinical text?

No, that's the core innovation of Phentrieve! The system uses multilingual embedding models that can map clinical text in various languages directly to HPO terms without requiring translation.

## Technical Questions

### Does Phentrieve require a GPU?

No, Phentrieve can run on CPU-only environments. However, a GPU with CUDA support will significantly improve performance, especially when using larger models or processing large amounts of text. According to our project memories, Phentrieve supports GPU acceleration with CUDA when available and gracefully falls back to CPU when unavailable.

### How much disk space do the indexes require?

The disk space requirements depend on the number of models you're using. Each model's vector index typically requires 100-500MB of storage. If you build indexes for all supported models, expect to use 1-3GB of disk space.

### Can I use Phentrieve offline?

Yes, once you've downloaded the necessary HPO data and built the vector indexes, Phentrieve can operate completely offline. This makes it suitable for environments with restricted internet access, such as clinical settings with privacy requirements.

## Usage Questions

### How do I choose the best embedding model?

For most medical text processing use cases, the BioLORD model (`FremyCompany/BioLORD-2023-M`) is recommended as it consistently outperforms other models in HPO term mapping tasks. If you're working specifically with German text, the Jina model (`jinaai/jina-embeddings-v2-base-de`) may also be worth considering.

For the most accurate decision, run a benchmark on a representative sample of your data using the `phentrieve benchmark run` command with different models.

### What is the difference between chunking strategies?

Phentrieve offers different text chunking strategies for processing clinical text:

- **Simple**: Basic chunking that splits text into paragraphs and then sentences. Good for well-structured notes.
- **Semantic**: More advanced chunking that uses semantic similarity to create meaningful chunks. Good for complex sentences.
- **Detailed**: The most fine-grained chunking that splits by punctuation and then applies semantic splitting. Best for dense clinical text.
- **Sliding Window**: Most configurable strategy with parameters for window size, step size, etc. Good when you need precise control.

Choose based on your text structure and specific needs.

### How does assertion detection work?

Assertion detection determines whether a phenotypic mention is affirmed, negated, uncertain, or described as normal. It uses both keyword-based detection and dependency-based parsing to accurately determine the status of each HPO term mention in the text.

## Docker and Deployment

### How do I access the web interface after deployment?

After deploying with Docker, the web interface is available at:

- [http://localhost:8080](http://localhost:8080) when running locally
- At your configured domain if using a production deployment with proper DNS configuration

### How do I update Phentrieve to a new version?

For Docker deployments:

```bash
# Pull the latest images
docker-compose pull

# Restart the services
docker-compose down
docker-compose up -d
```

For pip installations:

```bash
pip install --upgrade phentrieve
```

### Where is the data stored in Docker deployments?

In Docker deployments, all data is stored in the directory specified by the `PHENTRIEVE_HOST_DATA_DIR` environment variable in your `.env.docker` file. This directory is mounted into the containers at `/app/data`.
