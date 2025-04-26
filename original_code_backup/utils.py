import re


def get_embedding_dimension(model_name):
    """Get the embedding dimension for a given model.
    Different models produce embeddings with different dimensions.
    """
    # Models with non-standard dimensions
    dimension_map = {
        "sentence-transformers/distiluse-base-multilingual-cased-v2": 512,
        "BAAI/bge-m3": 1024,  # BGE-M3 uses 1024-dimensional embeddings
        "sentence-transformers/LaBSE": 768,  # LaBSE uses 768-dimensional embeddings
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,  # MiniLM uses 384-dimensional embeddings
        # "Alibaba-NLP/gte-multilingual-base": 768,  # GTE uses standard 768-dimensional embeddings (covered by default)
    }

    # Default dimension for most sentence transformer models
    return dimension_map.get(model_name, 768)


def get_model_slug(model_name):
    """
    Create a simple, filesystem-safe string (slug) from the model name.

    Args:
        model_name (str): String representing the sentence-transformer model name
            (e.g., 'sentence-transformers/model-name')

    Returns:
        str: Filesystem-safe string
    """
    # Extract the last part of the model path if it contains slashes
    if "/" in model_name:
        slug = model_name.split("/")[-1]
    else:
        slug = model_name

    # Replace any non-alphanumeric characters with underscores
    slug = re.sub(r"[^a-zA-Z0-9]", "_", slug).lower()

    # Remove any duplicate underscores
    slug = re.sub(r"_+", "_", slug)

    # Remove leading/trailing underscores
    slug = slug.strip("_")

    return slug


def get_index_dir():
    """
    Return the path for the ChromaDB persistent storage.

    Returns:
        str: Directory path string
    """
    return "hpo_chroma_index"


def generate_collection_name(model_name):
    """
    Generate a unique collection name based on the model slug.

    Args:
        model_name (str): Model name string

    Returns:
        str: Collection name string
    """
    return f"hpo_multilingual_{get_model_slug(model_name)}"
