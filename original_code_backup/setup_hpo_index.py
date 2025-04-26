import json
import os
import glob
import chromadb
import argparse
import logging
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from download_hpo import download_hpo_json, HPO_FILE_PATH
from extract_hpo_terms import extract_hpo_terms, HPO_TERMS_DIR
from utils import (
    get_model_slug,
    get_index_dir,
    generate_collection_name,
    get_embedding_dimension,
)
import torch

# Set up device - use CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")


# Logging will be configured based on debug flag
def configure_logging(debug=False):
    """Configure logging based on debug flag"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    # Also set root logger level
    logging.getLogger().setLevel(level)

    if debug:
        logging.debug("Debug logging enabled in setup_hpo_index.py")


# Default to INFO level initially
configure_logging(False)

# Default model
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def load_hpo_terms():
    """Load HPO terms from individual JSON files."""
    # Check if terms directory exists, if not extract them
    if not os.path.exists(HPO_TERMS_DIR) or not os.listdir(HPO_TERMS_DIR):
        logging.info(f"HPO terms directory not found or empty. Extracting terms...")
        # Make sure we have the HPO data first
        download_hpo_json()
        # Extract individual term files
        if not extract_hpo_terms():
            return []

    # Load all HPO terms from individual JSON files
    logging.info(f"Loading HPO terms from {HPO_TERMS_DIR}...")
    hpo_terms = []

    # Get all JSON files in the directory
    term_files = glob.glob(os.path.join(HPO_TERMS_DIR, "*.json"))
    logging.debug(f"Found {len(term_files)} term files")

    # Add a progress bar for loading HPO terms
    for file_path in tqdm(term_files, desc="Loading HPO terms", unit="files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                node = json.load(f)

            # Extract the HP ID
            node_id = (
                node.get("id", "")
                .replace("http://purl.obolibrary.org/obo/HP_", "HP:")
                .replace("_", ":")
            )
            if not node_id.startswith("HP:"):
                continue

            # Extract the label
            label = node.get("lbl", "")

            # Extract definition
            definition = ""
            if (
                "meta" in node
                and "definition" in node["meta"]
                and "val" in node["meta"]["definition"]
            ):
                definition = node["meta"]["definition"]["val"]

            # Extract synonyms
            synonyms = []
            if "meta" in node and "synonyms" in node["meta"]:
                for syn_obj in node["meta"]["synonyms"]:
                    if "val" in syn_obj:
                        synonyms.append(syn_obj["val"])

            # Extract comments
            comments = []
            if "meta" in node and "comments" in node["meta"]:
                comments = [c for c in node["meta"]["comments"] if c]

            # Add to our collection
            hpo_terms.append(
                {
                    "id": node_id,
                    "label": label,
                    "definition": definition,
                    "synonyms": synonyms,
                    "comments": comments,
                }
            )

        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error reading {file_path}: {e}")

    logging.info(f"Successfully loaded {len(hpo_terms)} HPO terms.")
    return hpo_terms


def create_hpo_documents(hpo_terms):
    """Creates descriptive documents for each HPO term."""
    logging.info("Creating HPO documents for indexing...")
    documents = []
    metadatas = []
    ids = []

    for term in tqdm(hpo_terms, desc="Processing HPO terms"):
        # Skip entries with no label (should be rare)
        if not term["label"]:
            continue

        # Create a comprehensive text document combining all relevant information
        # This creates a semantically-rich document that the model can embed
        doc_parts = []

        # Add primary information
        doc_parts.append(f"HPO ID: {term['id']}")
        doc_parts.append(f"Name: {term['label']}")

        # Add definition if available
        if term["definition"]:
            doc_parts.append(f"Definition: {term['definition']}")

        # Add synonyms if available
        if term["synonyms"]:
            doc_parts.append(f"Synonyms: {'; '.join(term['synonyms'])}")

        # Add comments if available - these often contain clinical context
        if term["comments"]:
            doc_parts.append(f"Comments: {' '.join(term['comments'])}")

        # Join everything into a document
        doc_text = ". ".join(doc_parts)

        documents.append(doc_text)
        # Convert lists to serializable strings for ChromaDB
        metadatas.append(
            {
                "hpo_id": term["id"],
                "hpo_name": term["label"],
                "definition": term["definition"],
                "synonyms_count": len(term["synonyms"]),
                "synonyms_text": (
                    "; ".join(term["synonyms"]) if term["synonyms"] else ""
                ),
                "has_comments": len(term["comments"]) > 0,
            }
        )
        ids.append(term["id"])  # Use HPO ID as the document ID

    return documents, metadatas, ids


def build_index(model_name=DEFAULT_MODEL, batch_size=100, trust_remote_code=False):
    """Loads HPO terms, generates embeddings, and builds the ChromaDB index.

    Args:
        model_name (str): Name of the sentence-transformer model to use
        batch_size (int): Number of documents to process at once

    Returns:
        bool: True if indexing was successful, False otherwise
    """
    start_time = time.time()

    # Get index directory and collection name using utilities
    index_dir = get_index_dir()
    collection_name = generate_collection_name(model_name)
    model_slug = get_model_slug(model_name)

    logging.info(f"Building HPO index for model: {model_name} (slug: {model_slug})")
    logging.info(f"Index directory: {index_dir}")
    logging.info(f"Collection name: {collection_name}")

    # Load HPO terms from individual files
    hpo_terms = load_hpo_terms()
    if not hpo_terms:
        logging.error("No HPO terms loaded. Exiting.")
        return False

    # Create documents for indexing
    documents, metadatas, ids = create_hpo_documents(hpo_terms)

    # Load the sentence transformer model
    logging.info(f"Loading the {model_name} model...")
    try:
        # Special handling for models which require trust_remote_code=True
        models_requiring_trust = [
            "jinaai/jina-embeddings-v2-base-de",
            "Alibaba-NLP/gte-multilingual-base",
        ]

        if model_name in models_requiring_trust or trust_remote_code:
            logging.info(
                f"Loading model '{model_name}' with trust_remote_code=True on {device}"
            )
            # Security note: Only use trust_remote_code=True for trusted sources
            model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            logging.info(f"Loading model '{model_name}' on {device}")
            model = SentenceTransformer(model_name)

        # Move model to GPU if available
        model = model.to(device)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return False
    logging.info("Model loaded successfully!")

    # Initialize ChromaDB
    logging.info(f"Initializing ChromaDB at {index_dir}...")
    try:
        client = chromadb.PersistentClient(path=index_dir)

        # Delete collection if it exists (for clean rebuilds)
        try:
            client.delete_collection(collection_name)
            logging.info(f"Removed existing collection: {collection_name}")
        except Exception as e:
            # Collection didn't exist or some other error
            logging.debug(f"Note: {e}")

        # Check model embedding dimension
        model_dimension = get_embedding_dimension(model_name)
        logging.info(
            f"Using embedding dimension {model_dimension} for model {model_name}"
        )

        # Create a new collection with specified metadata
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hpo_version": "latest",
                "model": model_name,
                "dimension": model_dimension,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hnsw:space": "cosine",
            },
        )
        logging.info(
            f"Created new collection: {collection_name} with dimension {model_dimension}"
        )
    except Exception as e:
        logging.error(f"Error initializing ChromaDB: {e}")
        return False

    # Generate embeddings and add to ChromaDB in batches
    total_batches = (len(documents) + batch_size - 1) // batch_size
    logging.info(
        f"Computing embeddings for {len(documents)} HPO terms using {device}..."
    )

    # Process documents in batches
    total_batches = len(documents) // batch_size + (
        1 if len(documents) % batch_size > 0 else 0
    )
    logging.info(f"Processing {len(documents)} documents in {total_batches} batches...")

    # Add a clear progress bar for batch processing
    for i in tqdm(
        range(0, len(documents), batch_size),
        desc="Batches",
        total=total_batches,
        unit="batch",
    ):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        try:
            # Generate embeddings for the batch
            embeddings = model.encode(batch_docs, device=device)

            # Add to ChromaDB
            collection.add(
                documents=batch_docs,
                embeddings=embeddings.tolist(),
                metadatas=batch_meta,
                ids=batch_ids,
            )
        except Exception as e:
            logging.error(
                f"Error processing batch {i//batch_size + 1}/{total_batches}: {e}"
            )
            continue

    end_time = time.time()
    logging.info(f"Index built successfully in {end_time - start_time:.2f} seconds!")
    logging.info(f"Indexed {len(documents)} HPO terms.")
    logging.info(f"Index location: {os.path.abspath(index_dir)}")
    logging.info(
        f"You can now use german_hpo_rag.py with --model-name '{model_name}' to query the index."
    )
    return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB index for HPO terms using a multilingual sentence transformer model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to process at once (default: 100)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging with more verbose output",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for model loading (required for some models)",
    )
    args = parser.parse_args()

    # Configure logging based on debug flag
    configure_logging(args.debug)

    if args.trust_remote_code:
        logging.info("trust_remote_code enabled for model loading")

    # Build the index
    success = build_index(
        model_name=args.model_name,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
    )

    # Exit with appropriate status code
    if not success:
        logging.error("Index building failed.")
        exit(1)
