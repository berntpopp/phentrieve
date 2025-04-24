import chromadb
import os
import sys
import re
import pysbd
import argparse
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import get_model_slug, get_index_dir, generate_collection_name
import torch

# Function to identify model dimension
def get_embedding_dimension(model_name):
    """Get the embedding dimension for a given model.
    Different models produce embeddings with different dimensions.
    """
    # Models with non-standard dimensions
    dimension_map = {
        "sentence-transformers/distiluse-base-multilingual-cased-v2": 512,
        # Add more models with different dimensions as needed
    }
    
    # Default dimension for most sentence transformer models
    return dimension_map.get(model_name, 768)

# Set up device - use CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Default values
DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to display results

def calculate_similarity(distance):
    """Convert distance to similarity score.
    
    Args:
        distance (float): Distance value from ChromaDB (cosine distance)
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # For standard cosine distance, range should be 0-2
    # However, our distances are larger - likely using L2 or another metric
    # So we'll normalize them to a more reasonable scale
    
    # For exceptionally large distances, apply a more aggressive normalization
    if distance > 2.0:
        # For large values, use an inversely scaled approach
        # This gives small but non-zero similarity even for large distances
        similarity = 1.0 / (1.0 + distance)
    else:
        # For normal cosine range values, use standard formula
        similarity = 1.0 - (distance / 2.0)
    
    # Ensure value is between 0 and 1
    return max(0.0, min(1.0, similarity))


def query_hpo(sentence, model, collection, n_results=10):
    """Generates embedding and queries the HPO index.
    
    Args:
        sentence (str): The German input sentence
        model: Loaded SentenceTransformer model instance
        collection: ChromaDB collection instance
        n_results (int): Number of results to fetch initially
        
    Returns:
        dict: ChromaDB query results dictionary
    """
    logging.info(f"Query: '{sentence}'")
    
    try:
        # Generate embedding for the query sentence
        query_embedding = model.encode([sentence])[0]  # Encode returns a list, get the first element
        
        # Query the collection - get more results than we need to filter by similarity
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 3,  # Get more results to allow better filtering
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    except Exception as e:
        logging.error(f"Error querying HPO: {e}")
        return None

def format_results(results, threshold=MIN_SIMILARITY_THRESHOLD, max_results=5):
    """Format the query results for display, filtering by similarity threshold.
    
    Args:
        results (dict): Raw results dictionary from collection.query
        threshold (float): Minimum similarity score to display
        max_results (int): Maximum number of results to display
        
    Returns:
        str: Formatted string for display
    """
    if not results or not results['ids'] or not results['ids'][0]:
        return "No matching HPO terms found."
    
    formatted_output = []
    count = 0
    
    # Print raw distances for debugging
    raw_distances = results['distances'][0][:5]  # Just look at first 5
    raw_similarities = [calculate_similarity(d) for d in raw_distances]
    formatted_output.append(f"DEBUG - Raw distances: {raw_distances}")
    formatted_output.append(f"DEBUG - Raw similarities: {raw_similarities}")
    formatted_output.append(f"DEBUG - Similarity threshold: {threshold}\n")
    
    # Prepare items with calculated similarity scores
    items = []
    for i, (hpo_id, metadata, distance, document) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0], 
        results['distances'][0],
        results['documents'][0]
    )):
        # Calculate similarity score from distance (cosine distance)
        similarity_score = calculate_similarity(distance)
        items.append((similarity_score, hpo_id, metadata, document))
    
    # Sort by similarity score (descending)
    items.sort(reverse=True)
    
    # Debug top 5 items regardless of threshold
    formatted_output.append("DEBUG - Top 5 items regardless of threshold:")
    for i, (similarity_score, hpo_id, metadata, document) in enumerate(items[:5]):
        formatted_output.append(
            f"DEBUG {i+1}. {metadata['hpo_id']} - {metadata['hpo_name']}\n"
            f"   Similarity: {similarity_score:.5f}"
        )
    formatted_output.append("")
    
    # Format the regular results
    count = 0
    regular_results = []
    for similarity_score, hpo_id, metadata, document in items:
        # Skip results with low similarity
        if similarity_score < threshold:
            continue
            
        # Limit to max_results
        if count >= max_results:
            break
            
        count += 1
        
        # Extract definition if available
        definition = "No definition"
        if 'Definition: ' in document:
            try:
                definition_part = document.split('Definition: ')[1]
                if '.' in definition_part:
                    definition = definition_part.split('.', 1)[0] + '.'
                else:
                    definition = definition_part
            except Exception as e:
                logging.debug(f"Error extracting definition: {e}")
                
        # Extract synonyms if available
        synonyms = ""
        if 'Synonyms: ' in document:
            try:
                synonyms_part = document.split('Synonyms: ')[1]
                if '.' in synonyms_part:
                    synonyms = synonyms_part.split('.', 1)[0]
                else:
                    synonyms = synonyms_part
                synonyms = f"\n   Synonyms: {synonyms}"
            except Exception as e:
                logging.debug(f"Error extracting synonyms: {e}")
        
        # Format the result with more detail
        regular_results.append(
            f"{count}. {metadata['hpo_id']} - {metadata['hpo_name']}\n"
            f"   Similarity: {similarity_score:.5f}\n"
            f"   Definition: {definition}{synonyms}"
        )
    
    if regular_results:
        formatted_output.append("REGULAR RESULTS (with threshold filtering):")
        formatted_output.extend(regular_results)
    else:
        formatted_output.append("No matching HPO terms found with sufficient similarity.\n\nTry lowering the similarity threshold with --similarity-threshold.")
        
    return "\n\n".join(formatted_output)

def segment_text(text, lang="de"):
    """Split text into sentences."""
    segmenter = pysbd.Segmenter(language=lang, clean=False)
    return segmenter.segment(text)


def process_input(text, model, collection, num_results=5, sentence_mode=False, similarity_threshold=MIN_SIMILARITY_THRESHOLD):
    """Process input text, either as a whole or sentence by sentence.
    
    Args:
        text (str): The input text to process
        model: Loaded SentenceTransformer model instance
        collection: ChromaDB collection instance
        num_results (int): Number of results to display per query
        sentence_mode (bool): Whether to process text sentence by sentence
        similarity_threshold (float): Minimum similarity threshold for results
    """
    if not sentence_mode:
        # Process whole text as one query
        results = query_hpo(text, model, collection, num_results)
        print("\nMatches:\n")
        print(format_results(results, threshold=similarity_threshold, max_results=num_results))
    else:
        # Process text sentence by sentence
        sentences = segment_text(text)
        if len(sentences) > 1:
            print(f"\nText segmented into {len(sentences)} sentences.")
        
        for i, sentence in enumerate(sentences, 1):
            if len(sentences) > 1:
                print(f"\n--- Sentence {i}/{len(sentences)} ---")
                print(f"\"{sentence}\"\n")
            
            results = query_hpo(sentence, model, collection, num_results)
            print(format_results(results, threshold=similarity_threshold, max_results=num_results))


def connect_to_chroma(index_dir, collection_name, model_name=None):
    """Connect to the ChromaDB index.
    
    Args:
        index_dir: Directory where ChromaDB indices are stored
        collection_name: Name of the collection to connect to
        model_name: Optional model name to handle dimension-specific collections
    """
    logging.info(f"Connecting to ChromaDB at {index_dir}")
    
    try:
        client = chromadb.PersistentClient(path=index_dir)
        
        # First try to find a model-specific collection
        try:
            logging.info(f"Trying model-specific collection: {collection_name}")
            collection = client.get_collection(name=collection_name)
            return collection
        except ValueError:
            # Fall back to default collection if model-specific one doesn't exist
            try:
                default_collection = 'hpo_multilingual'
                logging.info(f"Model-specific collection not found, trying default collection '{default_collection}'")
                collection = client.get_collection(name=default_collection)
                return collection
            except ValueError:
                logging.error(f"Error: Neither model-specific nor default collection found.")
                logging.info(f"You may need to run setup_hpo_index.py with the model {model_name} to create a collection compatible with its embedding dimension {get_embedding_dimension(model_name) if model_name else 'unknown'}.")
                return None
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {e}")
        return None


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Query HPO terms using German clinical descriptions."
    )
    parser.add_argument(
        "-t", "--text", 
        help="German text to query (if not provided, will use interactive mode)"
    )
    parser.add_argument(
        "-n", "--num-results", 
        type=int, 
        default=5, 
        help="Number of HPO terms to retrieve (default: 5)"
    )
    parser.add_argument(
        "-s", "--sentence-mode",
        action="store_true",
        help="Process input text sentence by sentence (helps with longer texts)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=MIN_SIMILARITY_THRESHOLD,
        help=f"Minimum similarity threshold (default: {MIN_SIMILARITY_THRESHOLD})"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model name (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()
    
    # Get index directory and collection name based on model
    index_dir = get_index_dir()
    collection_name = generate_collection_name(args.model_name)
    model_slug = get_model_slug(args.model_name)
    
    # Check if index exists
    if not os.path.exists(index_dir):
        logging.error(f"Error: Index directory '{index_dir}' not found. Please run setup_hpo_index.py first.")
        sys.exit(1)
    
    logging.info(f"Loading embedding model: {args.model_name}")
    try:
        # Special handling for Jina model which requires trust_remote_code=True
        jina_model_id = "jinaai/jina-embeddings-v2-base-de"
        if args.model_name == jina_model_id:
            logging.info(f"Loading Jina model '{args.model_name}' with trust_remote_code=True on {device}")
            # Security note: Only use trust_remote_code=True for trusted sources
            model = SentenceTransformer(args.model_name, trust_remote_code=True)
        else:
            logging.info(f"Loading model '{args.model_name}' on {device}")
            model = SentenceTransformer(args.model_name)
        
        # Move model to GPU if available
        model = model.to(device)
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer model: {e}")
        logging.error("Make sure you have run: pip install -r requirements.txt")
        sys.exit(1)
    logging.info("Model loaded successfully.")
    
    # Connect to ChromaDB
    collection = connect_to_chroma(index_dir, collection_name, args.model_name)
    if not collection:
        sys.exit(1)
    
    # Display summary
    print(f"Model: {args.model_name}")
    print(f"Collection: {collection_name}")
    print(f"Index entries: {collection.count()}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    
    # Process a one-time query if provided via command line
    if args.text:
        process_input(args.text, model, collection, args.num_results, args.sentence_mode, args.similarity_threshold)
        return
    
    # Interactive mode
    print("\n===== German HPO RAG Query Tool =====")
    print("Enter German clinical descriptions to find matching HPO terms.")
    print("Type 'exit', 'quit', or 'q' to exit the program.\n")
    
    while True:
        try:
            user_input = input("\nEnter German text (or 'q' to quit): ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting.")
                break
            
            if not user_input.strip():
                continue
            
            process_input(user_input, model, collection, args.num_results, args.sentence_mode, args.similarity_threshold)
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()