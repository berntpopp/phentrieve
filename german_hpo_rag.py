import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys
import pysbd
import argparse
import numpy as np

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
INDEX_DIR = "hpo_chroma_index"
COLLECTION_NAME = "hpo_multilingual"
MIN_SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score to display results

def query_hpo(sentence, model, collection, n_results=10):
    """Generates embedding and queries the HPO index."""
    # Generate embedding for the query sentence
    query_embedding = model.encode([sentence])[0]  # Encode returns a list, get the first element
    
    # Query the collection - get more results than we need to filter by similarity
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results * 2,  # Get more results than needed to allow filtering
        include=["documents", "metadatas", "distances"]
    )
    
    return results

def format_results(results, threshold=MIN_SIMILARITY_THRESHOLD, max_results=5):
    """Format the query results for display, filtering by similarity threshold."""
    if not results or not results['ids'] or not results['ids'][0]:
        return "No matching HPO terms found."
    
    formatted_output = []
    count = 0
    
    for i, (hpo_id, metadata, distance, document) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0], 
        results['distances'][0],
        results['documents'][0]
    )):
        # Use cosine similarity (higher is better)
        similarity_score = 1 / (1 + distance)
        
        # Skip results with low similarity
        if similarity_score < threshold:
            continue
            
        # Limit to max_results
        if count >= max_results:
            break
            
        count += 1
        
        # Format the result with more detail
        formatted_output.append(
            f"{count}. {metadata['hpo_id']} - {metadata['hpo_name']}\n"
            f"   Similarity: {similarity_score:.3f}\n"
            f"   Definition: {metadata.get('has_definition', False) and document.split('Definition: ')[1].split('.', 1)[0] + '.' if 'Definition: ' in document else 'No definition'}"
        )
    
    if not formatted_output:
        return "No matching HPO terms found with sufficient similarity."
        
    return "\n\n".join(formatted_output)

def segment_text(text, lang="de"):
    """Split text into sentences."""
    segmenter = pysbd.Segmenter(language=lang, clean=False)
    return segmenter.segment(text)

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
    args = parser.parse_args()
    
    # Check if index exists
    if not os.path.exists(INDEX_DIR):
        print("Error: HPO index not found. Please run setup_hpo_index.py first.")
        sys.exit(1)
    
    print("Loading embedding model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Make sure you have run: pip install -r requirements.txt")
        return
    print("Model loaded.")
    
    print(f"Connecting to ChromaDB at {INDEX_DIR}...")
    try:
        client = chromadb.PersistentClient(path=INDEX_DIR)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return
    print(f"Connected to ChromaDB index with {collection.count()} entries.")
    
    # Process a one-time query if provided via command line
    if args.text:
        process_input(args.text, model, collection, args.num_results, args.sentence_mode, args.similarity_threshold)
        return
    
    # Interactive mode
    print("\n===== German HPO RAG Query Tool =====")
    print("Enter German clinical descriptions to find matching HPO terms.")
    print("Type 'exit', 'quit', or 'q' to exit the program.\n")
    
    while True:
        user_input = input("\nEnter German text (or 'q' to quit): ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting.")
            break
        
        if not user_input.strip():
            continue
        
        process_input(user_input, model, collection, args.num_results, args.sentence_mode, args.similarity_threshold)

def process_input(text, model, collection, n_results=5, sentence_mode=False, similarity_threshold=MIN_SIMILARITY_THRESHOLD):
    """Process input text, either as whole or sentence by sentence."""
    if sentence_mode and len(text.strip()) > 100:  # Only split if text is substantial
        print("\nProcessing text sentence by sentence:")
        sentences = segment_text(text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            print(f"\n[Sentence {i+1}]: {sentence}")
            results = query_hpo(sentence, model, collection, n_results)
            print(format_results(results, threshold=similarity_threshold, max_results=n_results))
    else:
        # Process the whole text as a single query
        print("\nQuerying HPO terms...")
        results = query_hpo(text, model, collection, n_results)
        print(format_results(results, threshold=similarity_threshold, max_results=n_results))

if __name__ == "__main__":
    main()