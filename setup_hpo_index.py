import json
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from download_hpo import download_hpo_json, HPO_FILE_PATH
from extract_hpo_terms import extract_hpo_terms, HPO_TERMS_DIR
from tqdm import tqdm
import time

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
INDEX_DIR = "hpo_chroma_index"
COLLECTION_NAME = "hpo_multilingual"

def load_hpo_terms():
    """Load HPO terms from individual JSON files."""
    # Check if terms directory exists, if not extract them
    if not os.path.exists(HPO_TERMS_DIR) or not os.listdir(HPO_TERMS_DIR):
        print(f"HPO terms directory not found or empty. Extracting terms...")
        # Make sure we have the HPO data first
        download_hpo_json()
        # Extract individual term files
        if not extract_hpo_terms():
            return []

    # Load all HPO terms from individual JSON files
    print(f"Loading HPO terms from {HPO_TERMS_DIR}...")
    hpo_terms = []
    
    # Get all JSON files in the directory
    term_files = glob.glob(os.path.join(HPO_TERMS_DIR, "*.json"))
    
    for file_path in tqdm(term_files, desc="Loading HPO term files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                node = json.load(f)
                
            # Extract the HP ID
            node_id = node.get('id', '').replace('http://purl.obolibrary.org/obo/HP_', 'HP:').replace('_', ':')
            if not node_id.startswith('HP:'):
                continue
                
            # Extract the label
            label = node.get('lbl', '')
            
            # Extract definition
            definition = ""
            if 'meta' in node and 'definition' in node['meta'] and 'val' in node['meta']['definition']:
                definition = node['meta']['definition']['val']
            
            # Extract synonyms
            synonyms = []
            if 'meta' in node and 'synonyms' in node['meta']:
                for syn_obj in node['meta']['synonyms']:
                    if 'val' in syn_obj:
                        synonyms.append(syn_obj['val'])
            
            # Extract comments
            comments = []
            if 'meta' in node and 'comments' in node['meta']:
                comments = [c for c in node['meta']['comments'] if c]
            
            # Add to our collection
            hpo_terms.append({
                'id': node_id,
                'label': label,
                'definition': definition,
                'synonyms': synonyms,
                'comments': comments
            })
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Successfully loaded {len(hpo_terms)} HPO terms.")
    return hpo_terms

def create_hpo_documents(hpo_terms):
    """Creates descriptive documents for each HPO term."""
    print("Creating HPO documents for indexing...")
    documents = []
    metadatas = []
    ids = []
    
    for term in tqdm(hpo_terms, desc="Processing HPO terms"):
        # Skip entries with no label (should be rare)
        if not term['label']:
            continue
            
        # Create a comprehensive text document combining all relevant information
        # This creates a semantically-rich document that the model can embed
        doc_parts = []
        
        # Add primary information
        doc_parts.append(f"HPO ID: {term['id']}")
        doc_parts.append(f"Name: {term['label']}")
        
        # Add definition if available
        if term['definition']:
            doc_parts.append(f"Definition: {term['definition']}")
            
        # Add synonyms if available
        if term['synonyms']:
            doc_parts.append(f"Synonyms: {'; '.join(term['synonyms'])}")
            
        # Add comments if available - these often contain clinical context
        if term['comments']:
            doc_parts.append(f"Comments: {' '.join(term['comments'])}")
        
        # Join everything into a document
        doc_text = ". ".join(doc_parts)
        
        documents.append(doc_text)
        metadatas.append({
            'hpo_id': term['id'], 
            'hpo_name': term['label'],
            'has_definition': bool(term['definition']),
            'synonym_count': len(term['synonyms'])
        })
        ids.append(term['id'])  # Use HPO ID as the document ID
    
    return documents, metadatas, ids

def build_index(batch_size=100):
    """Loads HPO terms, generates embeddings, and builds the ChromaDB index."""
    start_time = time.time()
    
    # Load HPO terms from individual files
    hpo_terms = load_hpo_terms()
    if not hpo_terms:
        print("No HPO terms loaded. Exiting.")
        return
    
    # Create documents for indexing
    documents, metadatas, ids = create_hpo_documents(hpo_terms)
    
    # Load the sentence transformer model
    print(f"Loading the {MODEL_NAME} model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("Model loaded successfully!")
    
    # Initialize ChromaDB
    print(f"Initializing ChromaDB at {INDEX_DIR}...")
    try:
        client = chromadb.PersistentClient(path=INDEX_DIR)
        # Delete collection if it exists (for clean rebuilds)
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Removed existing collection: {COLLECTION_NAME}")
        except:
            pass  # Collection didn't exist
        
        # Create a new collection
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hpo_version": "latest", "model": MODEL_NAME}
        )
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return
    
    # Generate embeddings and add to ChromaDB in batches
    total_batches = (len(documents) + batch_size - 1) // batch_size
    print(f"Generating embeddings and adding to ChromaDB in {total_batches} batches...")
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Generate embeddings for the batch
        embeddings = model.encode(batch_docs)
        
        # Add to ChromaDB
        collection.add(
            documents=batch_docs,
            embeddings=embeddings.tolist(),
            metadatas=batch_meta,
            ids=batch_ids
        )
    
    end_time = time.time()
    print(f"Index built successfully in {end_time - start_time:.2f} seconds!")
    print(f"Indexed {len(documents)} HPO terms.")
    print(f"Index location: {os.path.abspath(INDEX_DIR)}")
    print("You can now use german_hpo_rag.py to query the index.")

if __name__ == "__main__":
    build_index()