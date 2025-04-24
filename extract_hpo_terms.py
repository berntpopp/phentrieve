import json
import os
import logging
from tqdm import tqdm
import shutil

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='extract_hpo_terms.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

HPO_FILE_PATH = os.path.join("data", "hp.json")
HPO_TERMS_DIR = os.path.join("data", "hpo_terms")

# Root ID for phenotypic abnormalities
PHENOTYPE_ROOT = "HP:0000118"  # Phenotypic abnormality
EXCLUDED_ROOTS = {
    "HP:0000005",  # Mode of inheritance
    "HP:0012823",  # Clinical modifier
    "HP:0031797",  # Clinical course
    "HP:0040279",  # Frequency
    "HP:0025354"   # Evidence
}

def extract_hpo_terms():
    """Extract individual HPO terms from the main JSON file into separate files."""
    
    # Check if the source file exists
    if not os.path.exists(HPO_FILE_PATH):
        logging.error(f"Error: {HPO_FILE_PATH} not found.")
        return False
        
    # Create/recreate the output directory
    if os.path.exists(HPO_TERMS_DIR):
        logging.info(f"Removing existing directory: {HPO_TERMS_DIR}")
        shutil.rmtree(HPO_TERMS_DIR)
    
    os.makedirs(HPO_TERMS_DIR, exist_ok=True)
    
    # Read the main HPO JSON file
    logging.info(f"Parsing {HPO_FILE_PATH}...")
    try:
        with open(HPO_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {HPO_FILE_PATH}.")
        return False
    
    # Extract nodes and their relationships
    nodes = {}
    edges = {}
    
    logging.info("Building HPO hierarchy structure...")
    for graph in data.get('graphs', []):
        # Collect all nodes
        for node in graph.get('nodes', []):
            node_id = node.get('id', '').replace('http://purl.obolibrary.org/obo/HP_', 'HP:').replace('_', ':')
            if node_id.startswith('HP:'):
                # Store the node for later processing
                nodes[node_id] = node
        
        # Collect all edges for hierarchy tracking
        for edge in graph.get('edges', []):
            if edge.get('pred') == 'is_a':
                # Fix URL to ID conversion for both subject and object
                subj = edge.get('subj', '').replace('http://purl.obolibrary.org/obo/HP_', 'HP:')
                obj = edge.get('obj', '').replace('http://purl.obolibrary.org/obo/HP_', 'HP:')
                
                # Replace underscore with colon for proper HP IDs
                subj = subj.replace('_', ':')
                obj = obj.replace('_', ':')
                
                logging.debug(f"Found is_a relationship: {subj} is_a {obj}")
                
                if obj not in edges:
                    edges[obj] = []
                edges[obj].append(subj)  # Parent -> Child direction for traversal
    
    # Log some stats about the data
    logging.info(f"Found {len(nodes)} nodes and {sum(len(children) for children in edges.values())} edges")
    
    # Check if the root node has any children in our edge map
    if PHENOTYPE_ROOT in edges:
        logging.info(f"Root node {PHENOTYPE_ROOT} has {len(edges[PHENOTYPE_ROOT])} direct children")
        for i, child in enumerate(edges[PHENOTYPE_ROOT][:10]):  # Print first 10 children
            logging.debug(f"Child {i+1}: {child} - {nodes[child]['lbl'] if child in nodes else 'Unknown'}")
    else:
        logging.error(f"Root node {PHENOTYPE_ROOT} has no children in the edge map!")
    
    # Build a set of all phenotype terms using a breadth-first approach
    phenotype_terms_ids = set()
    
    # Add the root node
    if PHENOTYPE_ROOT in nodes:
        phenotype_terms_ids.add(PHENOTYPE_ROOT)
        logging.info(f"Added root node {PHENOTYPE_ROOT} to phenotype terms")
    else:
        logging.error(f"Error: Phenotypic abnormality root node '{PHENOTYPE_ROOT}' not found in the HPO file!")
        return False
    
    # Breadth-first search to find all descendants
    logging.info("Identifying phenotype terms...")
    to_visit = [PHENOTYPE_ROOT]
    visited = set(to_visit)
    
    while to_visit:
        current_id = to_visit.pop(0)
        phenotype_terms_ids.add(current_id)
        
        # Find children of current node (all nodes that have current node as a parent)
        children = edges.get(current_id, [])
        logging.debug(f"Node {current_id} has {len(children)} children")
        
        for child_id in children:
            if child_id not in visited:
                # Don't visit excluded roots
                if child_id not in EXCLUDED_ROOTS:
                    to_visit.append(child_id)
                    visited.add(child_id)
                    logging.debug(f"Added {child_id} to visit queue")
                else:
                    logging.debug(f"Skipping excluded root: {child_id}")
            else:
                logging.debug(f"Already visited: {child_id}")
    
    # Filter excluded roots and their descendants
    for excluded_root in EXCLUDED_ROOTS:
        if excluded_root in phenotype_terms_ids:
            phenotype_terms_ids.remove(excluded_root)
            logging.debug(f"Removed excluded root: {excluded_root}")
            
        # Also remove any descendants of excluded roots
        to_remove = set()
        if excluded_root in edges:
            to_visit = [excluded_root]
            excluded_visited = set(to_visit)
            
            while to_visit:
                current_id = to_visit.pop(0)
                to_remove.add(current_id)
                
                # Find children
                for child_id in edges.get(current_id, []):
                    if child_id not in excluded_visited:
                        to_visit.append(child_id)
                        excluded_visited.add(child_id)
                        logging.debug(f"Added {child_id} to exclusion list (descendant of {excluded_root})")
            
            # Remove excluded descendants
            before = len(phenotype_terms_ids)
            phenotype_terms_ids -= to_remove
            after = len(phenotype_terms_ids)
            logging.debug(f"Removed {before - after} descendants of {excluded_root}")
    
    logging.info(f"Found {len(phenotype_terms_ids)} phenotypic abnormality terms in HPO.")
    
    # Save some samples for debugging
    samples = list(phenotype_terms_ids)[:10]
    logging.debug(f"Sample phenotype terms: {samples}")
    
    # Save each phenotype term as a separate JSON file
    logging.info("Saving individual HPO term files...")
    term_count = 0
    for term_id in tqdm(phenotype_terms_ids, desc="Extracting HPO terms"):
        if term_id in nodes:
            node = nodes[term_id]
            # Create a clean term ID without the URL part for the filename
            clean_id = term_id.replace(':', '_')
            
            # Create a file for each term
            output_path = os.path.join(HPO_TERMS_DIR, f"{clean_id}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(node, f, indent=2)
            term_count += 1
            
            if term_count % 1000 == 0:
                logging.debug(f"Processed {term_count} terms so far")
    
    logging.info(f"Successfully extracted {term_count} HPO terms to {HPO_TERMS_DIR}")
    return True

if __name__ == "__main__":
    try:
        logging.info("Starting HPO term extraction")
        extract_hpo_terms()
        logging.info("Finished HPO term extraction")
    except Exception as e:
        logging.exception("Error during HPO term extraction")