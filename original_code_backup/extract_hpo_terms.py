import json
import os
import logging
from tqdm import tqdm
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce log verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="extract_hpo_terms.log",
    filemode="w",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

HPO_FILE_PATH = os.path.join("data", "hp.json")
HPO_TERMS_DIR = os.path.join("data", "hpo_terms")

# Root ID for phenotypic abnormalities
PHENOTYPE_ROOT = "HP:0000118"  # Phenotypic abnormality
EXCLUDED_ROOTS = {
    "HP:0000005",  # Mode of inheritance
    "HP:0012823",  # Clinical modifier
    "HP:0031797",  # Clinical course
    "HP:0040279",  # Frequency
    "HP:0025354",  # Evidence
}


def normalize_id(id_str):
    """Normalize an HPO ID from various formats to standard HP:XXXXXXX format."""
    if not id_str:
        return ""

    # Handle URLs in format http://purl.obolibrary.org/obo/HP_0000118
    if id_str.startswith("http"):
        parts = id_str.split("/")
        if len(parts) > 0:
            id_str = parts[-1].replace("HP_", "HP:")

    # Replace underscore with colon if needed
    id_str = id_str.replace("_", ":")

    return id_str


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
        with open(HPO_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {HPO_FILE_PATH}.")
        return False
    except Exception as e:
        logging.error(f"Error reading {HPO_FILE_PATH}: {e}")
        return False

    # Extract nodes and their relationships
    nodes = {}
    edges = {}

    logging.info("Building HPO hierarchy structure...")

    # Process all graphs in the file
    for graph in data.get("graphs", []):
        # Process nodes first
        for node in graph.get("nodes", []):
            # Normalize the ID
            node_id = normalize_id(node.get("id", ""))

            # Only process HP terms
            if node_id.startswith("HP:"):
                nodes[node_id] = node

        # Now process the relationships (edges)
        for edge in graph.get("edges", []):
            # We're interested in 'is_a' relationships to build the hierarchy
            if (
                edge.get("pred") == "is_a"
                or edge.get("pred") == "http://purl.obolibrary.org/obo/BFO_0000050"
            ):
                subj = normalize_id(edge.get("sub", edge.get("subj", "")))
                obj = normalize_id(edge.get("obj", ""))

                # Only process edges between HP terms
                if subj.startswith("HP:") and obj.startswith("HP:"):
                    # Store the relationship: obj is parent of subj
                    if obj not in edges:
                        edges[obj] = []
                    if subj not in edges[obj]:
                        edges[obj].append(subj)  # Parent -> Children mapping

    # Log some stats
    logging.info(
        f"Found {len(nodes)} nodes and {sum(len(children) for children in edges.values())} edges"
    )

    # Verify the root node exists and has children
    if PHENOTYPE_ROOT not in nodes:
        logging.error(f"Root node {PHENOTYPE_ROOT} not found in the data!")
        return False

    if PHENOTYPE_ROOT not in edges or not edges[PHENOTYPE_ROOT]:
        logging.error(f"Root node {PHENOTYPE_ROOT} has no children!")
        return False

    logging.info(
        f"Root node {PHENOTYPE_ROOT} has {len(edges.get(PHENOTYPE_ROOT, []))} direct children"
    )

    # Collect all phenotype terms starting from the root
    phenotype_terms_ids = set()
    phenotype_terms_ids.add(PHENOTYPE_ROOT)  # Add the root itself

    # Breadth-first search to find all descendants of the phenotype root
    logging.info("Identifying all phenotype terms...")
    to_visit = [PHENOTYPE_ROOT]
    visited = set(to_visit)

    while to_visit:
        current_id = to_visit.pop(0)

        # Get all children of the current node
        children = edges.get(current_id, [])

        for child_id in children:
            if child_id not in visited and child_id not in EXCLUDED_ROOTS:
                phenotype_terms_ids.add(child_id)
                to_visit.append(child_id)
                visited.add(child_id)

    # Remove excluded roots and their descendants
    for excluded_root in EXCLUDED_ROOTS:
        if excluded_root in phenotype_terms_ids:
            phenotype_terms_ids.remove(excluded_root)

        # Find and remove descendants of excluded roots
        if excluded_root in edges:
            excluded_descendants = set()
            to_visit = [excluded_root]
            visited_exclusions = set([excluded_root])

            while to_visit:
                current_id = to_visit.pop(0)
                excluded_descendants.add(current_id)

                for child_id in edges.get(current_id, []):
                    if child_id not in visited_exclusions:
                        to_visit.append(child_id)
                        visited_exclusions.add(child_id)

            # Remove the excluded descendants from our phenotype terms
            phenotype_terms_ids -= excluded_descendants

    logging.info(
        f"Found {len(phenotype_terms_ids)} phenotypic abnormality terms in HPO"
    )

    # Save each phenotype term as a separate JSON file
    logging.info("Saving individual HPO term files...")
    term_count = 0

    with tqdm(total=len(phenotype_terms_ids), desc="Extracting HPO terms") as pbar:
        for term_id in phenotype_terms_ids:
            if term_id in nodes:
                node = nodes[term_id]
                # Create filename from term ID
                clean_id = term_id.replace(":", "_")

                # Create a file for each term
                output_path = os.path.join(HPO_TERMS_DIR, f"{clean_id}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(node, f, indent=2)
                term_count += 1
                pbar.update(1)

    logging.info(f"Successfully extracted {term_count} HPO terms to {HPO_TERMS_DIR}")
    return True


if __name__ == "__main__":
    try:
        logging.info("Starting HPO term extraction")
        result = extract_hpo_terms()
        if result:
            logging.info("Finished HPO term extraction successfully")
        else:
            logging.error("HPO term extraction failed")
    except Exception as e:
        logging.exception("Error during HPO term extraction")
