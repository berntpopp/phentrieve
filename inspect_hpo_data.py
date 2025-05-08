import pickle
import os
import sys
from pathlib import Path
from collections import Counter
import argparse

# --- Attempt to import config paths dynamically ---
# This assumes the script might be run from the repo root OR
# that the 'phentrieve' package is installed/findable.
try:
    # If run using 'python -m phentrieve.scripts.inspect_hpo_data'
    # or if installed and run from anywhere
    from phentrieve.utils import resolve_data_path, get_default_data_dir
    from phentrieve.config import (
        DEFAULT_ANCESTORS_FILENAME,
        DEFAULT_DEPTHS_FILENAME,
        PHENOTYPE_ROOT,
    )

    DATA_DIR_RESOLVED = resolve_data_path(None, "data_dir", get_default_data_dir)
    DEFAULT_ANCESTORS_PATH = DATA_DIR_RESOLVED / DEFAULT_ANCESTORS_FILENAME
    DEFAULT_DEPTHS_PATH = DATA_DIR_RESOLVED / DEFAULT_DEPTHS_FILENAME
    ROOT_NODE = PHENOTYPE_ROOT
except ImportError:
    # Fallback if run directly from scripts dir or if package structure is different
    print(
        "Warning: Could not import from phentrieve package."
        " Using relative paths assuming script is run from repo root.",
        file=sys.stderr,
    )
    DEFAULT_ANCESTORS_PATH = Path("./data/hpo_ancestors.pkl")
    DEFAULT_DEPTHS_PATH = Path("./data/hpo_term_depths.pkl")
    ROOT_NODE = "HP:0000118"  # Default if config import fails
# --- End Config Import ---


def load_pickle_file(file_path: Path):
    """Safely loads a pickle file."""
    if not file_path.exists():
        print(f"Error: File not found - {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Successfully loaded {file_path}")
        return data
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle {file_path}. File might be corrupt.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred loading {file_path}: {e}")
        return None


def inspect_ancestors(data, num_samples=5):
    """Inspects the loaded ancestors dictionary."""
    if not isinstance(data, dict):
        print("Error: Ancestor data is not a dictionary.")
        return

    print(f"\n--- Ancestor Data Inspection ---")
    print(f"Total terms with ancestor sets: {len(data)}")
    if not data:
        return

    sample_keys = list(data.keys())[:num_samples]
    print(f"\nShowing samples for first {len(sample_keys)} terms:")

    missing_self = 0
    missing_phenotype_root = 0

    for term_id in sample_keys:
        ancestors = data.get(term_id)
        if isinstance(ancestors, set):
            print(f"\nTerm: {term_id}")
            print(f"  Number of ancestors (incl. self): {len(ancestors)}")
            is_self_present = term_id in ancestors
            is_root_present = ROOT_NODE in ancestors
            print(f"  Includes self ({term_id})?: {is_self_present}")
            print(f"  Includes Phenotype Root ({ROOT_NODE})?: {is_root_present}")
            print(f"  Sample ancestors: {list(ancestors)[:10]}")
            if not is_self_present:
                missing_self += 1
            if not is_root_present and term_id != ROOT_NODE:
                missing_phenotype_root += 1
        else:
            print(
                f"  Error: Value for {term_id} is not a set (Type: {type(ancestors)})"
            )

    print(f"\nSummary across {len(sample_keys)} samples:")
    print(f"  Terms missing self-reference: {missing_self}")
    print(f"  Terms missing Phenotype Root ({ROOT_NODE}): {missing_phenotype_root}")


def inspect_depths(data, num_samples=5):
    """Inspects the loaded depths dictionary."""
    if not isinstance(data, dict):
        print("Error: Depth data is not a dictionary.")
        return

    print(f"\n--- Depth Data Inspection ---")
    print(f"Total terms with depth values: {len(data)}")
    if not data:
        return

    depth_values = [d for d in data.values() if isinstance(d, int)]
    if not depth_values:
        print("No valid integer depth values found.")
        return

    min_depth = min(depth_values)
    max_depth = max(depth_values)
    avg_depth = sum(depth_values) / len(depth_values)
    depth_counts = Counter(depth_values)
    negative_depth_count = sum(1 for d in depth_values if d < 0)

    print(f"\nDepth Statistics:")
    print(f"  Min Depth: {min_depth}")
    print(f"  Max Depth: {max_depth}")
    print(f"  Average Depth: {avg_depth:.2f}")
    print(f"  Count of terms with negative/invalid depth: {negative_depth_count}")
    print(f"  Most common depths: {depth_counts.most_common(5)}")

    if ROOT_NODE in data:
        print(f"  Depth of Phenotype Root ({ROOT_NODE}): {data[ROOT_NODE]}")
    else:
        print(f"  Phenotype Root ({ROOT_NODE}) not found in depth data.")

    sample_keys = list(data.keys())[:num_samples]
    print(f"\nShowing sample depths for first {len(sample_keys)} terms:")
    for term_id in sample_keys:
        print(f"  {term_id}: {data.get(term_id, 'Not Found')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect HPO ancestor and depth pickle files."
    )
    parser.add_argument(
        "--ancestors",
        type=str,
        default=str(DEFAULT_ANCESTORS_PATH),
        help="Path to hpo_ancestors.pkl file.",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default=str(DEFAULT_DEPTHS_PATH),
        help="Path to hpo_term_depths.pkl file.",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of sample terms to show."
    )

    args = parser.parse_args()

    ancestors_path = Path(args.ancestors).resolve()
    depths_path = Path(args.depths).resolve()

    print(f"Inspecting Ancestors File: {ancestors_path}")
    ancestor_data = load_pickle_file(ancestors_path)
    if ancestor_data:
        inspect_ancestors(ancestor_data, num_samples=args.samples)

    print("\n" + "=" * 50 + "\n")

    print(f"Inspecting Depths File: {depths_path}")
    depth_data = load_pickle_file(depths_path)
    if depth_data:
        inspect_depths(depth_data, num_samples=args.samples)
