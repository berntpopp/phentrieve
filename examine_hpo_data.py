#!/usr/bin/env python3
"""
Script to examine the precomputed HPO graph data.
"""

import pickle
import os


def main():
    # Load HPO ancestors
    ancestors_file = os.path.join("data", "hpo_ancestors.pkl")
    with open(ancestors_file, "rb") as f:
        ancestors = pickle.load(f)

    print(f"HPO ancestors: {type(ancestors)}, {len(ancestors)} terms")
    print("Sample entries:")
    for i, (term_id, ancestor_set) in enumerate(list(ancestors.items())[:3]):
        print(
            f"  {term_id}: {len(ancestor_set)} ancestors, e.g., {list(ancestor_set)[:3]}"
        )

    # Load HPO term depths
    depths_file = os.path.join("data", "hpo_term_depths.pkl")
    with open(depths_file, "rb") as f:
        depths = pickle.load(f)

    print(f"\nHPO term depths: {type(depths)}, {len(depths)} terms")
    print("Sample entries:")
    for term_id, depth in list(depths.items())[:3]:
        print(f"  {term_id}: depth {depth}")


if __name__ == "__main__":
    main()
