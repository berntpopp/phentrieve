"""
Dense retrieval functionality for HPO terms.

This module provides functionality for querying the ChromaDB index to retrieve
relevant HPO terms based on semantic similarity with input text.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import chromadb
from sentence_transformers import SentenceTransformer

from phentrieve.config import MIN_SIMILARITY_THRESHOLD
from phentrieve.utils import (
    calculate_similarity,
    generate_collection_name,
    get_default_index_dir,
    resolve_data_path,
)


def connect_to_chroma(
    index_dir: str, collection_name: str, model_name: Optional[str] = None
) -> Optional[chromadb.Collection]:
    """
    Connect to the ChromaDB index and retrieve the specified collection.

    Args:
        index_dir: Directory where ChromaDB indices are stored
        collection_name: Name of the collection to connect to
        model_name: Optional model name to handle dimension-specific
            collections

    Returns:
        ChromaDB collection or None if connection failed
    """
    try:
        # Convert Path to string and ensure it exists
        index_dir_str = str(index_dir)

        # Initialize ChromaDB client with proper settings to avoid tenant issues
        client = chromadb.PersistentClient(
            path=index_dir_str,
            settings=chromadb.Settings(
                anonymized_telemetry=False, allow_reset=True, is_persistent=True
            ),
        )

        try:
            # Get the collection
            collection = client.get_collection(name=collection_name)
            count = collection.count()
            logging.info(
                f"Connected to collection '{collection_name}' with {count} docs"
            )
            return collection
        except Exception as e:
            logging.error(f"Error getting collection '{collection_name}': {e}")

            # List available collections for debugging
            collections = client.list_collections()
            if collections:
                collection_names = [c.name for c in collections]
                collection_list = ", ".join(collection_names)
                logging.info(f"Available collections: {collection_list}")

                # If model_name is provided, check if we need to create a new format
                if model_name:
                    alternate_name = generate_collection_name(model_name)
                    if (
                        alternate_name != collection_name
                        and alternate_name in collection_names
                    ):
                        logging.info(f"Found alternate collection: {alternate_name}")
                        return client.get_collection(name=alternate_name)
            else:
                logging.error(f"No collections found in {index_dir}")
                logging.error("Please run setup script to build the index first.")

            return None

    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {index_dir}: {e}")
        return None


class DenseRetriever:
    """
    Dense retriever for HPO terms using ChromaDB.

    This class handles connecting to a ChromaDB collection and querying it
    to retrieve relevant HPO terms based on semantic similarity.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        collection: chromadb.Collection,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ):
        """
        Initialize the dense retriever.

        Args:
            model: SentenceTransformer model for encoding queries
            collection: ChromaDB collection to query
            min_similarity: Minimum similarity threshold for results
        """
        self.model = model
        self.collection = collection
        self.min_similarity = min_similarity
        # These will be set in from_model_name
        self.model_name: Optional[str] = None
        self.index_base_path: Optional[Path] = None

    @classmethod
    def from_model_name(
        cls,
        model: SentenceTransformer,
        model_name: str,
        index_dir: Optional[Union[str, Path]] = None,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ) -> Optional["DenseRetriever"]:
        """
        Create a retriever instance from a model name.

        Args:
            model: SentenceTransformer model instance
            model_name: Name of the model (used for collection naming)
            index_dir: Directory where ChromaDB indices are stored
            min_similarity: Minimum similarity threshold for results

        Returns:
            DenseRetriever instance or None if connection failed
        """
        collection_name = generate_collection_name(model_name)
        final_index_dir: Path
        if index_dir is not None:
            final_index_dir = Path(index_dir)
            logging.info(
                f"DenseRetriever.from_model_name: Using explicit index_dir: "
                f"{final_index_dir}"
            )
        else:
            # This will use get_default_index_dir(), which checks ENV VARS first
            final_index_dir = resolve_data_path(
                config_key="index_dir",  # Check phentrieve.yaml for 'index_dir' key
                default_func=get_default_index_dir,
            )
            logging.info(
                f"DenseRetriever.from_model_name: Resolved index dir: {final_index_dir}"
            )

        if not final_index_dir.exists() or not final_index_dir.is_dir():
            logging.error(
                f"DenseRetriever: Index dir '{final_index_dir}' not found or not a "
                f"directory."
            )
            env_index_dir_val = os.getenv("PHENTRIEVE_INDEX_DIR")
            env_data_root_val = os.getenv("PHENTRIEVE_DATA_ROOT_DIR")
            logging.error(
                f"  ENV VARS: PHENTRIEVE_INDEX_DIR='{env_index_dir_val}', "
                f"PHENTRIEVE_DATA_ROOT_DIR='{env_data_root_val}'"
            )
            return None

        collection = connect_to_chroma(
            str(final_index_dir), collection_name, model_name
        )

        if collection:
            instance = cls(model, collection, min_similarity)
            instance.model_name = model_name  # Store for reference
            instance.index_base_path = final_index_dir  # Store resolved path
            return instance
        else:
            logging.error(
                f"DenseRetriever: Failed to connect to Chroma collection "
                f"'{collection_name}' at '{final_index_dir}'."
            )
            return None

    def query(
        self, text: str, n_results: int = 10, include_similarities: bool = True
    ) -> dict[str, Any]:
        """
        Generate embedding for input text and query the HPO index.

        Args:
            text: The input text to query
            n_results: Number of results to retrieve
            include_similarities: Whether to include similarity scores in result

        Returns:
            Dictionary containing query results with distances and/or similarities
        """
        logging.info(f"Processing query: '{text}'")

        try:
            # Generate embedding for the query text
            # Get the device the model is on
            device = next(self.model.parameters()).device
            device_name = "cuda" if device.type == "cuda" else "cpu"

            query_embedding = self.model.encode([text], device=device_name)[0]

            # Get more initial results to allow better filtering
            query_n_results = n_results * 3
            logging.debug(f"Querying with {query_n_results} results")

            # Cast to dict[str, Any] since we'll be adding custom keys
            results: dict[str, Any] = dict(
                self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=query_n_results,
                    include=["documents", "metadatas", "distances"],
                )
            )

            # Add similarity scores if requested
            if (
                include_similarities
                and results.get("distances")
                and results["distances"][0]
            ):
                similarities = []
                for distance in results["distances"][0]:
                    similarity = calculate_similarity(distance)
                    similarities.append(similarity)

                results["similarities"] = [similarities]

            return results

        except Exception as e:
            logging.error(f"Error querying HPO index: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def filter_results(
        self,
        results: dict[str, Any],
        min_similarity: Optional[float] = None,
        max_results: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Filter query results by similarity threshold and maximum count.

        Args:
            results: Results dictionary from query method
            min_similarity: Override for minimum similarity threshold
            max_results: Maximum number of results to include

        Returns:
            Filtered results dictionary
        """
        if not results or not results.get("ids") or not results["ids"][0]:
            return results

        threshold = (
            min_similarity if min_similarity is not None else self.min_similarity
        )

        # Create filtered results
        filtered_ids = []
        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []
        filtered_similarities = []

        # Calculate similarities if not already included
        similarities = results.get("similarities", [[]])
        if not similarities[0]:
            similarities = [[calculate_similarity(d) for d in results["distances"][0]]]

        # Filter by similarity threshold
        filtered_indices = []
        for i, similarity in enumerate(similarities[0]):
            if similarity >= threshold:
                filtered_indices.append(i)

        # Sort by similarity (descending)
        filtered_indices.sort(key=lambda i: similarities[0][i], reverse=True)

        # Apply max_results limit if provided
        if max_results is not None and max_results > 0:
            filtered_indices = filtered_indices[:max_results]

        # Build filtered results
        for i in filtered_indices:
            filtered_ids.append(results["ids"][0][i])
            filtered_documents.append(results["documents"][0][i])
            filtered_metadatas.append(results["metadatas"][0][i])
            filtered_distances.append(results["distances"][0][i])
            filtered_similarities.append(similarities[0][i])

        return {
            "ids": [filtered_ids],
            "documents": [filtered_documents],
            "metadatas": [filtered_metadatas],
            "distances": [filtered_distances],
            "similarities": [filtered_similarities],
        }
