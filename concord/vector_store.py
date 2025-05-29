"""
Vector store for retrieving examples and documents.

This module provides a flexible vector storage system that can store and retrieve
both structured classification examples and raw documents. It is designed to support
both RAC (Retrieval-Augmented Classification) and RAG (Retrieval-Augmented Generation).
"""

import json
import logging
import pathlib as P
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Flexible vector database for storing and retrieving examples and documents."""

    def __init__(self, data_path: Union[str, P.Path]):
        """Initialize the vector store.

        Args:
            data_path: Path to the JSON file for storing the vector data
        """
        self.data_path = P.Path(data_path)
        self.items = []
        self.embeddings = []
        self.item_types = []  # Store type of each item ("example", "document", etc.)
        self._load()

    def _load(self):
        """Load data from disk."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)
                    self.items = data.get("items", [])
                    self.embeddings = [
                        np.array(emb) for emb in data.get("embeddings", [])
                    ]
                    self.item_types = data.get(
                        "item_types", ["example"] * len(self.items)
                    )
                logger.info(f"Loaded {len(self.items)} items from vector store")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                self.items = []
                self.embeddings = []
                self.item_types = []

    def save(self):
        """Save data to disk."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, "w") as f:
            json.dump(
                {
                    "items": self.items,
                    "embeddings": [emb.tolist() for emb in self.embeddings],
                    "item_types": self.item_types,
                },
                f,
            )
        logger.debug(f"Saved {len(self.items)} items to vector store")

    def add_item(
        self, item: Dict[str, Any], embedding: np.ndarray, item_type: str = "example"
    ):
        """Add an item with its embedding to the store.

        Args:
            item: The item data (dictionary)
            embedding: The vector embedding for the item
            item_type: Type of the item ("example", "document", etc.)
        """
        self.items.append(item)
        self.embeddings.append(embedding)
        self.item_types.append(item_type)
        self.save()
        logger.debug(f"Added {item_type} to vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 3,
        item_type: Optional[str] = None,
        threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find similar items by cosine similarity.

        Args:
            query_embedding: The embedding vector to search for
            limit: Maximum number of results to return
            item_type: Filter by item type (e.g., "example", "document")
            threshold: Minimum similarity score (0-1) to include in results

        Returns:
            List of tuples containing (item, similarity_score)
        """
        if not self.embeddings:
            return []

        # Calculate cosine similarity for all items
        similarities = []
        for i, emb in enumerate(self.embeddings):
            # Skip items that don't match the requested type
            if item_type and self.item_types[i] != item_type:
                continue

            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            if sim >= threshold:
                similarities.append((i, sim))

        # Sort by similarity (highest first) and take top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:limit]

        # Return items with their similarity scores
        return [(self.items[i], score) for i, score in top_results]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        type_counts = {}
        for t in self.item_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        return {"total_items": len(self.items), "type_counts": type_counts}

    def clear(self):
        """Clear all items from the store."""
        self.items = []
        self.embeddings = []
        self.item_types = []
        self.save()
        logger.info("Cleared all items from vector store")
