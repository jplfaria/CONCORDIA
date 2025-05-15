"""
Retrieval module for finding similar examples or documents.

This module provides functions for retrieving items from the vector store,
supporting both RAC (Retrieval-Augmented Classification) and
potential RAG (Retrieval-Augmented Generation) use cases.
"""

import logging
import pathlib as P
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embedding import embed_sentence
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Global vector store instance
_vector_store = None


def get_vector_store(cfg: Dict[str, Any]) -> VectorStore:
    """Get or initialize vector store.

    Args:
        cfg: Configuration dictionary

    Returns:
        Initialized VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        data_dir = P.Path(cfg.get("data_dir", "data"))
        store_path = data_dir / "vector_store.json"
        _vector_store = VectorStore(store_path)
        logger.debug(f"Initialized vector store at {store_path}")
    return _vector_store


def retrieve_similar_items(
    query_embedding: np.ndarray,
    cfg: Dict[str, Any],
    item_type: Optional[str] = "example",
    limit: int = 3,
    threshold: float = 0.0,
) -> List[Tuple[Dict[str, Any], float]]:
    """Retrieve similar items from the vector store.

    This generic retrieval function can be used to retrieve any type of item.

    Args:
        query_embedding: Embedding vector to search for
        cfg: Configuration dictionary
        item_type: Type of items to retrieve (None for all types)
        limit: Maximum number of results to return
        threshold: Minimum similarity score (0-1) to include

    Returns:
        List of tuples containing (item, similarity_score)
    """
    store = get_vector_store(cfg)
    items = store.search(
        query_embedding, limit=limit, item_type=item_type, threshold=threshold
    )
    logger.debug(f"Retrieved {len(items)} similar items of type '{item_type}'")
    return items


def retrieve_similar_examples(
    a: str, b: str, cfg: Dict[str, Any], limit: int = 3, threshold: float = 0.0
) -> List[Tuple[Dict[str, Any], float]]:
    """Retrieve similar classification examples for a pair of texts.

    This is a specialized retrieval function for classification examples.

    Args:
        a: First text
        b: Second text
        cfg: Configuration dictionary
        limit: Maximum number of results to return
        threshold: Minimum similarity score (0-1) to include

    Returns:
        List of tuples containing (example, similarity_score)
    """
    # Get embeddings for the current texts
    embedding_a = embed_sentence(a, cfg)
    embedding_b = embed_sentence(b, cfg)

    # Simple strategy: average the embeddings of both texts
    # More advanced strategies could be implemented later
    pair_embedding = (embedding_a + embedding_b) / 2

    # Retrieve examples with similarity scores
    return retrieve_similar_items(
        pair_embedding, cfg, item_type="example", limit=limit, threshold=threshold
    )


def add_classification_example(
    a: str,
    b: str,
    label: str,
    evidence: str,
    cfg: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
):
    """Add a classification example to the vector store.

    Args:
        a: First text
        b: Second text
        label: Classification label
        evidence: Justification or explanation
        cfg: Configuration dictionary
        metadata: Optional additional information about the example
    """
    # Get embeddings
    embedding_a = embed_sentence(a, cfg)
    embedding_b = embed_sentence(b, cfg)
    pair_embedding = (embedding_a + embedding_b) / 2

    # Create example with required fields
    example = {
        "text_a": a,
        "text_b": b,
        "label": label,
        "evidence": evidence,
        "timestamp": cfg.get("timestamp", None),  # Add timestamp if available
    }

    # Add optional metadata if provided
    if metadata:
        example.update(metadata)

    # Add to store
    store = get_vector_store(cfg)
    store.add_item(example, pair_embedding, item_type="example")
    logger.info(
        f"Added classification example to vector store: {label} - {a[:20]}... vs {b[:20]}..."
    )


def add_document(
    content: str,
    embedding: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    cfg: Dict[str, Any] = None,
):
    """Add a document to the vector store.

    This function allows storing general documents for RAG use cases.

    Args:
        content: Document content
        embedding: Pre-computed embedding (will be computed if None)
        metadata: Optional document metadata
        cfg: Configuration dictionary
    """
    if embedding is None and cfg is None:
        raise ValueError("Either embedding or cfg must be provided")

    # Compute embedding if not provided
    if embedding is None:
        embedding = embed_sentence(content, cfg)

    # Create document with required fields
    document = {
        "content": content,
        "timestamp": cfg.get("timestamp", None),
    }

    # Add optional metadata if provided
    if metadata:
        document.update(metadata)

    # Add to store
    store = get_vector_store(cfg)
    store.add_item(document, embedding, item_type="document")
    logger.info(f"Added document to vector store: {content[:50]}...")
