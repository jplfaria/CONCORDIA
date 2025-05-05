# concord/embedding.py
"""
Light wrapper around the PubMedBERT sentence-embedding model
(keeps old function names/arg-lists so pipeline keeps working).

v1.1 (2025-05-02)
 • ADD batch processing
 • ADD error handling with retries
 • ADD caching for better performance
 • IMPROVE device handling from config
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import torch

# Configure logging
logger = logging.getLogger(__name__)

_MODEL_ID = "NeuML/pubmedbert-base-embeddings"
_model: SentenceTransformer | None = None
_embedding_cache: Dict[str, torch.Tensor] = {}
_MAX_CACHE_SIZE = 10000  # Maximum number of cached embeddings

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    # Fallback for environments without sentence_transformers (e.g., tests)
    logger = logging.getLogger(__name__)
    logger.warning(
        "sentence_transformers not installed; using dummy SentenceTransformer"
    )

    class SentenceTransformer:
        def __init__(self, model_id):
            pass

        def to(self, device):
            pass

        def encode(self, texts, convert_to_tensor=True, device=None):
            if isinstance(texts, (list, tuple)):
                return [torch.zeros(768) for _ in texts]
            return torch.zeros(768)

    util = None


# ── internal helpers ───────────────────────────────────────────────
def _get_model(device: Optional[str] = None) -> SentenceTransformer:
    """
    Get or initialize the embedding model.

    Args:
        device: The device to load the model on ('cpu', 'cuda', etc.)

    Returns:
        Initialized SentenceTransformer model
    """
    global _model
    if _model is None:
        try:
            start_time = time.time()
            logger.info(f"Loading embedding model {_MODEL_ID}...")

            # Use provided device or default to CPU
            device = device or "cpu"

            # Handle device availability
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = "cpu"

            _model = SentenceTransformer(_MODEL_ID)
            _model.to(device)

            elapsed = time.time() - start_time
            logger.info(f"Model loaded in {elapsed:.2f}s on {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")

    return _model


def _manage_cache() -> None:
    """
    Manage the embedding cache size, removing least recently used items if needed.
    """
    global _embedding_cache
    if len(_embedding_cache) > _MAX_CACHE_SIZE:
        # Remove 20% of the cache (the oldest entries)
        remove_count = int(0.2 * len(_embedding_cache))
        keys_to_remove = list(_embedding_cache.keys())[:remove_count]
        for key in keys_to_remove:
            del _embedding_cache[key]
        logger.debug(f"Cache cleaned: removed {remove_count} entries")


def with_retries(max_retries: int = 3, backoff_factor: float = 0.5) -> Callable:
    """Decorator to retry functions on exception with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    wait_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt+1} failed: {e}. Retrying in {wait_time:.2f}s"
                    )
                    time.sleep(wait_time)

        return wrapper

    return decorator


# ── public API (legacy + modern) ───────────────────────────────────
@with_retries(max_retries=3)
def embed_sentence(text: str, cfg: Dict[str, Any] | None = None) -> torch.Tensor:
    """
    Return a 768-d tensor for *text*.

    Uses caching to avoid recomputing embeddings for identical text.

    Args:
        text: Text to embed
        cfg: Configuration dictionary with optional embedding settings

    Returns:
        Tensor embedding of the text
    """
    # Check if embedding is already in cache
    if text in _embedding_cache:
        return _embedding_cache[text]

    # Extract device from config or use default
    device = "cpu"
    if cfg and "embedding" in cfg:
        device = cfg.get("embedding", {}).get("device", "cpu")

    try:
        model = _get_model(device)
        embedding = model.encode(text, convert_to_tensor=True, device=device)

        # Cache the result
        _embedding_cache[text] = embedding
        _manage_cache()

        return embedding
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise RuntimeError(f"Embedding failed: {e}") from e


@with_retries(max_retries=3)
def batch_embed(
    texts: List[str], cfg: Dict[str, Any] | None = None, batch_size: int = 32
) -> List[torch.Tensor]:
    """
    Embed multiple texts efficiently in batches.

    Args:
        texts: List of texts to embed
        cfg: Configuration dictionary
        batch_size: Size of batches for processing

    Returns:
        List of tensor embeddings
    """
    device = "cpu"
    if cfg and "embedding" in cfg:
        device = cfg.get("embedding", {}).get("device", "cpu")

    # Filter texts that are already cached
    new_texts = []
    new_indices = []
    results = [None] * len(texts)

    for i, text in enumerate(texts):
        if text in _embedding_cache:
            results[i] = _embedding_cache[text]
        else:
            new_texts.append(text)
            new_indices.append(i)

    # If all embeddings were cached, return early
    if not new_texts:
        return results

    try:
        model = _get_model(device)

        # Process in batches
        for i in range(0, len(new_texts), batch_size):
            batch = new_texts[i : i + batch_size]
            batch_indices = new_indices[i : i + batch_size]

            start_time = time.time()
            embeddings = model.encode(batch, convert_to_tensor=True, device=device)
            elapsed = time.time() - start_time

            logger.debug(f"Embedded batch of {len(batch)} texts in {elapsed:.2f}s")

            # Store results and update cache
            for j, embedding in enumerate(embeddings):
                idx = batch_indices[j]
                text = new_texts[j - i]
                results[idx] = embedding
                _embedding_cache[text] = embedding

        _manage_cache()
        return results
    except Exception as e:
        logger.error(f"Error batch embedding texts: {e}")
        raise RuntimeError(f"Batch embedding failed: {e}") from e


def cosine_sim(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Calculate cosine similarity between two tensors.
    """
    try:
        sim = (
            util.pytorch_cos_sim(vec1, vec2).item()
            if util
            else torch.nn.functional.cosine_similarity(
                vec1.unsqueeze(0), vec2.unsqueeze(0)
            ).item()
        )
        return sim
    except Exception as e:
        logger.error(f"Cosine similarity failed: {e}")
        raise RuntimeError(f"Cosine similarity calculation failed: {e}") from e


def similarity(text1: str, text2: str, cfg: Dict[str, Any] | None = None) -> float:
    """Compute cosine similarity between two texts."""
    return cosine_sim(embed_sentence(text1, cfg), embed_sentence(text2, cfg))


def preload_model(cfg: Dict[str, Any] | None = None):
    """Load embedding model into global cache."""
    device = None
    if cfg and "embedding" in cfg:
        device = cfg["embedding"].get("device")
    return _get_model(device)


def clear_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()
