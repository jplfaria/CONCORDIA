# concord/embedding.py
"""
Light wrapper around PubMedBERT sentence-embedding model.

v1.2 (2025-05-02)
 • SIMPLIFIED caching without threading overhead
 • REMOVED redundant retry decorator (use utils.with_retries)
 • STREAMLINED device handling
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from .utils import with_retries

# Configure logging
logger = logging.getLogger(__name__)

_MODEL_ID = "NeuML/pubmedbert-base-embeddings"
_model: Optional["SentenceTransformer"] = None
_embedding_cache: Dict[str, torch.Tensor] = {}
_MAX_CACHE_SIZE = 5000  # Reduced cache size

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    logger.warning("sentence_transformers not installed; using dummy implementation")

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


def _get_model(device: Optional[str] = None) -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        start_time = time.time()
        logger.info(f"Loading embedding model {_MODEL_ID}...")

        # Simple device handling
        device = device or "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        _model = SentenceTransformer(_MODEL_ID)
        _model.to(device)

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s on {device}")

    return _model


def _manage_cache() -> None:
    """Simple cache management - remove oldest entries when full."""
    global _embedding_cache
    if len(_embedding_cache) > _MAX_CACHE_SIZE:
        # Remove 25% of oldest entries
        remove_count = _MAX_CACHE_SIZE // 4
        keys_to_remove = list(_embedding_cache.keys())[:remove_count]
        for key in keys_to_remove:
            del _embedding_cache[key]
        logger.debug(f"Cache cleaned: removed {remove_count} entries")


@with_retries(max_retries=3)
def embed_sentence(text: str, cfg: Dict[str, Any] | None = None) -> torch.Tensor:
    """Return a 768-d tensor for text with simple caching."""
    # Check cache first
    if text in _embedding_cache:
        return _embedding_cache[text]

    # Extract device from config
    device = "cpu"
    if cfg and "embedding" in cfg:
        device = cfg["embedding"].get("device", "cpu")

    try:
        model = _get_model(device)
        embedding = model.encode(text, convert_to_tensor=True, device=device)

        # Cache result
        _embedding_cache[text] = embedding
        _manage_cache()

        return embedding
    except Exception as e:
        logger.error(f"Embedding failed for text: {e}")
        raise RuntimeError(f"Embedding failed: {e}") from e


@with_retries(max_retries=3)
def batch_embed(
    texts: List[str], cfg: Dict[str, Any] | None = None, batch_size: int = 32
) -> List[torch.Tensor]:
    """Embed multiple texts efficiently."""
    device = "cpu"
    if cfg and "embedding" in cfg:
        device = cfg["embedding"].get("device", "cpu")

    # Check cache and separate cached vs new texts
    results = [None] * len(texts)
    new_texts = []
    new_indices = []

    for i, text in enumerate(texts):
        if text in _embedding_cache:
            results[i] = _embedding_cache[text]
        else:
            new_texts.append(text)
            new_indices.append(i)

    # Process new texts in batches
    if new_texts:
        model = _get_model(device)

        for i in range(0, len(new_texts), batch_size):
            batch = new_texts[i : i + batch_size]
            batch_indices = new_indices[i : i + batch_size]

            try:
                embeddings = model.encode(batch, convert_to_tensor=True, device=device)

                # Handle single vs multiple embeddings
                if len(batch) == 1:
                    embeddings = [embeddings]

                # Store results and cache
                for j, embedding in enumerate(embeddings):
                    idx = batch_indices[j]
                    results[idx] = embedding
                    _embedding_cache[batch[j]] = embedding

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Fallback to individual processing
                for j, text in enumerate(batch):
                    try:
                        embedding = model.encode(
                            text, convert_to_tensor=True, device=device
                        )
                        idx = batch_indices[j]
                        results[idx] = embedding
                        _embedding_cache[text] = embedding
                    except Exception as inner_e:
                        logger.error(f"Individual embedding failed for text: {inner_e}")
                        raise

    _manage_cache()
    return results


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    try:
        if util is not None:
            return float(util.pytorch_cos_sim(a, b))
        else:
            # Fallback implementation
            dot_product = torch.dot(a.flatten(), b.flatten())
            norm_a = torch.norm(a)
            norm_b = torch.norm(b)
            return float(dot_product / (norm_a * norm_b))
    except Exception as e:
        logger.error(f"Cosine similarity calculation failed: {e}")
        return 0.0


def similarity(text1: str, text2: str, cfg: Dict[str, Any] | None = None) -> float:
    """Compute cosine similarity between two text strings."""
    try:
        embedding1 = embed_sentence(text1, cfg)
        embedding2 = embed_sentence(text2, cfg)
        return cosine_sim(embedding1, embedding2)
    except Exception as e:
        logger.error(f"Text similarity calculation failed: {e}")
        return 0.0


def preload_model(cfg: Dict[str, Any] | None = None) -> None:
    """Preload the embedding model."""
    device = "cpu"
    if cfg and "embedding" in cfg:
        device = cfg["embedding"].get("device", "cpu")
    _get_model(device)


def clear_cache() -> None:
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "cache_size": len(_embedding_cache),
        "max_size": _MAX_CACHE_SIZE,
        "utilization": len(_embedding_cache) / _MAX_CACHE_SIZE,
    }
