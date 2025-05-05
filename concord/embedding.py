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
import logging
import time
import functools
import torch
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from sentence_transformers import SentenceTransformer, util

# Configure logging
logger = logging.getLogger(__name__)

_MODEL_ID = "NeuML/pubmedbert-base-embeddings"
_model: SentenceTransformer | None = None
_embedding_cache: Dict[str, torch.Tensor] = {}
_MAX_CACHE_SIZE = 10000  # Maximum number of cached embeddings

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
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait_time:.2f}s")
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
def batch_embed(texts: List[str], cfg: Dict[str, Any] | None = None, 
                batch_size: int = 32) -> List[torch.Tensor]:
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
            batch = new_texts[i:i+batch_size]
            batch_indices = new_indices[i:i+batch_size]
            
            start_time = time.time()
            embeddings = model.encode(batch, convert_to_tensor=True, device=device)
            elapsed = time.time() - start_time
            
            logger.debug(f"Embedded batch of {len(batch)} texts in {elapsed:.2f}s")
            
            # Store results and update cache
            for j, embedding in enumerate(embeddings):
                idx = batch_indices[j]
                text = new_texts[j-i]
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
    
    Args:
        vec1: First tensor
        vec2: Second tensor
        
    Returns:
        Cosine similarity as float
    """
    try:
        return float(util.cos_sim(vec1, vec2))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        raise RuntimeError(f"Similarity calculation failed: {e}") from e


def similarity(a: str, b: str, cfg: Dict[str, Any] | None = None) -> float:
    """
    One-liner convenience: encode both strings and return the cosine.
    Uses batch embedding for efficiency.
    
    Args:
        a: First text
        b: Second text
        cfg: Optional configuration
        
    Returns:
        Cosine similarity between embeddings
    """
    try:
        embs = batch_embed([a, b], cfg)
        return float(util.cos_sim(embs[0], embs[1]))
    except Exception as e:
        logger.error(f"Error calculating text similarity: {e}")
        raise RuntimeError(f"Text similarity calculation failed: {e}") from e


def preload_model(cfg: Dict[str, Any] | None = None) -> None:
    """
    Preload the embedding model to avoid delays on first use.
    
    Args:
        cfg: Configuration dictionary with optional embedding settings
    """
    device = "cpu"
    if cfg and "embedding" in cfg:
        device = cfg.get("embedding", {}).get("device", "cpu")
    
    logger.info("Preloading embedding model...")
    _get_model(device)
    logger.info("Embedding model preloaded and ready")


def clear_cache() -> None:
    """Clear the embedding cache to free memory."""
    global _embedding_cache
    cache_size = len(_embedding_cache)
    _embedding_cache = {}
    logger.info(f"Embedding cache cleared ({cache_size} entries)")