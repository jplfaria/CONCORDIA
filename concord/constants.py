"""
concord.constants
================
Centralized constants for the CONCORDIA engine.

This module contains shared constants used throughout the codebase.
"""

from __future__ import annotations

from typing import Final, List, Set

# Output field names
EVIDENCE_FIELD: Final[str] = "evidence"
SIM_FIELD: Final[str] = "similarity_Pubmedbert"
CONFLICT_FIELD: Final[str] = "duo_conflict"

# Embedding model constants
EMBEDDING_MODEL_ID: Final[str] = "NeuML/pubmedbert-base-embeddings"
DEFAULT_DEVICE: Final[str] = "cpu"
DEFAULT_BATCH_SIZE: Final[int] = 32
MAX_CACHE_SIZE: Final[int] = 10000

# Prompt constants
DEFAULT_PROMPT_VERSION: Final[str] = "v1.0"
EXAMPLES_PER_BUCKET: Final[int] = 3

# Similarity thresholds
EXACT_SIMILARITY_THRESHOLD: Final[float] = 0.98

# Label set
LABEL_SET: Final[Set[str]] = {
    "Exact",
    "Synonym",
    "Broader",
    "Narrower",
    "Related",
    "Uninformative",
    "Different",
}

# Engine modes
VALID_ENGINE_MODES: Final[List[str]] = [
    "local",
    "llm",
    "dual",
    "bucket",
    "duo",
]

# Default configuration paths
DEFAULT_CONFIG_PATH: Final[str] = "concord/config.yaml"

# API retries
MAX_RETRIES: Final[int] = 3
RETRY_BACKOFF_FACTOR: Final[float] = 0.5
