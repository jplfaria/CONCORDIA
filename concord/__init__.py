"""
concord
=======
CONCORDIA annotation engine.

v1.2 (2025-05-02)
 • ADD centralized logging configuration
 • ADD shared constants
 • IMPROVE overall package structure
"""

from __future__ import annotations
import logging
import logging.config
from typing import Dict, Any, Optional
from importlib.metadata import version

# Constants moved from pipeline.py
EVIDENCE_FIELD = "evidence"
SIM_FIELD = "similarity_Pubmedbert"
CONFLICT_FIELD = "duo_conflict"

# Setup logging
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the entire package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "concord": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    # Add file handler if log_file is specified
    if log_file:
        log_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "standard",
            "filename": log_file,
            "mode": "a",
        }
        log_config["loggers"]["concord"]["handlers"].append("file")
    
    logging.config.dictConfig(log_config)

# Setup default logging
setup_logging()

# Version
__version__ = "1.2.0"
__all__ = ["pipeline"]
