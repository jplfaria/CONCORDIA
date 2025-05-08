"""
concord.utils
============
Shared utilities for the CONCORDIA engine.

This module contains helper functions used across the codebase.
"""

from __future__ import annotations

import functools
import logging
import pathlib as P
import time
from typing import Any, Callable, Dict, List, TypeVar, Union

import yaml

from .constants import MAX_RETRIES, RETRY_BACKOFF_FACTOR

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


def with_retries(
    max_retries: int = MAX_RETRIES, backoff_factor: float = RETRY_BACKOFF_FACTOR
) -> Callable:
    """
    Decorator to retry functions on exception with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff

    Returns:
        Decorated function with retry capability
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    wait_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt+1} failed: {e}. Retrying in {wait_time:.2f}s"
                    )
                    time.sleep(wait_time)
            # This should never happen due to the raise above, but makes mypy happy
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


def load_yaml_config(path: Union[str, P.Path]) -> Dict[str, Any]:
    """
    Safely load and validate a YAML configuration file.

    Args:
        path: Path to the YAML file

    Returns:
        Parsed configuration dictionary

    Raises:
        ValueError: If the file cannot be read or parsed
    """
    try:
        path_obj = P.Path(path) if isinstance(path, str) else path
        with open(path_obj, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Config file {path} did not parse to a dictionary")

        return config
    except (IOError, yaml.YAMLError) as e:
        logger.error(f"Failed to load config from {path}: {e}")
        raise ValueError(f"Failed to load config from {path}: {e}")


def timing_decorator(func: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator to measure and log execution time of functions.

    Args:
        func: Function to time

    Returns:
        Decorated function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed_time:.4f}s")
        return result

    return wrapper


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation errors, empty if valid
    """
    errors = []

    # Check for required sections
    if "engine" not in config:
        errors.append("Missing 'engine' section")
    elif "mode" not in config["engine"]:
        errors.append("Missing 'engine.mode' setting")

    if "llm" not in config:
        errors.append("Missing 'llm' section")

    return errors


def ensure_dir_exists(path: Union[str, P.Path]) -> P.Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = P.Path(path) if isinstance(path, str) else path
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def chunked(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def validate_template(template: str, raise_error: bool = True) -> bool:
    """
    Validate that a template contains required placeholders.

    Args:
        template: Template string to validate
        raise_error: Whether to raise an error when validation fails

    Returns:
        True if template is valid, False otherwise

    Raises:
        ValueError: If template is invalid and raise_error is True
    """
    is_valid = "{A}" in template and "{B}" in template
    if not is_valid and raise_error:
        logger.error(f"Template missing required placeholders: {template[:50]}...")
        raise ValueError("Template missing required {A} or {B} placeholders")
    return is_valid
