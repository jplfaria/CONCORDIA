"""
concord.pipeline
================
File-oriented workflows used by the CLI.

v1.4  (2025-05-02)
 • REFACTOR extract shared annotation logic
 • ADD proper error handling and logging
 • IMPROVE type annotations
 • ADD performance optimizations
"""

from __future__ import annotations

import csv
import logging
import pathlib as P
import re
import time
import traceback
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, TypedDict

import pandas as pd
import yaml
from tqdm import tqdm

from .embedding import cosine_sim, embed_sentence  # local model
from .llm.argo_gateway import SYSTEM_MSG, ArgoGatewayClient, _parse, llm_label
from .llm.prompt_builder import build_annotation_prompt
from .modes import (
    annotate_fallback,
    annotate_local,
    annotate_rac,
    annotate_vote,
    annotate_zero_shot,
)

# Configure logging
logger = logging.getLogger(__name__)

# ─────────────────────────── constants ────────────────────────────
EVIDENCE_FIELD = "evidence"
SIM_FIELD = "similarity_Pubmedbert"
CONFLICT_FIELD = "duo_conflict"

# Define types for better annotation
EngineMode = Literal["local", "zero-shot", "vote", "rac"]


# ──────────────────────────── types ───────────────────────────────
class AnnotationResult(TypedDict):
    """Result of an annotation operation."""

    label: str
    similarity: Optional[float]
    evidence: str
    conflict: bool


class Config(TypedDict):
    """Configuration dictionary type."""

    engine: Dict[str, Any]
    llm: Dict[str, Any]
    embedding: Dict[str, Any]


# ──────────────────────── small helpers ───────────────────────────
def _iter_rows(df: pd.DataFrame) -> Iterator[dict[str, Any]]:
    """Iterate through dataframe rows as dictionaries."""
    for row in df.to_dict("records"):
        yield row


def _write(out: P.Path, rec: dict[str, Any], *, header: bool) -> None:
    """Write a record to CSV file, creating the file if it doesn't exist."""
    try:
        mode = "a" if out.exists() else "w"
        with out.open(mode, newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rec.keys())
            if header and mode == "w":
                w.writeheader()
            w.writerow(rec)
    except (IOError, PermissionError) as e:
        logger.error(f"Failed to write to {out}: {e}")
        raise


# ───────────────────────── dispatcher ─────────────────────────────
_LEGACY_MODE_MAP: dict[str, str] = {
    "llm": "zero-shot",
    "duo": "vote",
    "dual": "vote",
}

_ENGINE_MAP: dict[str, Any] = {
    "local": annotate_local,
    "zero-shot": annotate_zero_shot,
    "vote": annotate_vote,
    "rac": annotate_rac,
}


# ────────────────── core annotation logic ───────────────────────────
def _call_llm(a: str, b: str, prompt: str, cfg: Config) -> tuple[str, str]:
    """Call LLM with appropriate error handling."""
    try:
        llm_cfg = {
            k: v
            for k, v in cfg["llm"].items()
            if k not in {"temperature", "top_p", "max_tokens"}
        }
        client = ArgoGatewayClient(**llm_cfg)

        start = time.time()
        res = llm_label(a, b, client=client, cfg=cfg, template=prompt, with_note=True)
        elapsed = time.time() - start
        logger.debug(f"LLM call completed in {elapsed:.2f}s")

        if isinstance(res, tuple) and len(res) == 2:
            label, evidence = res
        else:
            label, evidence = res, ""
        return label, evidence or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"LLM annotation failed: {e}") from e


def _duo_vote(a: str, b: str, template: str, cfg: Config) -> tuple[str, str, bool]:
    """Legacy duo vote: three-temperature LLM voting."""
    # Allow configurable vote temperatures (legacy duo vote)
    temps = cfg["engine"].get("vote_temps", [0.8, 0.2, 0.0])
    if "{A}" not in template or "{B}" not in template:
        raise ValueError("Template missing required {A} or {B} placeholders")

    def _one(temp: float) -> tuple[str, str]:
        tcfg = {**cfg, "llm": {**cfg["llm"], "temperature": temp}}
        prompt = build_annotation_prompt(a, b, template)
        return _call_llm(a, b, prompt, tcfg)

    try:
        l1, e1 = _one(temps[0])
        l2, e2 = _one(temps[1])
        if l1 == l2:
            return l1, e1 or e2, False
        l3, e3 = _one(temps[2] if len(temps) > 2 else 0.0)
        votes = [l1, l2, l3]
        winner = max(set(votes), key=votes.count)
        evidence = {l1: e1, l2: e2, l3: e3}[winner]
        return winner, evidence, True
    except Exception as e:
        logger.error(f"Duo vote failed: {e}")
        raise RuntimeError(f"Duo annotation failed: {e}") from e


def _annotate_pair(a: str, b: str, cfg: Config) -> AnnotationResult:
    """Dispatch to the appropriate annotation mode runner."""
    # Fix #10: Add validation for empty strings
    if not a or not b:
        return {
            "label": "Uninformative",
            "similarity": None,
            "evidence": "One or both inputs were empty",
            "conflict": False,
        }

    raw_mode = cfg["engine"]["mode"]
    mode = _LEGACY_MODE_MAP.get(raw_mode, raw_mode)
    # Inline local mode: use pipeline embed & cosim so tests can patch these
    if mode == "local":
        try:
            sim = cosine_sim(embed_sentence(a, cfg), embed_sentence(b, cfg))
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise RuntimeError(f"Similarity calculation failed: {e}") from e
        label = (
            "Exact" if sim > cfg["engine"].get("sim_threshold", 0.98) else "Different"
        )
        return {"label": label, "similarity": sim, "evidence": "", "conflict": False}
    runner = _ENGINE_MAP.get(mode)
    if runner is None:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        return runner(a, b, cfg)
    except Exception as e:
        logger.error(f"FALLBACK MODE ACTIVATED for mode '{mode}': {e}")
        return annotate_fallback(a, b, cfg, e)


# Helper for batching LLM calls
def _annotate_batch_chunk(
    rows: list[dict[str, Any]], cfg: Config, col_a: str, col_b: str
) -> list[AnnotationResult]:
    """
    Annotate multiple pairs in a single LLM request.

    Uses the same template and prompt format as single-pair processing
    to ensure consistent results between batched and non-batched modes.
    """
    # Get the base template that would be used for individual prompts
    from .llm.prompts import get_prompt_template

    base_template = get_prompt_template(cfg)

    # Calculate pairwise similarities if sim_hint enabled
    from .embedding import cosine_sim, embed_sentence

    use_sim_hint = cfg["engine"].get("sim_hint", False)
    sims: list[float | None] = []
    if use_sim_hint:
        for row in rows:
            a_val, b_val = row[col_a], row[col_b]
            sims.append(
                cosine_sim(embed_sentence(a_val, cfg), embed_sentence(b_val, cfg))
            )
    else:
        sims = [None] * len(rows)

    # Build a multi-pair prompt that preserves the template format for each pair
    lines = [
        "I need you to classify multiple pairs using the EXACT SAME criteria you would use for individual pairs."
    ]
    lines.append(
        "For EACH pair, answer with ONLY the format: <Pair Number>. <Label> — <brief explanation>"
    )
    lines.append(
        "Use EXACTLY the same classification approach for each pair as you would if evaluating one pair at a time."
    )
    lines.append("\nHere are the pairs to classify:\n")

    # Add each pair with its own properly formatted prompt
    for idx, row in enumerate(rows, start=1):
        a, b = row[col_a], row[col_b]
        # Create individual prompt for this pair using the same template as single-pair mode
        pair_template = base_template.replace("{A}", a).replace("{B}", b)
        # Add similarity hint if enabled
        if use_sim_hint:
            sim_hint = f"Cosine similarity ≈ {sims[idx-1]:.3f}"
            lines.append(f"PAIR {idx}:\n{sim_hint}\n\n{pair_template}\n")
        else:
            lines.append(f"PAIR {idx}:\n{pair_template}\n")

    prompt = "\n".join(lines)

    # Use the same LLM configuration as single-pair mode
    llm_cfg = {
        k: v
        for k, v in cfg["llm"].items()
        if k not in {"temperature", "top_p", "max_tokens"}
    }
    client = ArgoGatewayClient(**llm_cfg)
    raw = client.chat(prompt, system=SYSTEM_MSG)

    # Parse results - first split by pair markers then extract label from each
    results: list[AnnotationResult] = []

    # Extract each pair's response by looking for "PAIR N" or "N." pattern
    pair_responses = re.split(
        r"(?:PAIR\s+\d+[:.]*|^\s*\d+[).:])", raw, flags=re.MULTILINE
    )

    # First element is usually intro text, skip it
    pair_responses = [pr.strip() for pr in pair_responses if pr.strip()]

    # Parse each pair response using the same parser as single-pair mode
    for i, pr in enumerate(pair_responses[: len(rows)]):
        if not pr:
            continue
        label, note = _parse(pr)
        # Use original index and ensure sims index is in bounds
        sim_value = sims[i] if i < len(sims) else None
        results.append(
            {
                "label": label,
                "similarity": sim_value,
                "evidence": note,
                "conflict": False,
            }
        )

    # If we have fewer results than rows, pad with fallback results
    while len(results) < len(rows):
        results.append(
            {
                "label": "Unknown",
                "similarity": None,
                "evidence": "Failed to extract response from batch processing",
                "conflict": False,
            }
        )

    return results


# ───────────────────────── public API ─────────────────────────────
def run_pair(a: str, b: str, cfg_path: P.Path) -> Tuple[str, float | None, str]:
    """
    Annotate a single pair of texts.

    Args:
        a: First text
        b: Second text
        cfg_path: Path to configuration file

    Returns:
        Tuple of (label, similarity, evidence)
    """
    try:
        cfg = _load_cfg(cfg_path)
        result = _annotate_pair(a, b, cfg)
        return result["label"], result["similarity"], result["evidence"]
    except Exception as e:
        logger.error(f"run_pair failed: {e}")
        logger.debug(traceback.format_exc())
        raise


def run_file(
    file_path: P.Path,
    cfg_path: P.Path,
    col_a: str | None,
    col_b: str | None,
    *,
    out_path: P.Path | None = None,
    overwrite: bool = False,
    sep: str | None = None,
    batch_size: int = 32,
    llm_batch_size: int = 1,
) -> P.Path:
    """
    Process a file containing pairs of texts to annotate.

    Args:
        file_path: Path to input CSV file
        cfg_path: Path to configuration file
        col_a: Column name for first text
        col_b: Column name for second text
        out_path: Path to output CSV file (default: derived from input)
        overwrite: Whether to overwrite existing output file
        sep: CSV separator (default: auto-detect)
        batch_size: Size of batches for processing
        llm_batch_size: Size of batches for LLM calls (default: 1)

    Returns:
        Path to output CSV file
    """
    try:
        cfg = _load_cfg(cfg_path)
        mode = cfg["engine"]["mode"]

        # Load and validate input file
        try:
            df = (
                pd.read_csv(file_path, engine="python")
                if sep is None
                else pd.read_csv(file_path, sep=sep)
            )
        except Exception as e:
            logger.error(f"Failed to read input file {file_path}: {e}")
            raise ValueError(f"Invalid input file: {e}") from e

        # Determine text columns
        text_cols = [
            c
            for c in df.columns
            if df[c].dtype == "object" and not c.lower().endswith("id")
        ]
        if not text_cols:
            raise ValueError("No text columns found in input file")

        col_a, col_b = col_a or text_cols[0], col_b or text_cols[1]

        # Validate columns exist
        if col_a not in df.columns or col_b not in df.columns:
            raise ValueError(f"Columns {col_a} and/or {col_b} not found in input file")

        # Determine output path
        out_path = out_path or file_path.with_suffix(f".{mode}.csv")
        if out_path.exists() and not overwrite:
            logger.info(
                f"Skipping {out_path} - file exists (use --overwrite to replace)"
            )
            return out_path

        # Process rows
        write_header, conflicts, failed_chunks = True, 0, 0
        start_time = time.time()

        pbar = tqdm(total=len(df), unit="row", desc="Processing")

        rows = list(_iter_rows(df))
        mode_val = cfg["engine"]["mode"]
        # Fix #8: Mixed Mode Batch Processing - Check if mode supports batching
        supports_batch = mode_val in [
            "zero-shot"
        ]  # Add other supported modes here as needed
        chunk_size = llm_batch_size if supports_batch and llm_batch_size > 1 else 1
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            try:
                if mode_val == "zero-shot" and chunk_size > 1:
                    results = _annotate_batch_chunk(chunk, cfg, col_a, col_b)
                else:
                    results = [_annotate_pair(r[col_a], r[col_b], cfg) for r in chunk]
                for row, result in zip(chunk, results):
                    if result["conflict"]:
                        conflicts += 1
                    rec = dict(row)
                    rec["relationship"] = result["label"]
                    if result.get("similarity") is not None:
                        rec[SIM_FIELD] = round(result["similarity"], 3)
                    rec[EVIDENCE_FIELD] = result["evidence"]
                    if mode_val in ("duo", "vote"):
                        rec[CONFLICT_FIELD] = result["conflict"]
                        rec["votes"] = ";".join(result.get("votes", []))
                    _write(out_path, rec, header=write_header)
                    write_header = False
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                failed_chunks += 1
            finally:
                pbar.update(len(chunk))
        pbar.close()

        # Final reporting
        total_time = time.time() - start_time
        logger.info(f"Processed {len(df)} rows in {total_time:.2f}s")

        if mode == "duo":
            logger.info(f"Duo conflicts resolved by tie-breaker: {conflicts}")
            print(f"[duo] conflicts resolved by tie-breaker: {conflicts}")

        # Report any failed chunks
        if failed_chunks > 0:
            logger.warning(f"{failed_chunks} chunks failed processing")
            print(f"[Warning] {failed_chunks} chunks failed during processing")

        return out_path
    except Exception as e:
        logger.error(f"run_file failed: {e}")
        logger.debug(traceback.format_exc())
        raise


# ──────────────────────────── utils ───────────────────────────────
def _load_cfg(path: P.Path) -> Config:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary
    """
    try:
        with path.open() as fh:
            cfg = yaml.safe_load(fh)

        # Validate minimum required configuration
        if "engine" not in cfg:
            raise ValueError("Configuration missing 'engine' section")
        if "mode" not in cfg["engine"]:
            raise ValueError("Configuration missing 'engine.mode' setting")

        # Add default sections if missing
        if "embedding" not in cfg:
            cfg["embedding"] = {}
        if "device" not in cfg["embedding"]:
            cfg["embedding"]["device"] = "cpu"

        return cfg
    except IOError as e:
        logger.error(f"Failed to read configuration file {path}: {e}")
        raise ValueError(f"Invalid configuration file: {e}") from e
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file {path}: {e}")
        raise ValueError(f"Invalid YAML in configuration: {e}") from e
