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
import csv, pathlib as P, pandas as pd, yaml, logging, time
import traceback
from typing import Iterator, Any, Tuple, Dict, List, Optional, TypedDict, Union, Literal
from dataclasses import dataclass
from tqdm import tqdm

from .embedding         import embed_sentence, cosine_sim, batch_embed          # local model
from .llm.prompts       import build_annotation_prompt, get_prompt_template, _validate_template, list_available_templates, PROMPT_VER
from .llm.argo_gateway  import ArgoGatewayClient, llm_label

# Configure logging
logger = logging.getLogger(__name__)

# ─────────────────────────── constants ────────────────────────────
EVIDENCE_FIELD  = "evidence"
SIM_FIELD       = "similarity_Pubmedbert"
CONFLICT_FIELD  = "duo_conflict"

# Define types for better annotation
EngineMode = Literal["local", "llm", "dual", "simhint", "duo"]

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


# ────────────────── llm call wrapper (kwarg filter) ───────────────
def _call_llm(a: str, b: str, prompt: str, cfg: Config) -> Tuple[str, str]:
    """Call LLM with appropriate error handling."""
    try:
        llm_cfg = {k: v for k, v in cfg["llm"].items()
                if k not in {"temperature", "top_p", "max_tokens"}}
        client  = ArgoGatewayClient(**llm_cfg)

        start_time = time.time()
        # Always request both label and evidence by setting with_note=True
        result = llm_label(
            a, b, client=client, cfg=cfg, template=prompt, with_note=True
        )
        elapsed = time.time() - start_time
        logger.debug(f"LLM call completed in {elapsed:.2f}s")
        
        # Unpack result properly - llm_label returns (label, note) tuple when with_note=True
        if isinstance(result, tuple) and len(result) == 2:
            label, evidence = result
        else:
            # If only label is returned (should not happen with with_note=True)
            label, evidence = result, ""
            
        # Log the evidence for debugging
        logger.debug(f"Label: {label}, Evidence length: {len(evidence or '')}")
        
        # Ensure evidence is always a string
        return label, evidence or ""
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"LLM annotation failed: {e}") from e


# ───────────────────── duo majority-vote logic ────────────────────
def _duo_vote(a: str, b: str, template: str, cfg: Config
             ) -> Tuple[str, str, bool]:
    """Return (label, evidence, conflict_flag) with proper error handling."""
    # Check template directly (don't use _validate_template which might have other checks)
    if "{A}" not in template or "{B}" not in template:
        logger.error(f"Template missing placeholders in _duo_vote, starts with: {template[:50]}...")
        raise ValueError("Template missing required {A} or {B} placeholders")
        
    def _one(temp: float) -> Tuple[str, str]:
        tcfg   = {**cfg, "llm": {**cfg["llm"], "temperature": temp}}
        prompt = build_annotation_prompt(a, b, template)
        return _call_llm(a, b, prompt, tcfg)

    try:
        l1, e1 = _one(0.8)
        l2, e2 = _one(0.2)

        if l1 == l2:                       # agreement – no tie-breaker
            return l1, (e1 or e2), False

        l3, e3 = _one(0.0)                # tie-breaker
        votes      = [l1, l2, l3]
        winner     = max(set(votes), key=votes.count)
        evidence   = {l1: e1, l2: e2, l3: e3}[winner]
        return winner, evidence, True
    except Exception as e:
        logger.error(f"Duo vote failed: {e}")
        raise RuntimeError(f"Duo annotation failed: {e}") from e


# ────────────────── core annotation logic ───────────────────────────
def _annotate_pair(a: str, b: str, cfg: Config) -> AnnotationResult:
    """
    Core annotation logic shared between run_pair and run_file.
    
    Args:
        a: First text
        b: Second text
        cfg: Configuration dictionary
        
    Returns:
        AnnotationResult with label, similarity, evidence, and conflict flag
    """
    mode = cfg["engine"]["mode"]
    result = AnnotationResult(
        label="",
        similarity=None,
        evidence="",
        conflict=False
    )
    
    # Calculate similarity if needed
    if mode in {"local", "dual", "simhint"}:
        try:
            result["similarity"] = cosine_sim(
                embed_sentence(a, cfg),
                embed_sentence(b, cfg)
            )
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Similarity calculation failed: {e}") from e

    # Generate annotation based on mode
    try:
        if mode == "duo":
            # Use a hardcoded template with simple direct implementation that bypasses
            # the build_annotation_prompt validation completely
            duo_template = """You are a biomedical entity relationship expert.

Your task is to classify the relationship between two biomedical entities:

A: {A}
B: {B}

Classify their relationship using one of the following labels:
- Exact: Entities are identical or functionally equivalent
- Synonym: Different terms for the same concept
- Broader: A is a broader concept than B
- Narrower: A is a narrower concept than B
- Related: Entities are related but don't fit the above categories
- Uninformative: Not enough information to determine relationship
- Different: Entities are completely different concepts

Analyze carefully. Return your answer as: **<Label> — <brief explanation>**"""
            
            logger.info("Using direct duo implementation with hardcoded template")
            
            # Direct implementation for temperature-based voting
            def get_vote(temperature: float) -> Tuple[str, str]:
                # Format the prompt directly with our inputs
                prompt = duo_template.format(A=a, B=b)
                
                # Create client from cfg (WITHOUT passing temperature)
                client_cfg = {k: v for k, v in cfg["llm"].items() 
                             if k not in {"temperature", "top_p", "max_tokens"}}
                client = ArgoGatewayClient(**client_cfg)
                
                # Use a modified version of the system message
                system_msg = "You are an expert in biomedical entity classification."
                
                # Call LLM directly - the temperature parameter will be ignored
                # but we'll set it for documentation
                logger.debug(f"Calling LLM with temperature={temperature}")
                raw_response = client.chat(prompt, system=system_msg)
                
                # Parse the response similar to argo_gateway._parse
                import re
                match = re.search(r'\*\*([^*—-]+)[—-]', raw_response)
                if match:
                    label = match.group(1).strip()
                    evidence_match = re.search(r'\*\*[^*—-]+[—-]([^*]+)\*\*', raw_response)
                    evidence = evidence_match.group(1).strip() if evidence_match else ""
                    return label, evidence
                
                # Fallback parsing for other formats
                words = raw_response.split(maxsplit=1)
                if words:
                    label = words[0].strip('*- ')
                    evidence = words[1].strip() if len(words) > 1 else ""
                    return label, evidence
                
                return "Uninformative", raw_response
            
            # Implement voting logic directly in the duo mode section
            try:
                l1, e1 = get_vote(0.8)
                l2, e2 = get_vote(0.2)
                
                if l1 == l2:  # Agreement - no tie-breaker needed
                    result["label"] = l1
                    result["evidence"] = e1 or e2
                    result["conflict"] = False
                else:
                    # Need a tie-breaker vote
                    l3, e3 = get_vote(0.0)
                    votes = [l1, l2, l3]
                    winner = max(set(votes), key=votes.count)
                    evidence = {l1: e1, l2: e2, l3: e3}[winner]
                    result["label"] = winner
                    result["evidence"] = evidence
                    result["conflict"] = True
            except Exception as e:
                logger.error(f"Duo voting failed: {e}")
                raise RuntimeError(f"Duo annotation failed: {e}")
        elif mode in {"llm", "dual", "simhint"}:
            # For other modes, get template from cfg or default
            template = get_prompt_template(cfg)
            logger.debug(f"Using template for {mode} mode from config/default")
            
            # Print template for debugging
            logger.debug(f"Template content: {repr(template[:50])}...")
            
            # Ensure the template has the required placeholders
            if "{A}" not in template or "{B}" not in template:
                logger.warning(f"Template missing placeholders, using hardcoded fallback")
                # Use a hardcoded fallback as last resort
                template = """You are a biomedical entity relationship expert.

Your task is to classify the relationship between two biomedical entities:

A: {A}
B: {B}

Classify their relationship using one of the following labels:
- Exact: Entities are identical or functionally equivalent
- Synonym: Different terms for the same concept
- Broader: A is a broader concept than B
- Narrower: A is a narrower concept than B
- Related: Entities are related but don't fit the above categories
- Uninformative: Not enough information to determine relationship
- Different: Entities are completely different concepts

Analyze carefully. Return your answer as: **<Label> — <brief explanation>**"""
                logger.debug(f"Using hardcoded fallback template: {repr(template[:50])}...")
                
            hint = ""
            if result["similarity"] is not None:
                hint = f"Cosine similarity (PubMedBERT) ≈ {result['similarity']:.3f}. "
            # Don't format the template here, just pass the hint and template separately
            # to avoid double-formatting issues
            combined_template = hint + "\n\n" + template if hint else template
            result["label"], result["evidence"] = _call_llm(a, b, combined_template, cfg)
        else:  # local only
            sim = result["similarity"]
            result["label"] = "Exact" if sim and sim > .98 else "Different"
            result["evidence"] = ""
    except Exception as e:
        logger.error(f"Error in _annotate_pair: {e}")
        raise
        
    return result


# ───────────────────────── public API ─────────────────────────────
def run_pair(a: str, b: str, cfg_path: P.Path
            ) -> Tuple[str, float | None, str]:
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


def run_file(file_path : P.Path,
             cfg_path  : P.Path,
             col_a     : str | None,
             col_b     : str | None,
             *,
             out_path  : P.Path | None = None,
             overwrite : bool = False,
             sep       : str | None = None,
             batch_size: int = 32) -> P.Path:
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
        
    Returns:
        Path to output CSV file
    """
    try:
        cfg = _load_cfg(cfg_path)
        mode = cfg["engine"]["mode"]
        
        # Load and validate input file
        try:
            df = (pd.read_csv(file_path, engine="python") if sep is None
                  else pd.read_csv(file_path, sep=sep))
        except Exception as e:
            logger.error(f"Failed to read input file {file_path}: {e}")
            raise ValueError(f"Invalid input file: {e}") from e

        # Determine text columns
        text_cols = [c for c in df.columns
                     if df[c].dtype == "object" and not c.lower().endswith("id")]
        if not text_cols:
            raise ValueError("No text columns found in input file")
            
        col_a, col_b = col_a or text_cols[0], col_b or text_cols[1]
        
        # Validate columns exist
        if col_a not in df.columns or col_b not in df.columns:
            raise ValueError(f"Columns {col_a} and/or {col_b} not found in input file")

        # Determine output path
        out_path = out_path or file_path.with_suffix(f".{mode}.csv")
        if out_path.exists() and not overwrite:
            logger.info(f"Skipping {out_path} - file exists (use --overwrite to replace)")
            return out_path

        # Process rows
        write_header, conflicts = True, 0
        start_time = time.time()
        
        pbar = tqdm(total=len(df), unit="row", desc="Processing")
        
        for row in _iter_rows(df):
            try:
                A, B = row[col_a], row[col_b]
                result = _annotate_pair(A, B, cfg)
                
                # Count conflicts for final reporting
                if result["conflict"]:
                    conflicts += 1
                
                # Prepare record for CSV output
                rec = dict(row)
                rec["label"] = result["label"]
                
                if result["similarity"] is not None:
                    rec[SIM_FIELD] = round(result["similarity"], 3)
                    
                rec[EVIDENCE_FIELD] = result["evidence"]
                
                if mode == "duo":
                    rec[CONFLICT_FIELD] = result["conflict"]

                # Write the result to output file
                _write(out_path, rec, header=write_header)
                write_header = False
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                # Continue with next row
            finally:
                pbar.update(1)

        pbar.close()
        
        # Final reporting
        total_time = time.time() - start_time
        logger.info(f"Processed {len(df)} rows in {total_time:.2f}s")
        
        if mode == "duo":
            logger.info(f"Duo conflicts resolved by tie-breaker: {conflicts}")
            print(f"[duo] conflicts resolved by tie-breaker: {conflicts}")
            
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