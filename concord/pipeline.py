"""
concord.pipeline
================
Core workflows:

* run_file(...) – stream-process a CSV/TSV/JSON table
* run_pair(...) – compare two ad-hoc strings (used by CLI)

Supported modes

    llm     → only the LLM
    local   → cosine heuristic only
    dual    → embeddings + LLM
    simhint → embeddings + “similarity hint” line in the prompt
"""

from __future__ import annotations

import enum
import pathlib as P
import re
import yaml
from typing import Set, Tuple

import pandas as pd
from tqdm import tqdm

from .local.embeddings import similarity
from .llm.argo_gateway import ArgoGatewayClient, llm_label

# ────────────────────────────────────────────────────────────────────
class Mode(str, enum.Enum):
    LLM     = "llm"
    LOCAL   = "local"
    DUAL    = "dual"
    SIMHINT = "simhint"            # ⬅ new experimental mode

_SIM_COL = "similarity_Pubmedbert"

# ....................................................................
def _local_rule(s: float | None) -> str:
    """Three-way heuristic for LOCAL mode."""
    if s is None:
        return "Uninformative"
    if s > 0.90:
        return "Exact"
    if s > 0.60:
        return "Related"
    return "Different"

# ....................................................................
def _infer_cols(df: pd.DataFrame,
                col_a: str | None,
                col_b: str | None) -> Tuple[str, str]:
    """Guess which two columns hold annotation strings."""
    if col_a and col_b:
        return col_a, col_b

    # conventional names
    if {"annotation_a", "annotation_b"}.issubset(df.columns):
        return "annotation_a", "annotation_b"
    if {"old_annotation", "new_annotation"}.issubset(df.columns):
        return "old_annotation", "new_annotation"

    # first two textual columns not ending in “…id”
    text_cols = [
        c for c in df.columns
        if df[c].dtype == object and not re.search(r"id$", c, re.I)
    ]
    if len(text_cols) < 2:
        raise ValueError(
            "Could not infer annotation columns – please supply --col-a / --col-b."
        )
    return text_cols[0], text_cols[1]

# ....................................................................
def _load_any(path: P.Path, sep: str | None) -> pd.DataFrame:
    """Load .csv / .tsv / .json with the fast C parser when possible."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, sep=sep or ",", engine="c")
    if ext in {".tsv", ".tab"}:
        return pd.read_csv(path, sep=sep or "\t", engine="c")
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported extension: {ext}")

# ────────────────────────────────────────────────────────────────────
def run_file(
    file_path: P.Path,
    cfg_path: P.Path,
    col_a: str | None = None,
    col_b: str | None = None,
    *,
    out_path: P.Path | None,
    sep: str | None = None,
) -> P.Path:
    """
    Incrementally process *file_path* and append results to *out_path*.

    * Safe to resume – already-processed pairs are skipped.
    * Streams one row at a time so large files fit in RAM.
    """
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))

    df = _load_any(file_path, sep=sep)
    col_a, col_b = _infer_cols(df, col_a, col_b)

    out_file  = out_path or file_path.with_suffix(".concordia.csv")
    write_hdr = not out_file.exists()
    processed: Set[Tuple[str, str]] = set()

    if not write_hdr:
        done = pd.read_csv(out_file, usecols=[col_a, col_b])
        processed = set(done.itertuples(index=False, name=None))

    # create one LLM client if any mode needs it
    if mode in {Mode.LLM, Mode.DUAL, Mode.SIMHINT}:
        llm = ArgoGatewayClient(**cfg["llm"])

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing"):
        t1, t2 = getattr(row, col_a), getattr(row, col_b)
        if (t1, t2) in processed:
            continue

        sim: float | None = None
        label: str
        note:  str = ""

        # ─ embeddings first when needed ─
        if mode != Mode.LLM:
            sim = similarity(t1, t2)

        # ─ choose path by mode ─
        if mode == Mode.LLM:
            label, note = llm_label(t1, t2, llm, with_note=True)

        elif mode == Mode.LOCAL:
            label = _local_rule(sim)

        elif mode == Mode.DUAL:
            label, note = llm_label(t1, t2, llm, with_note=True)

        else:                               # SIMHINT
            if sim is None:                 # paranoia; shouldn’t happen
                label, note = llm_label(t1, t2, llm, with_note=True)
            else:
                # prepend hint line
                llm_hint = (
                    f"Cosine similarity (PubMedBERT) ≈ {sim:.3f} "
                    "(weak prior—override if biology disagrees)."
                )
                label, note = llm_label(
                    t1, t2, llm, with_note=True,  # type: ignore[arg-type]
                )
                # keep note unchanged – we only tweak the prompt

        # ─ write one row ─
        rec = {**row._asdict(), _SIM_COL: sim, "label": label, "note": note}
        pd.DataFrame([rec]).to_csv(
            out_file, index=False, header=write_hdr, mode="a"
        )
        write_hdr = False

    return out_file

# ────────────────────────────────────────────────────────────────────
def run_pair(text1: str, text2: str, cfg_path: P.Path) -> Tuple[str, float | None, str]:
    """
    Compare two ad-hoc strings outside the file workflow.
    Returns (label, similarity, note)
    """
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))

    sim = None if mode == Mode.LLM else similarity(text1, text2)

    if mode in {Mode.LLM, Mode.DUAL, Mode.SIMHINT}:
        client = ArgoGatewayClient(**cfg["llm"])
        label, note = llm_label(text1, text2, client, with_note=True)
    else:
        label, note = _local_rule(sim), ""

    return label, sim, note