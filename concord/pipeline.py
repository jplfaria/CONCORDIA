"""
concord.pipeline
================
Core work-flows:

* run_file(...) – stream-process a CSV/TSV/JSON table
* run_pair(...) – compare two ad-hoc strings (used by CLI)

Both apply the three Concordia modes:

    llm   → only the LLM
    local → cosine-heuristic only
    dual  → embeddings + LLM
"""

from __future__ import annotations
import enum, pathlib as P, re, yaml
from typing import Tuple, Set

import pandas as pd
from tqdm import tqdm

from .local.embeddings import similarity
from .llm.argo_gateway import ArgoGatewayClient, llm_label

# ──────────────────────────────────────────────────────────────────────
class Mode(str, enum.Enum):
    LLM   = "llm"      # gateway only
    LOCAL = "local"    # cosine heuristic only
    DUAL  = "dual"     # both

# ----------------------------------------------------------------------
_SIM_COL = "similarity_Pubmedbert"

def _local_rule(s: float | None) -> str:
    """Three-way heuristic used by LOCAL mode."""
    if s is None:
        return "Uninformative"
    if s > 0.90:
        return "Exact"
    if s > 0.60:
        return "Related"
    return "Different"

# ----------------------------------------------------------------------
def _infer_cols(df: pd.DataFrame,
                col_a: str | None,
                col_b: str | None) -> Tuple[str, str]:
    """Guess which two columns hold annotation strings."""
    if col_a and col_b:
        return col_a, col_b

    # honour conventional names if present
    if {"annotation_a", "annotation_b"}.issubset(df.columns):
        return "annotation_a", "annotation_b"
    if {"old_annotation", "new_annotation"}.issubset(df.columns):
        return "old_annotation", "new_annotation"

    # otherwise: first two *textual* columns that do **not** look like IDs
    is_text = [c for c in df.columns if df[c].dtype == object]
    text_cols = [c for c in is_text if not re.search(r"id$", c, re.I)]
    if len(text_cols) < 2:
        raise ValueError("Could not infer annotation columns – please supply --col-a / --col-b.")

    return text_cols[0], text_cols[1]

# ----------------------------------------------------------------------
def _load_any(path: P.Path, sep: str | None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".csv"}:
        return pd.read_csv(path, sep=sep or ",")
    if ext in {".tsv", ".tab"}:
        return pd.read_csv(path, sep=sep or "\t")
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported extension {ext}")

# ──────────────────────────────────────────────────────────────────────
def run_file(file_path: P.Path,
             cfg_path: P.Path,
             col_a: str | None = None,
             col_b: str | None = None,
             *,
             out_path: P.Path | None,
             sep: str | None = None) -> P.Path:
    """
    Incrementally process *file_path* and stream results to *out_path*.

    • Resumes safely if the output already exists.
    • Skips rows already present in the output.
    """
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))

    df   = _load_any(file_path, sep=sep)
    col_a, col_b = _infer_cols(df, col_a, col_b)

    out_file   = out_path or file_path.with_suffix(".concordia.csv")
    write_hdr  = not out_file.exists()
    processed: Set[Tuple[str, str]] = set()

    if not write_hdr:
        done_df = pd.read_csv(out_file, usecols=[col_a, col_b])
        processed = set(done_df.itertuples(index=False, name=None))

    # initialise LLM client once if needed
    if mode in {Mode.LLM, Mode.DUAL}:
        llm = ArgoGatewayClient(**cfg["llm"])

    # thresholds (kept in cfg for possible future use)
    lo, hi = cfg["engine"].get("hybrid_thresholds", {}).values() if "hybrid_thresholds" in cfg["engine"] else (0.3, 0.85)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing"):
        t1, t2 = getattr(row, col_a), getattr(row, col_b)
        if (t1, t2) in processed:
            continue

        sim: float | None = None
        label: str
        note:  str = ""

        # ─ local part ─
        if mode != Mode.LLM:
            sim = similarity(t1, t2)

        # ─ LLM part ─
        if mode == Mode.LLM:
            label, note = llm_label(t1, t2, llm, with_note=True)

        elif mode == Mode.LOCAL:
            label = _local_rule(sim)

        else:                           # DUAL
            label, note = llm_label(t1, t2, llm, with_note=True)

        rec = {**row._asdict(), _SIM_COL: sim, "label": label, "note": note}
        pd.DataFrame([rec]).to_csv(out_file, index=False,
                                   header=write_hdr, mode="a")
        write_hdr = False

    return out_file

# ──────────────────────────────────────────────────────────────────────
def run_pair(text1: str, text2: str, cfg_path: P.Path):
    """
    Compare two free-text annotations outside the file workflow.
    Returns (label, similarity, note).
    """
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))

    sim = None if mode == Mode.LLM else similarity(text1, text2)

    if mode in {Mode.LLM, Mode.DUAL}:
        client = ArgoGatewayClient(**cfg["llm"])
        label, note = llm_label(text1, text2, client, with_note=True)
    else:
        label, note = _local_rule(sim), ""

    return label, sim, note