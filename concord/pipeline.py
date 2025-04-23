"""
concord.pipeline
----------------
File-mode:  stream results row-by-row (safe resume)
Pair-mode :  instant label/similarity for two strings
"""

from __future__ import annotations
import csv, enum, pathlib as P, re, traceback, yaml
from typing import Optional

import pandas as pd
from tqdm import tqdm

from concord.io.loader import load_table
from concord.local.embeddings    import similarity
from concord.llm.argo_gateway    import ArgoGatewayClient, llm_label


# ----------------------------------------------------------------------
class Mode(str, enum.Enum):
    LOCAL  = "local"
    LLM    = "llm"
    HYBRID = "hybrid"


def _local_rule(sim: float) -> str:
    if sim > 0.90:
        return "Identical"
    if sim > 0.60:
        return "Partial"
    return "New"


# ---------- column utility ------------------------------------------------
def _infer_cols(df: pd.DataFrame, col_a: Optional[str], col_b: Optional[str]):
    """Find two annotation columns if user didn’t specify."""
    if col_a and col_b:
        return col_a, col_b

    if {"old_annotation", "new_annotation"}.issubset(df.columns):
        return "old_annotation", "new_annotation"

    text_cols = [
        c
        for c in df.columns
        if df[c].dtype == object and not re.search(r"id$", c, re.I)
    ]
    if len(text_cols) < 2:
        raise ValueError("Could not infer annotation columns "
                         "(need at least two text columns).")
    return text_cols[0], text_cols[1]


def _row_key(row: dict, col_a: str, col_b: str):
    """Unique key to know if a row has been processed before."""
    return (row.get("gene_id"), row[col_a], row[col_b])


# ---------- file-mode ------------------------------------------------------
def run_file(
    file_path: P.Path,
    cfg_path: P.Path,
    col_a: str | None = None,
    col_b: str | None = None,
    *,
    out_path: P.Path | None = None,
    sep: str | None = None,
) -> P.Path:
    """
    Compare two annotation columns in a table (CSV/TSV/JSON).

    Results are appended incrementally; reruns skip completed pairs.
    """
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg.get("engine", {}).get("mode", "llm"))

    df = load_table(file_path, sep=sep)
    col_a, col_b = _infer_cols(df, col_a, col_b)

    # ------------------------------------------------------------------ output
    out_file      = out_path or file_path.with_suffix(".concordia.csv")
    header_needed = not out_file.exists()

    done: set[tuple] = set()
    if not header_needed:
        prev = pd.read_csv(out_file)
        done = {_row_key(r, col_a, col_b) for _, r in prev.iterrows()}

    fout   = out_file.open("a", newline="")
    writer = csv.DictWriter(fout,
                            fieldnames=list(df.columns) + ["similarity",
                                                           "label",
                                                           "note"])
    if header_needed:
        writer.writeheader()

    # ------------------------------------------------------------------ prep
    if mode in (Mode.LLM, Mode.HYBRID):
        llm = ArgoGatewayClient(**cfg.get("llm", {}))

    lo, hi = cfg.get("engine", {}).get("hybrid_thresholds",
                                       {"lower": 0.60, "upper": 0.85}
                                      ).values()

    # ------------------------------------------------------------------ loop
    for row in tqdm(df.to_dict(orient="records"), desc="Processing"):
        if _row_key(row, col_a, col_b) in done:
            continue

        text1, text2 = row[col_a], row[col_b]
        try:
            sim = None if mode == Mode.LLM else similarity(text1, text2)

            if mode == Mode.LLM:
                label, note = llm_label(text1, text2, llm, with_note=True)

            elif mode == Mode.LOCAL:
                label, note = _local_rule(sim), ""

            else:  # hybrid
                if sim < lo or sim > hi:
                    label, note = _local_rule(sim), ""
                else:
                    label, note = llm_label(text1, text2, llm, with_note=True)

        except Exception as exc:
            label, note, sim = "Error", f"{type(exc).__name__}: {exc}", None
            traceback.print_exc()

        writer.writerow({**row,
                         "similarity": sim,
                         "label": label,
                         "note": note})
        fout.flush()            # durability per row

    fout.close()
    return out_file


# ---------- pair-mode ------------------------------------------------------
def run_pair(text1: str, text2: str, cfg_path: P.Path):
    """
    Fast path for CLI --text-a / --text-b.
    Returns (label, similarity, note)
    """
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg.get("engine", {}).get("mode", "llm"))

    sim = None if mode == Mode.LLM else similarity(text1, text2)

    if mode == Mode.LLM:
        client = ArgoGatewayClient(**cfg.get("llm", {}))
        label, note = llm_label(text1, text2, client, with_note=True)

    elif mode == Mode.LOCAL:
        label, note = _local_rule(sim), ""

    else:  # hybrid
        lo, hi = cfg.get("engine", {}).get("hybrid_thresholds",
                                           {"lower": 0.60, "upper": 0.85}
                                          ).values()
        if sim < lo or sim > hi:
            label, note = _local_rule(sim), ""
        else:
            client = ArgoGatewayClient(**cfg.get("llm", {}))
            label, note = llm_label(text1, text2, client, with_note=True)

    return label, sim, note