"""
concord.pipeline
----------------
File-mode:   streams results row-by-row (safe resume)
Pair-mode:   one-off comparison for two strings

Modes
-----
local   : cosine similarity + heuristic label
llm     : gateway label only
hybrid  : cosine; gateway in grey zone
dual    : cosine label *and* LLM label for every pair
"""

from __future__ import annotations
import csv, enum, pathlib as P, re, traceback, yaml
from typing import Optional

import pandas as pd
from tqdm import tqdm

from concord.io.loader    import load_table
from concord.local.embeddings import similarity
from concord.llm.argo_gateway import ArgoGatewayClient, llm_label


# ----------------------------------------------------------------------
class Mode(str, enum.Enum):
    LOCAL  = "local"
    LLM    = "llm"
    HYBRID = "hybrid"
    DUAL   = "dual"


def _local_rule(sim: float) -> str:
    if sim > 0.90:
        return "Identical"
    if sim > 0.60:
        return "Partial"
    return "New"


# ---------- column detection ------------------------------------------
def _infer_cols(df: pd.DataFrame, col_a: str | None, col_b: str | None):
    if col_a and col_b:
        return col_a, col_b

    if {"old_annotation", "new_annotation"}.issubset(df.columns):
        return "old_annotation", "new_annotation"

    text_cols = [
        c for c in df.columns
        if df[c].dtype == object and not re.search(r"id$", c, re.I)
    ]
    if len(text_cols) < 2:
        raise ValueError("Need at least two text columns "
                         "(or specify --col-a / --col-b).")
    return text_cols[0], text_cols[1]


def _row_key(row: dict, col_a: str, col_b: str):
    return (row.get("gene_id"), row[col_a], row[col_b])


# ---------- FILE-MODE --------------------------------------------------
def run_file(
    file_path: P.Path,
    cfg_path: P.Path,
    col_a: str | None = None,
    col_b: str | None = None,
    *,
    out_path: P.Path | None = None,
    sep: str | None = None,
) -> P.Path:

    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg.get("engine", {}).get("mode", "llm"))

    df = load_table(file_path, sep=sep)
    col_a, col_b = _infer_cols(df, col_a, col_b)

    # ---------- output setup ------------------------------------------
    out_file      = out_path or file_path.with_suffix(".concordia.csv")
    first_write   = not out_file.exists()
    done: set[tuple] = set()
    if not first_write:
        prev = pd.read_csv(out_file)
        done = {_row_key(r, col_a, col_b) for _, r in prev.iterrows()}

    fout   = out_file.open("a", newline="")
    writer = csv.DictWriter(
        fout,
        fieldnames=list(df.columns) +
        ["similarity", "cosine_label", "label", "note"]
    )
    if first_write:
        writer.writeheader()

    # ---------- helpers ----------------------------------------------
    if mode in (Mode.LLM, Mode.HYBRID, Mode.DUAL):
        llm = ArgoGatewayClient(**cfg.get("llm", {}))

    lo, hi = cfg.get("engine", {}).get(
        "hybrid_thresholds", {"lower": 0.30, "upper": 0.85}).values()

    # ---------- main loop --------------------------------------------
    for row in tqdm(df.to_dict(orient="records"), desc="Processing"):
        if _row_key(row, col_a, col_b) in done:
            continue

        text1, text2 = row[col_a], row[col_b]
        record = row.copy()
        try:
            sim = None if mode == Mode.LLM else similarity(text1, text2)
            record["similarity"] = sim

            # ------------- local label --------------------------------
            if mode != Mode.LLM:
                cosine_label = _local_rule(sim)
                record["cosine_label"] = cosine_label

            # ------------- choose final label -------------------------
            if mode == Mode.LOCAL:
                record["label"], record["note"] = cosine_label, ""

            elif mode == Mode.LLM:
                llm_lbl, note = llm_label(text1, text2, llm, with_note=True)
                record.update(label=llm_lbl, note=note)

            elif mode == Mode.DUAL:
                llm_lbl, note = llm_label(text1, text2, llm, with_note=True)
                record.update(label=llm_lbl, note=note)

            else:   # HYBRID
                if sim < lo or sim > hi:
                    record.update(label=cosine_label, note="")
                else:
                    llm_lbl, note = llm_label(text1, text2, llm, with_note=True)
                    record.update(label=llm_lbl, note=note)

        except Exception as exc:
            record.update(similarity=None,
                          label="Error",
                          note=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

        writer.writerow(record)
        fout.flush()

    fout.close()
    return out_file


# ---------- PAIR-MODE --------------------------------------------------
def run_pair(text1: str, text2: str, cfg_path: P.Path):
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg.get("engine", {}).get("mode", "llm"))

    sim = None if mode == Mode.LLM else similarity(text1, text2)

    if mode == Mode.LOCAL:
        return _local_rule(sim), sim, ""

    client = ArgoGatewayClient(**cfg.get("llm", {}))

    if mode == Mode.LLM:
        lbl, note = llm_label(text1, text2, client, with_note=True)
        return lbl, sim, note

    if mode == Mode.DUAL:
        cosine_lbl = _local_rule(sim)
        llm_lbl, note = llm_label(text1, text2, client, with_note=True)
        return f"{llm_lbl} (cos={cosine_lbl})", sim, note

    # hybrid
    lo, hi = cfg.get("engine", {}).get(
        "hybrid_thresholds", {"lower": 0.30, "upper": 0.85}).values()
    if sim < lo or sim > hi:
        return _local_rule(sim), sim, ""
    llm_lbl, note = llm_label(text1, text2, client, with_note=True)
    return llm_lbl, sim, note