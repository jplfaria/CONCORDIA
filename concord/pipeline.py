"""
concord.pipeline
----------------
Load input, pick engine(s), stream results row-by-row (crash safe).
"""

from __future__ import annotations
import pathlib as P, re
from typing import Optional

import pandas as pd
from tqdm import tqdm
import yaml

from .local.embeddings import similarity
from .llm.argo_gateway import ArgoGatewayClient, llm_label


class Mode(str):
    LOCAL = "local"
    LLM   = "llm"
    DUAL  = "dual"    # embeddings + LLM; no hybrid anymore


# ----------------------------------------------------------------------
_EMB_ID = "NeuML/pubmedbert-base-embeddings"
_SIM_COL = f"similarity_{_EMB_ID.split('/')[-1].split('-')[0].capitalize()}"

_UNINFO_RE = re.compile(
    r"\b(hypo(?:thetical)? protein|uncharacteri[sz]ed|gp\d+|orf\d+|fig\|\d)",
    re.I,
)
def _uninformative(s: str) -> bool:
    return bool(_UNINFO_RE.search(s)) or len(s.split()) <= 2


def _local_label(sim: float, a: str, b: str) -> str:
    if _uninformative(a) or _uninformative(b):
        return "Uninformative"
    if sim > 0.93:
        return "Exact"
    if sim > 0.78:
        return "Synonym"
    if 0.50 < sim <= 0.78:
        return "Related"
    return "Different"


# ----------------------------------------------------------------------
def _infer_cols(df: pd.DataFrame, col_a, col_b):
    if col_a and col_b:
        return col_a, col_b
    text_cols = [
        c for c in df.columns
        if df[c].dtype == object and not c.lower().endswith("id")
    ]
    if len(text_cols) < 2:
        raise ValueError("Could not infer annotation columns.")
    return text_cols[0], text_cols[1]


# ----------------------------------------------------------------------
def _load_table(path: P.Path, sep: str | None):
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, sep=sep or ",")
    if ext in {".tsv", ".tab"}:
        return pd.read_csv(path, sep=sep or "\t")
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type {ext}")


# ----------------------------------------------------------------------
def run_file(
    file_path: P.Path,
    cfg_path: P.Path,
    col_a: Optional[str],
    col_b: Optional[str],
    *,
    out_path: P.Path | None,
    sep: str | None = None,
) -> P.Path:
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg.get("engine", {}).get("mode", "llm"))

    df = _load_table(file_path, sep)
    col_a, col_b = _infer_cols(df, col_a, col_b)

    out_file = out_path or file_path.with_suffix(".concordia.csv")
    write_hdr = not out_file.exists()
    done = set()
    if not write_hdr:
        done = set(pd.read_csv(out_file)[[col_a, col_b]].itertuples(index=False, name=None))

    if mode in {Mode.LLM, Mode.DUAL}:
        llm = ArgoGatewayClient(**cfg.get("llm", {}))

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing"):
        t1, t2 = getattr(row, col_a), getattr(row, col_b)
        if (t1, t2) in done:
            continue

        sim = None if mode == Mode.LLM else similarity(t1, t2)

        if mode == Mode.LOCAL:
            label, note = _local_label(sim, t1, t2), ""

        elif mode == Mode.LLM:
            label, note = llm_label(t1, t2, llm, with_note=True)

        else:  # DUAL
            label, note = llm_label(t1, t2, llm, with_note=True)

        rec = {**row._asdict(), _SIM_COL: sim, "label": label, "note": note}
        pd.DataFrame([rec]).to_csv(out_file, index=False, header=write_hdr, mode="a")
        write_hdr = False

    return out_file


# ----------------------------------------------------------------------
def run_pair(a: str, b: str, cfg_path: P.Path):
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg.get("engine", {}).get("mode", "llm"))

    sim = None if mode == Mode.LLM else similarity(a, b)

    if mode == Mode.LOCAL:
        lbl, note = _local_label(sim, a, b), ""
    else:
        client = ArgoGatewayClient(**cfg.get("llm", {}))
        lbl, note = llm_label(a, b, client, with_note=True)

    return lbl, sim, note