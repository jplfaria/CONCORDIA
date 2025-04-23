import pandas as pd, yaml, enum, pathlib as P, re
from tqdm import tqdm
from .local.embeddings import similarity
from .llm.argo_gateway import ArgoGatewayClient, llm_label

class Mode(str, enum.Enum):
    LOCAL = "local"
    LLM = "llm"
    HYBRID = "hybrid"

def _local_rule(s: float) -> str:
    if s > .9:  return "Identical"
    if s > .6:  return "Partial"
    return "New"

# -------------------------------------------------------------
def _infer_cols(df, col_a, col_b):
    if col_a and col_b:
        return col_a, col_b
    if {"old_annotation", "new_annotation"}.issubset(df.columns):
        return "old_annotation", "new_annotation"
    # take first two non-id-ish columns
    text_cols = [c for c in df.columns
                 if df[c].dtype == object and not re.search(r'id$', c, re.I)]
    if len(text_cols) < 2:
        raise ValueError("Could not infer annotation columns.")
    return text_cols[0], text_cols[1]

# -------------------------------------------------------------
def run_file(csv_path: P.Path, cfg_path: P.Path,
             col_a=None, col_b=None, *, out_path: P.Path | None) -> P.Path:

    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))
    df   = pd.read_csv(csv_path)

    col_a, col_b = _infer_cols(df, col_a, col_b)

    if mode in (Mode.LLM, Mode.HYBRID):
        llm = ArgoGatewayClient(**cfg["llm"])

    lo, hi = cfg["engine"]["hybrid_thresholds"].values()
    rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing"):
        text1, text2 = getattr(row, col_a), getattr(row, col_b)
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

        rows.append({**row._asdict(),
                     "similarity": sim,
                     "label": label,
                     "note": note})

    out_file = out_path or csv_path.with_suffix(".concordia.csv")
    pd.DataFrame(rows).to_csv(out_file, index=False)
    return out_file

# -------------------------------------------------------------
def run_pair(text1, text2, cfg_path: P.Path):
    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))

    sim = None if mode == Mode.LLM else similarity(text1, text2)

    if mode == Mode.LLM:
        client = ArgoGatewayClient(**cfg["llm"])
        label, note = llm_label(text1, text2, client, with_note=True)
    elif mode == Mode.LOCAL:
        label, note = _local_rule(sim), ""
    else:
        lo, hi = cfg["engine"]["hybrid_thresholds"].values()
        if sim < lo or sim > hi:
            label, note = _local_rule(sim), ""
        else:
            client = ArgoGatewayClient(**cfg["llm"])
            label, note = llm_label(text1, text2, client, with_note=True)

    return label, sim, note