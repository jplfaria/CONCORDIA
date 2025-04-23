import pandas as pd, csv, yaml, enum, pathlib as P, re, traceback
from tqdm import tqdm
from datetime import datetime
from .local.embeddings import similarity
from .llm.argo_gateway import ArgoGatewayClient, llm_label

class Mode(str, enum.Enum):
    LOCAL = "local"
    LLM   = "llm"
    HYBRID = "hybrid"

def _local_rule(sim: float) -> str:
    if sim > .9:  return "Identical"
    if sim > .6:  return "Partial"
    return "New"

# ---------- helpers ---------------------------------------------------
def _infer_cols(df, col_a, col_b):
    if col_a and col_b:
        return col_a, col_b
    if {"old_annotation", "new_annotation"}.issubset(df.columns):
        return "old_annotation", "new_annotation"
    text_cols = [c for c in df.columns
                 if df[c].dtype == object and not re.search(r'id$', c, re.I)]
    if len(text_cols) < 2:
        raise ValueError("Could not infer annotation columns.")
    return text_cols[0], text_cols[1]

def _row_key(row_dict, col_a, col_b):
    """Key used to decide ‘already processed?’"""
    return (row_dict.get("gene_id"), row_dict[col_a], row_dict[col_b])

# ---------- main entry points ----------------------------------------
def run_file(csv_path: P.Path, cfg_path: P.Path,
             col_a=None, col_b=None, *, out_path: P.Path | None) -> P.Path:

    cfg  = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))
    df   = pd.read_csv(csv_path)
    col_a, col_b = _infer_cols(df, col_a, col_b)

    out_file = out_path or csv_path.with_suffix(".concordia.csv")

    # ------------------------------------------------------------------
    # 1. Load already-completed keys (if any)
    done: set[tuple] = set()
    if out_file.exists():
        prev = pd.read_csv(out_file)
        done = {_row_key(r, col_a, col_b) for _, r in prev.iterrows()}

    # 2. Prepare LLM client & thresholds
    if mode in (Mode.LLM, Mode.HYBRID):
        llm = ArgoGatewayClient(**cfg["llm"])
    lo, hi = cfg["engine"]["hybrid_thresholds"].values()

    # 3. Open CSV writer (append or create with header)
    need_header = not out_file.exists()
    fout = out_file.open("a", newline="")
    writer = csv.DictWriter(fout, fieldnames=list(df.columns)+
                            ["similarity", "label", "note"])
    if need_header:
        writer.writeheader()

    # 4. Iterate rows
    for row in tqdm(df.to_dict(orient="records"), desc="Processing"):
        key = _row_key(row, col_a, col_b)
        if key in done:
            continue   # already processed in a previous run

        try:
            text1, text2 = row[col_a], row[col_b]
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

        except Exception as e:   # capture any failure but continue loop
            label = "Error"
            sim   = None
            note  = f"{type(e).__name__}: {e}"
            # optional: print stack trace for debugging
            traceback.print_exc()

        # write incremental result
        writer.writerow({**row,
                         "similarity": sim,
                         "label": label,
                         "note": note})
        fout.flush()  # ensure on-disk immediately

    fout.close()
    return out_file

# ---------- ad-hoc comparison ----------------------------------------
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