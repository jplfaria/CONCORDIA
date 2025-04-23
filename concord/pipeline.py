import pandas as pd, yaml, enum, pathlib as P
from tqdm import tqdm
from .local.embeddings import similarity
from .llm.argo_gateway import ArgoGatewayClient, llm_label

class Mode(str, enum.Enum):
    LOCAL = "local"
    LLM   = "llm"
    HYBRID = "hybrid"

def _local_rule(s: float) -> str:
    if s > .9:
        return "Identical"
    if s > .6:
        return "Partial"
    return "New"

# ------------------------------------------------------------------ #
def run_file(csv_path: P.Path, cfg_path: P.Path,
             out_path: P.Path | None = None) -> P.Path:
    cfg = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))
    df   = pd.read_csv(csv_path)

    if mode in (Mode.LLM, Mode.HYBRID):
        llm = ArgoGatewayClient(**cfg["llm"])

    lo, hi = cfg["engine"]["hybrid_thresholds"].values()
    out_rows = []

    itr = tqdm(df.itertuples(index=False), total=len(df), desc="Processing")

    for row in itr:
        sim = None
        if mode != Mode.LLM:
            sim = similarity(row.old_annotation, row.new_annotation)

        # decide label
        if mode == Mode.LLM:
            label = llm_label(row.old_annotation, row.new_annotation, llm)
        elif mode == Mode.LOCAL:
            label = _local_rule(sim)
        else:  # hybrid
            if sim < lo or sim > hi:
                label = _local_rule(sim)
            else:
                label = llm_label(row.old_annotation, row.new_annotation, llm)

        out_rows.append({"gene_id": row.gene_id,
                         "similarity": sim,
                         "label": label})

    out_file = out_path or csv_path.with_suffix(".concordia.csv")
    pd.DataFrame(out_rows).to_csv(out_file, index=False)
    return out_file

# ------------------------------------------------------------------ #
def run_pair(text_a: str, text_b: str, cfg_path: P.Path):
    cfg = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"].get("mode", "llm"))

    sim = None
    if mode != Mode.LLM:
        sim = similarity(text_a, text_b)

    if mode == Mode.LLM:
        client = ArgoGatewayClient(**cfg["llm"])
        label = llm_label(text_a, text_b, client)
    elif mode == Mode.LOCAL:
        label = _local_rule(sim)
    else:  # hybrid
        lo, hi = cfg["engine"]["hybrid_thresholds"].values()
        if sim < lo or sim > hi:
            label = _local_rule(sim)
        else:
            client = ArgoGatewayClient(**cfg["llm"])
            label = llm_label(text_a, text_b, client)

    return label, sim