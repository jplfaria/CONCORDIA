from .local.embeddings import similarity
from .llm.argo_gateway import ArgoGatewayClient, llm_label
import pandas as pd, enum, yaml

class Mode(enum.Enum): LOCAL="local"; LLM="llm"; HYBRID="hybrid"
def run(csv_path, cfg_path="concord/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    mode = Mode(cfg["engine"]["mode"])
    df   = pd.read_csv(csv_path)
    if mode in (Mode.LLM, Mode.HYBRID): llm = ArgoGatewayClient(**cfg["llm"])
    out=[]
    lo,hi = cfg["engine"]["hybrid_thresholds"].values()
    for _,r in df.iterrows():
        s = similarity(r.old_annotation,r.new_annotation)
        if mode==Mode.LOCAL or (mode==Mode.HYBRID and (s<lo or s>hi)):
            lbl = "Synonym" if s>.9 else "Partial" if s>.6 else "New"
        else:
            lbl = llm_label(r.old_annotation,r.new_annotation,llm)
        out.append({"gene_id":r.gene_id,"similarity":s,"label":lbl})
    pd.DataFrame(out).to_csv(csv_path.with_suffix(".concordia.csv"),index=False)
