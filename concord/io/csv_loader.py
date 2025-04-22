import pandas as pd, pathlib as P
def load_changes(path: str | P.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"gene_id", "old_annotation", "new_annotation"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must have columns {req}")
    return df
