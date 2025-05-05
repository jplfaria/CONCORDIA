"""
concord.io.loader
-----------------
Load tabular data in CSV, TSV, or JSON format for the pipeline.
"""

from __future__ import annotations

import pathlib as P

import pandas as pd


def load_table(path: P.Path, *, sep: str | None = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    path : pathlib.Path
        Input file ( .csv / .tsv / .json )
    sep  : str | None
        Delimiter override for text files.  If None, infer from extension.

    Returns
    -------
    pandas.DataFrame
    """
    ext = path.suffix.lower()

    if ext in {".csv", ".txt"}:
        return pd.read_csv(path, sep=sep or ",")
    if ext in {".tsv", ".tab"}:
        return pd.read_csv(path, sep=sep or "\t")
    if ext == ".json":
        return pd.read_json(path)  # list-of-objects *or* column-orient
    raise ValueError(
        f"Unsupported file type '{ext}'. " "Accepted: .csv .tsv .tab .json"
    )
