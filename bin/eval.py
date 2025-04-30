#!/usr/bin/env python3
"""
bin/eval.py
===========

Simple scorer for Concordia outputs.

Usage
-----
    python bin/eval.py GOLD.csv RUN.csv [--col LABEL_COL]

Assumptions
-----------
* GOLD.csv and RUN.csv share a key column (default: `gene_id`) **and**
  a label column (default: `label`).
* All other columns are ignored.
"""

from __future__ import annotations
import argparse, sys
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support as prfs,
    accuracy_score,
)

# ----------------------------------------------------------------------
def load(path: str, key: str, label: str) -> pd.Series:
    df = pd.read_csv(path, usecols=[key, label])
    if df[key].duplicated().any():
        dup = df[df[key].duplicated()][key].tolist()
        sys.exit(f"[error] duplicate keys in {path}: {dup[:5]} …")
    return df.set_index(key)[label]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gold", help="gold-standard CSV")
    p.add_argument("run",  help="system output CSV")
    p.add_argument("--key", "--id", default="gene_id",
                   help="column used to align rows (default: gene_id)")
    p.add_argument("--col", "--label", default="label",
                   help="column holding the predicted label (default: label)")
    args = p.parse_args()

    gold = load(args.gold, args.key, args.col)
    run  = load(args.run,  args.key, args.col)

    # ----- align & sanity ------------------------------------------------
    common = gold.index.intersection(run.index)
    if len(common) == 0:
        sys.exit("[error] no matching keys between gold and run!]")

    y_true = gold.loc[common]
    y_pred = run.loc[common].reindex(common)

    # ----- metrics -------------------------------------------------------
    labels = sorted(y_true.unique())
    P, R, F, _ = prfs(y_true, y_pred, labels=labels,
                      average=None, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print(f"Samples evaluated: {len(common)}")
    print(f"Micro-accuracy  : {acc:6.3f}")

    macroP, macroR, macroF, _ = prfs(
        y_true, y_pred, labels=labels,
        average="macro", zero_division=0)
    print(f"Macro P/R/F     : {macroP:6.3f}  {macroR:6.3f}  {macroF:6.3f}\n")

    print(" Per-class scores")
    print(" label        P      R      F   n")
    for lbl, p_, r_, f_ in zip(labels, P, R, F):
        n = (y_true == lbl).sum()
        print(f" {lbl:<10} {p_:6.3f} {r_:6.3f} {f_:6.3f}  {n:5d}")


if __name__ == "__main__":
    main()