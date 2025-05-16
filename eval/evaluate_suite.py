#!/usr/bin/env python
"""Evaluate multiple CONCORDIA prediction CSVs against a gold-standard.

This helper aggregates metrics across *many* prediction files so that you can
quickly identify the best-performing configuration.

Example
-------
$ python eval/evaluate_suite.py \
    --gold eval/Benchmark_subset__200_pairs_v1.csv \
    --pred-dir example_data/test_results \
    --pattern "gpt4o_*.csv" --plot

Outputs
-------
• <out>/summary_metrics.csv   – accuracy, macro- & weighted-F1 per run
• <out>/confusion_<stem>.png  – confusion matrix per run (if --plot)
• <out>/f1_comparison.png     – bar chart of macro-F1 across runs (if --plot)
"""
from __future__ import annotations

import argparse
import pathlib as P
from typing import List

import numpy as np
import pandas as pd
from rich import print as rprint

from concord.metrics import calculate_classification_metrics, plot_confusion_matrix


def gather_predictions(pred_dir: P.Path, pattern: str) -> List[P.Path]:
    """Return list of CSV files matching *pattern* inside *pred_dir*."""
    return sorted(pred_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk evaluation of CONCORDIA runs")
    parser.add_argument("--gold", required=True, help="Gold-standard CSV with labels")
    parser.add_argument(
        "--pred-dir", required=True, help="Directory containing prediction CSVs"
    )
    parser.add_argument(
        "--pattern", default="*.csv", help="Glob pattern to match files"
    )
    parser.add_argument(
        "--out", default="eval/benchmark_results", help="Output directory"
    )
    parser.add_argument(
        "--rel-col", default="relationship_label", help="Label column name"
    )
    parser.add_argument(
        "--pred-col",
        default="relationship",
        help="Label column name in prediction files",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Make confusion-matrix and bar plots"
    )
    args = parser.parse_args()

    gold_path = P.Path(args.gold).expanduser().resolve()
    pred_dir = P.Path(args.pred_dir).expanduser().resolve()
    out_dir = P.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = gather_predictions(pred_dir, args.pattern)
    if not csv_paths:
        rprint(
            f"[red]No prediction CSVs match pattern '{args.pattern}' in {pred_dir}.[/red]"
        )
        return

    records = []
    for csv_path in csv_paths:
        try:
            gold_df = pd.read_csv(gold_path)
            pred_df = pd.read_csv(csv_path)

            # Check if prediction file has the expected column
            if args.pred_col not in pred_df.columns:
                rprint(
                    f"[yellow]Warning: '{csv_path.name}' missing '{args.pred_col}' column.[/yellow]"
                )
                continue

            # Extract true and predicted labels
            y_true = gold_df[args.rel_col].values
            y_pred = pred_df[args.pred_col].values

            # Calculate metrics
            metrics = calculate_classification_metrics(y_true, y_pred)

            # Add columns info so confusion matrix plotting works
            metrics["confusion_matrix"]["classes"] = sorted(
                set(np.concatenate([y_true, y_pred]))
            )

            # Extract metrics
            per_class = metrics["per_class"]
            macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
            weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / sum(
                v["support"] for v in per_class.values()
            )
            acc = metrics["overall"].get("accuracy", 0.0)

            records.append(
                {
                    "file": csv_path.name,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                }
            )

            # Optional confusion-matrix plot
            if args.plot:
                cm_path = out_dir / f"confusion_{csv_path.stem}.png"
                plot_confusion_matrix(metrics, cm_path)

        except Exception as e:
            rprint(f"[red]Error processing {csv_path.name}: {e}[/red]")

    # ───────────── summary table ───────────────────────────────
    df = pd.DataFrame(records).sort_values("macro_f1", ascending=False)
    summary_path = out_dir / "summary_metrics.csv"
    df.to_csv(summary_path, index=False)

    rprint("[bold green]Evaluation complete![/bold green]")
    rprint(df)
    rprint(f"[cyan]Summary metrics saved to {summary_path}[/cyan]")

    # ───────────── optional bar plot ───────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(8, max(4, 0.3 * len(df))))
            sns.barplot(data=df, y="file", x="macro_f1", palette="crest")
            plt.title("Macro-F1 for each configuration")
            plt.tight_layout()
            bar_path = out_dir / "f1_comparison.png"
            plt.savefig(bar_path, dpi=150)
            rprint(f"[cyan]Bar plot saved to {bar_path}[/cyan]")
        except ImportError:
            rprint(
                "[yellow]matplotlib/seaborn not installed – skipping bar plot.[/yellow]"
            )


if __name__ == "__main__":
    main()
