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
• <out>/all_metrics_comparison.png     – combined bar chart of key metrics across runs (if --plot)
• <out>/misclassifications_<stem>.csv – list of misclassified instances per run
• <out>/summary_metrics.csv now includes MCC (Matthews Correlation Coefficient)
• <out>/per_class_f1_heatmap.png – heatmap of F1 scores per class across all runs (if --plot)
• <out>/evaluation_report.html – comprehensive HTML report summarizing all metrics and plots
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
    """Return list of CSV files matching *pattern* inside *pred_dir*, excluding 'evaluation_output'."""
    all_files = pred_dir.glob(
        pattern
    )  # This might be recursive if pattern contains '**'
    # Filter out files that are within an 'evaluation_output' directory
    # This check is robust even if 'pattern' itself causes recursive search (e.g. '**/*.csv')
    # and 'evaluation_output' is a subdirectory of 'pred_dir'.
    filtered_files = [
        f
        for f in all_files
        if "evaluation_output" not in [p.name for p in f.relative_to(pred_dir).parents]
    ]
    return sorted(filtered_files)


def plot_combined_metrics_summary(df_summary: pd.DataFrame, output_dir: P.Path):
    """Generates a single plot with subplots for key summary metrics."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics_to_plot = [
        {"col": "accuracy", "title": "Accuracy"},
        {"col": "macro_f1", "title": "Macro F1"},
        {"col": "weighted_f1", "title": "Weighted F1"},
        {"col": "macro_precision", "title": "Macro Precision"},
        {"col": "weighted_precision", "title": "Weighted Precision"},
        {"col": "macro_recall", "title": "Macro Recall"},
        {"col": "weighted_recall", "title": "Weighted Recall"},
        {"col": "mcc", "title": "Matthews Corr. Coeff."},
    ]

    num_metrics = len(metrics_to_plot)
    # Adjust subplot layout as needed, e.g., 3x3 for up to 9 metrics
    # For 7 metrics, a 3x3 or 4x2 layout works. Let's use 4 rows, 2 cols for now.
    # This gives a bit more horizontal space for y-axis labels (file names)
    nrows = 4
    ncols = 2
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, max(8, 1 * num_metrics))
    )
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    # Sort DataFrame by macro_f1 for consistent y-axis ordering in plots
    df_sorted = df_summary.sort_values("macro_f1", ascending=False)

    for i, metric_info in enumerate(metrics_to_plot):
        if i < len(axes):
            ax = axes[i]
            df_summary_melted = df_sorted.melt(
                id_vars="file", value_vars=[metric_info["col"]]
            )
            sns.barplot(
                data=df_summary_melted[
                    df_summary_melted["variable"] == metric_info["col"]
                ],
                x="value",
                y="file",
                hue="file",  # Assign y to hue as per FutureWarning
                palette="viridis",
                legend=False,  # Set legend to False as per FutureWarning
                ax=ax,  # Ensure plotting on the correct subplot axis
            )
            ax.set_title(metric_info["title"])
            ax.set_xlabel(metric_info["col"].replace("_", " ").title())
            ax.set_ylabel(
                ""
            )  # Remove y-label for cleaner subplots, main y-info is file names
        else:
            # Handle case where there are more metrics than subplots (should not happen with current setup)
            rprint(
                f"[yellow]Warning: Not enough subplots for metric {metric_info['title']}[/yellow]"
            )

    # Hide any unused subplots if num_metrics < nrows * ncols
    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    plot_path = output_dir / "all_metrics_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)  # Close the figure to free memory
    rprint(f"[cyan]Combined metrics plot saved to {plot_path}[/cyan]")


def plot_per_class_f1_heatmap(df_pivot: pd.DataFrame, output_path: P.Path):
    """Generates and saves a heatmap of per-class F1 scores."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df_pivot.empty:
        rprint("[yellow]Pivot table is empty, skipping heatmap generation.[/yellow]")
        return

    plt.figure(
        figsize=(
            max(10, len(df_pivot.columns) * 0.5),
            max(8, len(df_pivot.index) * 0.3),
        )
    )
    sns.heatmap(
        df_pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "F1 Score"},
    )
    plt.title("Per-Class F1 Score Heatmap")
    plt.ylabel("Class Label")
    plt.xlabel("Model Configuration (File)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=150)
    plt.close()
    rprint(f"[cyan]Per-class F1 heatmap saved to {output_path}[/cyan]")


# def image_to_base64(image_path: P.Path) -> str | None:
#     """Converts an image file to a base64 encoded string for HTML embedding."""
#     if not image_path or not image_path.exists():
#         return None
#     try:
#         with open(image_path, "rb") as img_file:
#             encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
#         return f"data:image/png;base64,{encoded_string}"
#     except Exception as e:
#         rprint(f"[yellow]Could not encode image {image_path}: {e}[/yellow]")
#         return None


def generate_html_report(
    summary_df: pd.DataFrame, run_details: List[dict], out_dir: P.Path
):
    """Generates a comprehensive HTML report of the evaluation results."""
    out_dir / "evaluation_report.html"

    # # Simplified HTML for debugging - COMMENTED OUT FOR DIAGNOSIS
    # html_content = """
    # <!DOCTYPE html>
    # <html lang="en">
    # <head>
    #     <meta charset="UTF-8">
    #     <title>Test Report</title>
    # </head>
    # <body>
    #     <h1>Test</h1>
    #     <p>This is a simplified report for debugging.</p>
    # </body>
    # </html>
    # """

    # with open(report_path, "w") as f: # COMMENTED OUT FOR DIAGNOSIS
    #     f.write("<html><body><h1>Debug</h1></body></html>") # Write minimal content
    # rprint(f"[cyan]HTML report (debug) saved to {report_path}[/cyan]")
    pass  # Ensure function body is not empty


def main() -> None:
    # Auto-detect latest benchmark run directory
    base_results_dir = P.Path("eval/results")
    latest_pred_dir = None
    # Default output dir if no benchmark runs are found or --pred-dir is specified manually without a corresponding run structure
    default_out_dir = base_results_dir / "evaluation_output_default"

    if base_results_dir.exists():
        benchmark_dirs = sorted(
            [
                d
                for d in base_results_dir.iterdir()
                if d.is_dir() and d.name.startswith("benchmark_run_")
            ]
        )
        if benchmark_dirs:
            latest_pred_dir = benchmark_dirs[-1]
            default_out_dir = latest_pred_dir / "evaluation_output"

    parser = argparse.ArgumentParser(description="Bulk evaluation of CONCORDIA runs")
    parser.add_argument("--gold", required=True, help="Gold-standard CSV with labels")
    parser.add_argument(
        "--pred-dir",
        default=str(latest_pred_dir) if latest_pred_dir else None,
        help="Directory containing prediction CSVs. Defaults to the latest 'benchmark_run_*' directory in 'eval/results/'.",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.csv",
        help="Glob pattern to find prediction CSVs within --pred-dir (default: '**/*.csv')",
    )
    parser.add_argument(
        "--pred-s1-col",
        default="annotation_a",
        help="Column name for sentence 1 in prediction CSVs (default: annotation_a)",
    )
    parser.add_argument(
        "--pred-s2-col",
        default="annotation_b",
        help="Column name for sentence 2 in prediction CSVs (default: annotation_b)",
    )
    parser.add_argument(
        "--pred-rel-col",
        default="relationship",
        help="Column name for the predicted relationship in prediction CSVs (default: relationship)",
    )
    parser.add_argument(
        "--out",
        default=str(default_out_dir),
        help="Output directory for evaluation results (default: <latest_pred_dir>/evaluation_output/ or eval/results/evaluation_output_default/)",
    )
    parser.add_argument(
        "--gold-s1-col",
        default="annotation_a",
        help="Column name for sentence 1 in gold CSV (default: annotation_a)",
    )
    parser.add_argument(
        "--gold-s2-col",
        default="annotation_b",
        help="Column name for sentence 2 in gold CSV (default: annotation_b)",
    )
    parser.add_argument(
        "--gold-rel-col",
        default="relationship_label",
        help="Column name for relationship label in gold CSV (default: relationship_label)",
    )

    parser.add_argument(
        "--plot", action="store_true", help="Make confusion-matrix and bar plots"
    )
    args = parser.parse_args()

    if not args.pred_dir:
        rprint(
            f"[red]Error: Could not auto-detect a benchmark run directory in '{str(base_results_dir)}', and --pred-dir was not specified.[/red]"
        )
        rprint(
            "[red]Please ensure there is at least one 'benchmark_run_*' directory or provide --pred-dir explicitly.[/red]"
        )
        return  # Or import sys; sys.exit(1)

    gold_path = P.Path(args.gold).expanduser().resolve()
    pred_dir = P.Path(args.pred_dir).expanduser().resolve()
    out_dir = P.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = gather_predictions(pred_dir, args.pattern)
    if not csv_paths:
        rprint(
            f"[red]No prediction CSVs match pattern '{args.pattern}' in {pred_dir}.[/red]"
        )
    records = []
    per_class_f1_data_list = []
    run_details_for_report = []
    for csv_path in csv_paths:
        try:
            # Read gold standard (once per run, could be moved outside loop if memory allows for large gold files)
            df_gold = pd.read_csv(
                gold_path,
                usecols=[args.gold_s1_col, args.gold_s2_col, args.gold_rel_col],
            )

            # Read prediction file, selecting only specified columns
            pred_df = pd.read_csv(
                csv_path,
                usecols=[args.pred_s1_col, args.pred_s2_col, args.pred_rel_col],
            )

            # Verify necessary columns are in the dataframes after selection
            if not all(
                col in df_gold.columns
                for col in [args.gold_s1_col, args.gold_s2_col, args.gold_rel_col]
            ):
                rprint(
                    f"[red]Error: Gold standard file '{gold_path}' is missing one or more required columns: {args.gold_s1_col}, {args.gold_s2_col}, {args.gold_rel_col}. Check --gold-*col arguments.[/red]"
                )
                return  # Exit if gold standard is misconfigured

            if not all(
                col in pred_df.columns
                for col in [args.pred_s1_col, args.pred_s2_col, args.pred_rel_col]
            ):
                rprint(
                    f"[yellow]Warning: Prediction file '{csv_path.name}' is missing one or more columns: {args.pred_s1_col}, {args.pred_s2_col}, {args.pred_rel_col}. Skipping.[/yellow]"
                )
                continue

            # Align pred_df with gold_df to ensure y_true and y_pred are comparable
            # Using an inner merge, so only pairs present in both are evaluated.
            merged_df = pd.merge(
                pred_df,  # Left DataFrame
                df_gold,  # Right DataFrame
                left_on=[
                    args.pred_s1_col,
                    args.pred_s2_col,
                ],  # Keys from left DataFrame (pred_df)
                right_on=[
                    args.gold_s1_col,
                    args.gold_s2_col,
                ],  # Keys from right DataFrame (df_gold)
                how="inner",
                suffixes=[
                    "_pred",
                    "_gold",
                ],  # Note: suffix order changed to match df order
            )

            if merged_df.empty:
                rprint(
                    f"[yellow]Skipping {csv_path.name}: No common sentence pairs with gold standard after merge.[/yellow]"
                )
                continue

            # Determine the correct column names after merge
            # The relationship label from gold will have _gold suffix
            # Access relationship columns directly by their names as defined in args, because they don't clash by default.
            # If they were to clash (e.g., both gold and pred files used 'relationship' as the column name),
            # then suffixes '_pred' and '_gold' would be applied by the merge.
            # The 'relationship' column from pred_df and 'relationship_label' from df_gold do not clash by default.
            y_true = merged_df[args.gold_rel_col].values
            y_pred = merged_df[args.pred_rel_col].values

            # Calculate metrics
            metrics = calculate_classification_metrics(y_true, y_pred)

            # Add columns info so confusion matrix plotting works
            metrics["confusion_matrix"]["classes"] = sorted(
                set(np.concatenate([y_true, y_pred]))
            )

            # Extract metrics
            per_class_metrics = metrics["per_class"]
            num_classes = len(per_class_metrics)
            total_support = sum(v["support"] for v in per_class_metrics.values())

            macro_f1 = (
                sum(v["f1"] for v in per_class_metrics.values()) / num_classes
                if num_classes > 0
                else 0
            )
            weighted_f1 = (
                sum(v["f1"] * v["support"] for v in per_class_metrics.values())
                / total_support
                if total_support > 0
                else 0
            )

            macro_precision = (
                sum(v["precision"] for v in per_class_metrics.values()) / num_classes
                if num_classes > 0
                else 0
            )
            weighted_precision = (
                sum(v["precision"] * v["support"] for v in per_class_metrics.values())
                / total_support
                if total_support > 0
                else 0
            )

            macro_recall = (
                sum(v["recall"] for v in per_class_metrics.values()) / num_classes
                if num_classes > 0
                else 0
            )
            weighted_recall = (
                sum(v["recall"] * v["support"] for v in per_class_metrics.values())
                / total_support
                if total_support > 0
                else 0
            )

            acc = metrics["overall"].get("accuracy", 0.0)
            mcc = metrics["overall"].get("mcc", 0.0)  # Extract MCC

            records.append(
                {
                    "file": csv_path.stem,  # Use stem for report consistency
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "macro_precision": macro_precision,
                    "weighted_precision": weighted_precision,
                    "macro_recall": macro_recall,
                    "weighted_recall": weighted_recall,
                    "mcc": mcc,  # Add MCC to records
                }
            )

            # --- Collect Per-Class F1 for Heatmap ---
            for class_label, class_metrics in per_class_metrics.items():
                per_class_f1_data_list.append(
                    {
                        "file": csv_path.stem,  # Use stem for cleaner column names in heatmap
                        "class_label": str(class_label),
                        "f1_score": class_metrics.get("f1", 0.0),
                    }
                )
            # ------------------------------------------

            # Optional confusion-matrix plot
            cm_plot_path = None
            if args.plot:
                cm_plot_path = out_dir / f"confusion_{csv_path.stem}.png"
                plot_confusion_matrix(metrics, cm_plot_path)

            # --- Generate Misclassification Report ---
            misclass_file_path = None  # Initialize
            misclassified_mask = y_true != y_pred
            if np.any(misclassified_mask):
                # Select columns for misclassification report, using original column names from gold and pred with suffixes
                # The sentence columns in merged_df will be args.pred_s1_col, args.pred_s2_col (from pred_df, no suffix if not clashing)
                # or args.gold_s1_col_gold, args.gold_s2_col_gold if they were different from pred_df's sentence columns.
                # Given suffixes=["_pred", "_gold"], and pred_df is left:
                # - pred sentence cols: args.pred_s1_col, args.pred_s2_col (if no clash)
                #   or args.pred_s1_col_pred, args.pred_s2_col_pred (if clashing with df_gold's original names)
                # - gold sentence cols: args.gold_s1_col_gold, args.gold_s2_col_gold
                # Let's assume for simplicity that sentence column names in pred and gold are distinct after suffixing if they were originally the same.
                # The keys used for merging (args.pred_s1_col, etc.) will be present directly if they don't clash.
                # If they do clash, they get suffixes. Since we use suffixes=["_pred", "_gold"], and pred_df is left:
                # - Columns unique to pred_df or used as left_on key: original name (e.g., args.pred_s1_col)
                # - Columns unique to df_gold or used as right_on key: original name + "_gold" (e.g., args.gold_s1_col + "_gold")
                # - Columns present in both AND NOT merge keys: original_name_pred, original_name_gold

                # For sentence columns in misclassification report, these are the merge keys.
                # If pred_s1_col and gold_s1_col are the same (e.g. 'annotation_a' by default),
                # pandas keeps the original name from the left df ('annotation_a') rather than applying suffixes like '_pred'.
                s1_col_for_report = args.pred_s1_col  # e.g., 'annotation_a'
                s2_col_for_report = args.pred_s2_col  # e.g., 'annotation_b'

            misclassified_df = merged_df[
                y_true != y_pred
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning
            misclassified_df = misclassified_df[
                [
                    s1_col_for_report,
                    s2_col_for_report,
                    args.gold_rel_col,  # True label column (e.g., 'relationship_label')
                    args.pred_rel_col,  # Predicted label column (e.g., 'relationship')
                ]
            ].rename(
                columns={
                    s1_col_for_report: "sentence1",
                    s2_col_for_report: "sentence2",
                    args.gold_rel_col: "gold_relationship_label",
                    args.pred_rel_col: "pred_relationship_label",
                }
            )

            misclass_file_path = out_dir / f"misclassifications_{csv_path.stem}.csv"
            misclassified_df.to_csv(misclass_file_path, index=False)
            rprint(
                f"[cyan]Misclassifications report saved to {misclass_file_path}[/cyan]"
            )
            # ------------------------------------------

            # --- Collect details for HTML report ---
            current_run_details = records[
                -1
            ].copy()  # Get the latest record (metrics for this run)
            current_run_details["confusion_matrix_path"] = (
                str(cm_plot_path) if cm_plot_path and cm_plot_path.exists() else None
            )
            current_run_details["misclassifications_path"] = (
                str(misclass_file_path)
                if misclass_file_path and misclass_file_path.exists()
                else None
            )
            run_details_for_report.append(current_run_details)
            # -------------------------------------

        except Exception as e:
            rprint(f"[red]Error processing {csv_path.name}: {e}[/red]")

    # --- Generate summary metrics & plots ---
    if not records:
        rprint(
            "[yellow]No valid prediction files were processed. Cannot generate summary metrics or plots.[/yellow]"
        )
        rprint(
            f"[yellow]Please ensure --pred-dir '{str(pred_dir)}' contains valid prediction CSVs matching pattern '{args.pattern}'.[/yellow]"
        )
        return  # Exit gracefully

    rprint("\n[bold green]Evaluation complete![/bold green]")
    df = pd.DataFrame(records).sort_values("macro_f1", ascending=False)
    summary_path = out_dir / "summary_metrics.csv"
    df.to_csv(summary_path, index=False)

    rprint("[bold green]Evaluation complete![/bold green]")
    rprint(df)
    rprint(f"[cyan]Summary metrics saved to {summary_path}[/cyan]")

    # ───────────── optional bar plot ───────────────────────────
    if args.plot:
        try:
            plot_combined_metrics_summary(df, out_dir)

            # --- Generate Per-Class F1 Heatmap ---
            if per_class_f1_data_list:
                df_per_class = pd.DataFrame(per_class_f1_data_list)
                # Get all unique classes and files to ensure consistent heatmap dimensions
                all_classes = sorted(df_per_class["class_label"].unique())
                all_files = sorted(
                    df_per_class["file"].unique()
                )  # Files sorted as they were processed

                heatmap_pivot_df = df_per_class.pivot_table(
                    index="class_label",
                    columns="file",
                    values="f1_score",
                    fill_value=0,  # Fill missing class/file F1s with 0
                )
                # Reindex to ensure all classes and files are present and in sorted order
                heatmap_pivot_df = heatmap_pivot_df.reindex(
                    index=all_classes, columns=all_files, fill_value=0
                )

                plot_per_class_f1_heatmap(
                    heatmap_pivot_df, out_dir / "per_class_f1_heatmap.png"
                )
            else:
                rprint(
                    "[yellow]No per-class F1 data collected; skipping heatmap.[/yellow]"
                )
            # -------------------------------------

            # --- Generate HTML Report ---
            generate_html_report(df, run_details_for_report, out_dir)
            # --------------------------

        except ImportError:
            rprint(
                "[yellow]matplotlib/seaborn not installed – skipping bar plots.[/yellow]"
            )
        except Exception as e:
            rprint(f"[red]Error during plotting: {e}[/red]")


if __name__ == "__main__":
    main()
