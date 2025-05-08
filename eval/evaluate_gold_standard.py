#!/usr/bin/env python
"""
Evaluation script for CONCORDIA synthetic gold standard dataset.

This script calculates evaluation metrics on the synthetic gold standard
data to assess the performance of relationship classification.
"""

import argparse
import os
import pathlib as P
import sys

# Add parent directory to sys.path to import concord modules
sys.path.append(str(P.Path(__file__).parent.parent))

from concord.metrics import (
    evaluate_gold_standard,
    plot_confusion_matrix,
    print_evaluation_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CONCORDIA gold standard data"
    )
    parser.add_argument(
        "--gold-standard",
        type=str,
        default="./synthetic_gold_standard_v1.csv",
        help="Path to gold standard CSV file",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to predictions CSV file (if not provided, will use gold standard for perfect accuracy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--relationship-column",
        type=str,
        default="relationship",
        help="Column containing relationship labels",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate confusion matrix plot"
    )

    args = parser.parse_args()

    # Resolve gold standard and predictions paths relative to current working dir
    gold_standard_path = P.Path(args.gold_standard)
    if not gold_standard_path.is_absolute():
        gold_standard_path = gold_standard_path.resolve()

    if args.predictions:
        predictions_path = P.Path(args.predictions)
    else:
        # If no predictions provided, use gold standard for perfect accuracy
        predictions_path = gold_standard_path
    if not predictions_path.is_absolute():
        predictions_path = predictions_path.resolve()

    output_dir = P.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = P.Path(__file__).parent / output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    metrics_output_path = output_dir / "evaluation_metrics.json"
    plot_output_path = output_dir / "confusion_matrix.png" if args.plot else None

    # Run evaluation
    print(f"Evaluating with gold standard: {gold_standard_path}")
    print(f"Using predictions from: {predictions_path}")

    metrics_dict = evaluate_gold_standard(
        predictions_path=predictions_path,
        gold_standard_path=gold_standard_path,
        relationship_column=args.relationship_column,
        output_path=metrics_output_path,
    )

    # Print summary
    print_evaluation_summary(metrics_dict)

    # Generate plot if requested
    if args.plot:
        plot_confusion_matrix(metrics_dict, plot_output_path)
        print(f"Confusion matrix plot saved to: {plot_output_path}")

    print(f"Metrics saved to: {metrics_output_path}")


if __name__ == "__main__":
    main()
