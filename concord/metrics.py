"""
concord.metrics
==============
Optional performance tracking and metrics for the CONCORDIA engine.

This module provides opt-in metrics collection to avoid overhead when not needed.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import pathlib as P
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
R = TypeVar("R")

# Global control for metrics collection
_METRICS_ENABLED = os.getenv("CONCORDIA_METRICS", "false").lower() in (
    "true",
    "1",
    "yes",
)


@dataclass
class MetricCollector:
    """Optional metrics collector with minimal overhead when disabled."""

    enabled: bool = field(default_factory=lambda: _METRICS_ENABLED)
    timings: Dict[str, List[float]] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
    _start_times: Dict[str, float] = field(default_factory=dict)

    def start_timer(self, name: str) -> None:
        """Start a timer (no-op if disabled)."""
        if self.enabled:
            self._start_times[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop timer and return elapsed time (0.0 if disabled)."""
        if not self.enabled:
            return 0.0

        if name not in self._start_times:
            return 0.0

        elapsed = time.time() - self._start_times[name]
        del self._start_times[name]

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
        return elapsed

    def increment_counter(self, name: str, amount: int = 1) -> int:
        """Increment counter (no-op if disabled)."""
        if not self.enabled:
            return 0

        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += amount
        return self.counters[name]

    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge value (no-op if disabled)."""
        if self.enabled:
            self.gauges[name] = value

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary (empty if disabled)."""
        if not self.enabled:
            return {"enabled": False}

        summary = {
            "enabled": True,
            "timings": {},
            "counters": self.counters,
            "gauges": self.gauges,
        }

        # Calculate timing statistics
        for name, values in self.timings.items():
            if not values:
                continue

            stats = {
                "count": len(values),
                "total": sum(values),
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
            }

            if len(values) > 1:
                stats["std_dev"] = statistics.stdev(values)

            summary["timings"][name] = stats

        return summary

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log metrics summary (no-op if disabled)."""
        if not self.enabled:
            return

        summary = self.get_summary()

        logger.log(level, "Performance Metrics Summary:")

        # Log timing stats
        if summary["timings"]:
            logger.log(level, "Timings:")
            for name, stats in summary["timings"].items():
                logger.log(
                    level,
                    f"  {name}: avg={stats['mean']:.4f}s, count={stats['count']}, "
                    f"min={stats['min']:.4f}s, max={stats['max']:.4f}s",
                )

        # Log counters
        if summary["counters"]:
            logger.log(level, "Counters:")
            for name, value in summary["counters"].items():
                logger.log(level, f"  {name}: {value}")

        # Log gauges
        if summary["gauges"]:
            logger.log(level, "Gauges:")
            for name, value in summary["gauges"].items():
                logger.log(level, f"  {name}: {value}")

    def save_to_file(self, path: Union[str, P.Path]) -> None:
        """Save metrics to file (no-op if disabled)."""
        if not self.enabled:
            logger.debug("Metrics disabled, not saving to file")
            return

        summary = self.get_summary()
        summary["timestamp"] = datetime.now().isoformat()

        try:
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Metrics saved to {path}")
        except IOError as e:
            logger.error(f"Failed to save metrics to {path}: {e}")

    def reset(self) -> None:
        """Reset all metrics."""
        if self.enabled:
            self.timings.clear()
            self.counters.clear()
            self.gauges.clear()
            self._start_times.clear()


# Lazy initialization of global metrics collector
_metrics: Optional[MetricCollector] = None


def get_metrics() -> MetricCollector:
    """Get global metrics collector with lazy initialization."""
    global _metrics
    if _metrics is None:
        _metrics = MetricCollector()
    return _metrics


def enable_metrics() -> None:
    """Enable metrics collection globally."""
    global _METRICS_ENABLED
    _METRICS_ENABLED = True
    collector = get_metrics()
    collector.enabled = True
    logger.info("Metrics collection enabled")


def disable_metrics() -> None:
    """Disable metrics collection globally."""
    global _METRICS_ENABLED
    _METRICS_ENABLED = False
    collector = get_metrics()
    collector.enabled = False
    logger.debug("Metrics collection disabled")


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled."""
    return _METRICS_ENABLED


# Provide access to the metrics collector
metrics = get_metrics()


def timed(name: Optional[str] = None) -> Callable:
    """
    Decorator to time function execution (no-op if metrics disabled).

    Args:
        name: Timer name (defaults to function name)

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        timer_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            if not _METRICS_ENABLED:
                return func(*args, **kwargs)

            collector = get_metrics()
            collector.start_timer(timer_name)
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = collector.stop_timer(timer_name)
                if elapsed > 0:
                    logger.debug(f"{timer_name} completed in {elapsed:.4f}s")

        return wrapper

    # Handle case where decorator is used without parentheses
    if callable(name):
        func, name = name, None
        return decorator(func)

    return decorator


def evaluate_gold_standard(
    predictions_path: Union[str, P.Path],
    gold_standard_path: Union[str, P.Path],
    relationship_column: str = "relationship",
    pred_column1: str = "original_annotation",
    pred_column2: str = "new_annotation",
    output_path: Optional[Union[str, P.Path]] = None,
) -> Dict[str, Any]:
    """
    Evaluate predictions against a gold standard dataset.

    Args:
        predictions_path: Path to the predictions CSV
        gold_standard_path: Path to the gold standard CSV
        relationship_column: Column containing relationship labels
        pred_column1: First column used in predictions
        pred_column2: Second column used in predictions
        output_path: Optional path to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating predictions against gold standard: {gold_standard_path}")

    # Load data
    try:
        gold_df = pd.read_csv(gold_standard_path)
        pred_df = pd.read_csv(predictions_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Fallback: if predictions use 'label' instead of the expected relationship_column
    if relationship_column not in pred_df.columns and "label" in pred_df.columns:
        logger.warning(
            f"Prediction file missing '{relationship_column}', using 'label' column instead."
        )
        pred_df[relationship_column] = pred_df["label"]

    # Extract true and predicted labels
    y_true_raw = gold_df[relationship_column].astype(str).values
    y_pred_raw = pred_df[relationship_column].astype(str).values

    # Remap "Related" to "Different" for both true and predicted labels
    y_true = np.array(
        [("Different" if label == "Related" else label) for label in y_true_raw]
    )
    y_pred = np.array(
        [("Different" if label == "Related" else label) for label in y_pred_raw]
    )

    # Ensure prediction dataframe has the same structure
    required_columns = [pred_column1, pred_column2, relationship_column]
    if not all(col in pred_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in pred_df.columns]
        raise ValueError(f"Prediction file missing required columns: {missing}")

    # Calculate metrics
    # Use the updated LABEL_SET from prompts for class labels in metrics
    from .llm.prompts import LABEL_SET

    class_labels = sorted(list(LABEL_SET))

    metrics_dict = calculate_classification_metrics(y_true, y_pred, labels=class_labels)

    # Save results if requested
    if output_path:
        try:
            # Ensure output directory exists
            P.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            logger.info(f"Evaluation metrics saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save evaluation metrics to {output_path}: {e}")

    return metrics_dict


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate various classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of labels to include in the report.
                If None, labels are inferred from data.

    Returns:
        Dictionary with classification metrics
    """
    # Ensure labels for metrics are derived from the updated LABEL_SET if not provided explicitly
    if labels is None:
        from .llm.prompts import LABEL_SET

        labels = sorted(list(LABEL_SET))

    # Remap "Related" to "Different" in y_true and y_pred again just in case they weren't already.
    # This ensures consistency if this function is called directly.
    y_true_mapped = np.array(
        [("Different" if str(label) == "Related" else str(label)) for label in y_true]
    )
    y_pred_mapped = np.array(
        [("Different" if str(label) == "Related" else str(label)) for label in y_pred]
    )

    # Get unique labels present in the data after mapping
    unique_labels = sorted(list(set(y_true_mapped) | set(y_pred_mapped)))

    # Ensure the `labels` parameter for sklearn metrics uses only those present in data
    # or the explicitly passed `labels` list if it's more restrictive.
    # This avoids errors with labels not present in y_true/y_pred.
    if labels:
        # Filter labels to only those present in the data or in the global LABEL_SET
        # This is to ensure consistency and avoid errors if a label was expected but not found
        # It's important that `labels` here aligns with the classes the confusion matrix expects
        current_label_set = set(unique_labels)
        final_labels = [label for label in labels if label in current_label_set]
        if (
            not final_labels
        ):  # if all expected labels are missing from data, use unique_labels from data
            final_labels = unique_labels
    else:
        final_labels = unique_labels

    accuracy = accuracy_score(y_true_mapped, y_pred_mapped)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_mapped, y_pred_mapped, average=None, labels=final_labels, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_mapped,
        y_pred_mapped,
        average="macro",
        labels=final_labels,
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true_mapped,
            y_pred_mapped,
            average="weighted",
            labels=final_labels,
            zero_division=0,
        )
    )
    mcc = matthews_corrcoef(y_true_mapped, y_pred_mapped)

    # Confusion matrix
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=final_labels)

    return {
        "accuracy": accuracy,
        "precision_per_class": dict(zip(final_labels, precision.tolist())),
        "recall_per_class": dict(zip(final_labels, recall.tolist())),
        "f1_per_class": dict(zip(final_labels, f1.tolist())),
        "support_per_class": dict(zip(final_labels, support.tolist())),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "matthews_corrcoef": mcc,
        "confusion_matrix": cm.tolist(),
        "labels": final_labels,  # Ensure labels used for CM are returned
    }


def print_evaluation_summary(metrics_dict: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of evaluation metrics.

    Args:
        metrics_dict: Dictionary with evaluation metrics
    """
    print("\n===== EVALUATION METRICS =====")
    print(f"Overall Accuracy: {metrics_dict['overall']['accuracy']:.4f}")
    print(f"Weighted F1-Score: {metrics_dict['overall']['weighted_f1']:.4f}")
    print(
        f"Matthews Correlation Coefficient (MCC): {metrics_dict['overall']['mcc']:.4f}"
    )

    print("\nPer-Class Metrics:")
    for cls, metrics in metrics_dict["per_class"].items():
        print(f"  {cls}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1']:.4f}")
        print(f"    Support: {metrics['support']}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    classes = metrics_dict["confusion_matrix"]["classes"]
    matrix = metrics_dict["confusion_matrix"]["matrix"]

    # Format confusion matrix for display
    cm_df = pd.DataFrame(matrix, index=classes, columns=classes)
    print(cm_df)


def plot_confusion_matrix(
    metrics_dict: Dict[str, Any], output_path: Optional[Union[str, P.Path]] = None
) -> None:
    """
    Plot confusion matrix and save to file.

    Args:
        metrics_dict: Dictionary with evaluation metrics
        output_path: Path to save the plot (if None, will display instead)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        classes = metrics_dict["confusion_matrix"]["classes"]
        matrix = metrics_dict["confusion_matrix"]["matrix"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
            logger.info(f"Confusion matrix plot saved to {output_path}")
        else:
            plt.show()
    except ImportError:
        logger.warning(
            "Matplotlib and/or seaborn not available. Cannot plot confusion matrix."
        )
