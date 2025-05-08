"""
concord.metrics
==============
Performance tracking and metrics for the CONCORDIA engine.

This module records and reports performance metrics to help identify
bottlenecks and optimization opportunities.
"""

from __future__ import annotations

import functools
import json
import logging
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
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
R = TypeVar("R")


@dataclass
class MetricCollector:
    """Collect and analyze performance metrics."""

    # Storage for metrics
    timings: Dict[str, List[float]] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)

    # Start times for ongoing operations
    _start_times: Dict[str, float] = field(default_factory=dict)

    def start_timer(self, name: str) -> None:
        """
        Start a timer for an operation.

        Args:
            name: Name of the timer
        """
        self._start_times[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and record the elapsed time.

        Args:
            name: Name of the timer

        Returns:
            Elapsed time in seconds

        Raises:
            ValueError: If timer wasn't started
        """
        if name not in self._start_times:
            raise ValueError(f"Timer '{name}' was not started")

        elapsed = time.time() - self._start_times[name]
        del self._start_times[name]

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

        return elapsed

    def increment_counter(self, name: str, amount: int = 1) -> int:
        """
        Increment a counter.

        Args:
            name: Counter name
            amount: Amount to increment by

        Returns:
            New counter value
        """
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += amount
        return self.counters[name]

    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Gauge value
        """
        self.gauges[name] = value

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.

        Returns:
            Dictionary with metric summaries
        """
        summary = {"timings": {}, "counters": self.counters, "gauges": self.gauges}

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
        """
        Log a summary of all metrics.

        Args:
            level: Logging level
        """
        summary = self.get_summary()

        logger.log(level, "Performance Metrics Summary:")

        # Log timing stats
        if summary["timings"]:
            logger.log(level, "Timings:")
            for name, stats in summary["timings"].items():
                logger.log(
                    level,
                    f"  {name}: avg={stats['mean']:.4f}s, count={stats['count']}, "
                    f"min={stats['min']:.4f}s, max={stats['max']:.4f}s, "
                    f"total={stats['total']:.4f}s",
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
        """
        Save metrics to a JSON file.

        Args:
            path: Path to output file
        """
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
        self.timings.clear()
        self.counters.clear()
        self.gauges.clear()
        self._start_times.clear()


# Global metrics collector instance
metrics = MetricCollector()


def timed(name: Optional[str] = None) -> Callable:
    """
    Decorator to time function execution and record in metrics.

    Args:
        name: Timer name (defaults to function name)

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        timer_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            metrics.start_timer(timer_name)
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = metrics.stop_timer(timer_name)
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
    y_true = gold_df[relationship_column].values

    # Ensure prediction dataframe has the same structure
    required_columns = [pred_column1, pred_column2, relationship_column]
    if not all(col in pred_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in pred_df.columns]
        raise ValueError(f"Prediction file missing required columns: {missing}")

    y_pred = pred_df[relationship_column].values

    # Calculate metrics
    metrics_dict = calculate_classification_metrics(y_true, y_pred)

    # Save results if requested
    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            logger.info(f"Evaluation metrics saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save evaluation metrics to {output_path}: {e}")

    return metrics_dict


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate classification metrics.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels

    Returns:
        Dictionary with evaluation metrics
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1 score for each category
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    # Calculate weighted F1-score
    weighted_f1 = np.sum(f1 * support) / np.sum(support) if np.sum(support) > 0 else 0

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Build results dictionary
    metrics_dict = {
        "overall": {"accuracy": float(accuracy), "weighted_f1": float(weighted_f1)},
        "per_class": {},
        "confusion_matrix": {"matrix": cm.tolist(), "classes": classes.tolist()},
    }

    # Add per-class metrics
    for i, cls in enumerate(classes):
        metrics_dict["per_class"][str(cls)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return metrics_dict


def print_evaluation_summary(metrics_dict: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of evaluation metrics.

    Args:
        metrics_dict: Dictionary with evaluation metrics
    """
    print("\n===== EVALUATION METRICS =====")
    print(f"Overall Accuracy: {metrics_dict['overall']['accuracy']:.4f}")
    print(f"Weighted F1-Score: {metrics_dict['overall']['weighted_f1']:.4f}")

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
