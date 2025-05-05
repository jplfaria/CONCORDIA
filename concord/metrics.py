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
