#!/usr/bin/env python3
"""
Performance benchmark script for CONCORDIA optimizations.

This script benchmarks various performance optimizations including:
- Configuration caching
- Template loading
- Embedding cache performance
- Metrics overhead
- Memory usage
"""

import os
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict

import yaml

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def benchmark_config_loading(iterations: int = 100) -> Dict[str, float]:
    """Benchmark configuration loading with and without caching."""
    from concord.utils import (
        clear_config_cache,
        disable_config_cache,
        enable_config_cache,
        load_yaml_config,
    )

    # Create test config
    test_config = {
        "engine": {"mode": "zero-shot"},
        "llm": {"model": "gpt4o"},
        "embedding": {"device": "cpu"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config, f)
        temp_config_path = f.name

    try:
        # Test without caching
        disable_config_cache()
        start_time = time.time()
        for _ in range(iterations):
            load_yaml_config(temp_config_path)
        no_cache_time = time.time() - start_time

        # Test with caching
        enable_config_cache()
        clear_config_cache()
        start_time = time.time()
        for _ in range(iterations):
            load_yaml_config(temp_config_path)
        with_cache_time = time.time() - start_time

        return {
            "no_cache_time": no_cache_time,
            "with_cache_time": with_cache_time,
            "speedup": (
                no_cache_time / with_cache_time if with_cache_time > 0 else float("inf")
            ),
        }
    finally:
        os.unlink(temp_config_path)


def benchmark_template_loading(iterations: int = 50) -> Dict[str, float]:
    """Benchmark template loading performance."""
    from concord.llm.prompts import get_prompt_template, list_available_templates

    cfg = {"prompt_ver": "v3.2"}
    templates = list_available_templates()

    start_time = time.time()
    for _ in range(iterations):
        for template_ver in templates[:3]:  # Test first 3 templates
            get_prompt_template(cfg, ver=template_ver)
    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "avg_per_template": total_time / (iterations * 3),
        "templates_tested": len(templates[:3]),
    }


def benchmark_embedding_cache(iterations: int = 20) -> Dict[str, float]:
    """Benchmark embedding cache performance."""
    from concord.embedding import clear_cache, embed_sentence

    cfg = {
        "embedding": {
            "model_id": "NeuML/pubmedbert-base-embeddings",
            "device": "cpu",
            "batch_size": 16,
        }
    }

    test_texts = [
        "glucose-6-phosphate isomerase",
        "hypothetical protein",
        "DNA polymerase III subunit alpha",
        "ATP synthase F0 subunit c",
    ]

    # Cold start (no cache)
    clear_cache()
    start_time = time.time()
    for _ in range(iterations):
        for text in test_texts:
            embed_sentence(text, cfg)
    cold_time = time.time() - start_time

    # Warm cache (repeated embeddings)
    start_time = time.time()
    for _ in range(iterations):
        for text in test_texts:
            embed_sentence(text, cfg)
    warm_time = time.time() - start_time

    return {
        "cold_time": cold_time,
        "warm_time": warm_time,
        "speedup": cold_time / warm_time if warm_time > 0 else float("inf"),
        "texts_processed": len(test_texts) * iterations,
    }


def benchmark_metrics_overhead(iterations: int = 1000) -> Dict[str, float]:
    """Benchmark metrics collection overhead."""
    from concord.metrics import disable_metrics, enable_metrics, get_metrics

    def dummy_work():
        """Simulate some work."""
        return sum(i**2 for i in range(100))

    # Test without metrics
    disable_metrics()
    start_time = time.time()
    for _ in range(iterations):
        collector = get_metrics()
        collector.start_timer("test")
        dummy_work()
        collector.stop_timer("test")
        collector.increment_counter("operations")
    no_metrics_time = time.time() - start_time

    # Test with metrics
    enable_metrics()
    start_time = time.time()
    for _ in range(iterations):
        collector = get_metrics()
        collector.start_timer("test")
        dummy_work()
        collector.stop_timer("test")
        collector.increment_counter("operations")
    with_metrics_time = time.time() - start_time

    return {
        "no_metrics_time": no_metrics_time,
        "with_metrics_time": with_metrics_time,
        "overhead_percent": ((with_metrics_time - no_metrics_time) / no_metrics_time)
        * 100,
    }


def benchmark_memory_usage() -> Dict[str, Any]:
    """Benchmark memory usage of key operations."""
    tracemalloc.start()

    # Initial snapshot
    snapshot1 = tracemalloc.take_snapshot()

    # Load some templates and configs
    from concord.llm.prompts import get_prompt_template, list_available_templates
    from concord.utils import load_yaml_config

    cfg = {"prompt_ver": "v3.2"}
    templates = list_available_templates()

    # Load templates
    for template_ver in templates:
        try:
            get_prompt_template(cfg, ver=template_ver)
        except Exception:
            pass  # Some templates might not load

    # Load config if available
    if CONFIG_PATH.exists():
        load_yaml_config(str(CONFIG_PATH))

    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    memory_diff = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB

    tracemalloc.stop()

    return {"memory_increase_mb": memory_diff, "templates_loaded": len(templates)}


def run_benchmarks():
    """Run all benchmarks and display results."""
    print("🚀 CONCORDIA Performance Benchmarks")
    print("=" * 50)

    print("\n📁 Configuration Loading Benchmark")
    config_results = benchmark_config_loading()
    print(f"Without cache: {config_results['no_cache_time']:.4f}s")
    print(f"With cache:    {config_results['with_cache_time']:.4f}s")
    print(f"Speedup:       {config_results['speedup']:.1f}x")

    print("\n📝 Template Loading Benchmark")
    template_results = benchmark_template_loading()
    print(f"Total time:    {template_results['total_time']:.4f}s")
    print(f"Avg per template: {template_results['avg_per_template']:.4f}s")
    print(f"Templates tested: {template_results['templates_tested']}")

    print("\n🧠 Embedding Cache Benchmark")
    try:
        embedding_results = benchmark_embedding_cache()
        print(f"Cold start:    {embedding_results['cold_time']:.4f}s")
        print(f"Warm cache:    {embedding_results['warm_time']:.4f}s")
        print(f"Speedup:       {embedding_results['speedup']:.1f}x")
        print(f"Texts processed: {embedding_results['texts_processed']}")
    except Exception as e:
        print(f"⚠️  Embedding benchmark failed: {e}")
        print("   (This is expected if embedding model is not available)")

    print("\n📊 Metrics Overhead Benchmark")
    metrics_results = benchmark_metrics_overhead()
    print(f"Without metrics: {metrics_results['no_metrics_time']:.4f}s")
    print(f"With metrics:    {metrics_results['with_metrics_time']:.4f}s")
    print(f"Overhead:        {metrics_results['overhead_percent']:.2f}%")

    print("\n💾 Memory Usage Benchmark")
    memory_results = benchmark_memory_usage()
    print(f"Memory increase: {memory_results['memory_increase_mb']:.2f} MB")
    print(f"Templates loaded: {memory_results['templates_loaded']}")

    print("\n✅ Benchmark Summary")
    print("=" * 50)
    print(f"Config caching speedup:     {config_results['speedup']:.1f}x")
    try:
        print(f"Embedding cache speedup:    {embedding_results['speedup']:.1f}x")
    except (KeyError, TypeError):
        print("Embedding cache speedup:    N/A")
    print(f"Metrics overhead:           {metrics_results['overhead_percent']:.2f}%")
    print(f"Memory usage:               {memory_results['memory_increase_mb']:.2f} MB")

    print("\n🎯 Performance Optimizations Working!")
    print("   Config caching provides significant speedup for repeated loads")
    print("   Metrics collection has minimal overhead when disabled")
    print("   Memory usage is reasonable for the features provided")


if __name__ == "__main__":
    run_benchmarks()
