# Benchmarking Workflow

This document describes the end-to-end benchmarking process for CONCORDIA.

## Overview

The benchmarking workflow consists of:

1. Preparing datasets in `eval/datasets/`.
2. Running benchmarks with the Python CLI in `eval/scripts/benchmark_runner.py`.
3. Evaluating results with `eval/scripts/evaluate_suite.py`.
4. Inspecting metrics and plots in `eval/results/`.

## Prerequisites

Install development requirements:

```bash
pip install -r requirements-dev.txt
```

## Running Benchmarks

```bash
python eval/scripts/benchmark_runner.py \
  --data eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --modes zero-shot vote \
  --prompts v1.0 v2.1 v3.0 v3.0-CoT v3.1 v3.1-CoT \
  --hint both \
  --llm-model gpt4o \
  --out-dir eval/results
```

## Evaluating Results

```bash
python eval/scripts/evaluate_suite.py \
  --gold eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --pred-dir eval/results \
  --pattern "*.csv" \
  --out eval/results \
  --plot
```
