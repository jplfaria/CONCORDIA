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

While individual benchmark configurations can be run using `benchmark_runner.py` as shown below, the `eval/scripts/run_custom_benchmarks.sh` script is provided to automate running benchmarks across a suite of predefined models and prompt versions.

### Using `run_custom_benchmarks.sh` (Recommended for full suite)

The `run_custom_benchmarks.sh` script iterates through models like `gpt4o`, `gpto3mini`, etc., and different prompt versions, generating results in a timestamped directory under `eval/results/` with subfolders for each model.

Example usage (from the project root):
```bash
bash eval/scripts/run_custom_benchmarks.sh
```
Consult the script for current default models, prompts, and modes. It currently defaults to `zero-shot` mode.

### Using `benchmark_runner.py` (For specific configurations)

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

After running benchmarks (e.g., using `run_custom_benchmarks.sh`), evaluate the generated predictions:

```bash
python eval/evaluate_suite.py \
  --gold eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --pred-dir eval/results/your_benchmark_run_timestamp_dir \
  --pattern "**/*.csv" \
  --out eval/results/your_benchmark_run_timestamp_dir/evaluation_output \
  --plot
```

Replace `your_benchmark_run_timestamp_dir` with the actual directory name created by the benchmark script (e.g., `benchmark_run_20250521_011119`).
