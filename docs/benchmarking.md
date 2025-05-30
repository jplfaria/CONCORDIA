# Benchmarking Workflow

This document describes the end-to-end benchmarking process for CONCORDIA.

## Overview

The benchmarking workflow consists of:

1. Preparing datasets in `eval/datasets/`.
2. Running benchmarks with the Python CLI in `eval/scripts/benchmark_runner.py` (or more commonly, using the `eval/scripts/run_custom_benchmarks.sh` script).
3. Evaluating results with `eval/evaluate_suite.py`.
4. Inspecting metrics and plots. Raw outputs from benchmark runs are typically generated in a timestamped subdirectory within `eval/results/` (this directory is ignored by Git). An example of a complete run, including evaluation outputs, can be found in `eval/example_outputs/` (this directory is tracked by Git).

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

After running benchmarks (e.g., using `run_custom_benchmarks.sh`), evaluate the generated predictions using `evaluate_suite.py`.

The main `eval/results/` directory, where `run_custom_benchmarks.sh` saves its output, is ignored by Git to prevent clutter from multiple test runs. A complete example of a benchmark run's output, including the `evaluation_output` subdirectory with plots and metrics, can be found in `eval/example_outputs/`. This directory is tracked by Git and serves as a reference for the expected structure and content.

Key points for `evaluate_suite.py`:
- If `--pred-dir` is not specified, the script will attempt to automatically use the latest `benchmark_run_*` directory found in `eval/results/`.
- The default pattern for finding prediction files (if `--pattern` is not specified) is `"*_predictions.csv"`. The example below uses a more general pattern.
- For advanced use cases with custom CSV formats, arguments like `--gold-s1-col`, `--pred-rel-col`, etc., are available to specify column names. Run `python eval/evaluate_suite.py --help` for details.

Example evaluation command:
```bash
python eval/evaluate_suite.py \
  --gold eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --pred-dir eval/results/your_benchmark_run_timestamp_dir \
  --pattern "**/*.csv" \
  --out eval/results/your_benchmark_run_timestamp_dir/evaluation_output \
  --plot  # Ensures generation of .png plots (confusion matrices, summary charts)
```

If omitting `--pred-dir` to use the auto-detected latest run, the command might look like:
```bash
python eval/evaluate_suite.py \
  --gold eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --pattern "**/*.csv" \
  --plot  # Ensures generation of .png plots
```
Note: When using auto-detection for `--pred-dir`, the `--out` path will also be relative to the auto-detected directory (e.g., `evaluation_output` inside it). If you specify `--pred-dir`, ensure your `--out` path is also appropriate, typically pointing to an `evaluation_output` subdirectory within your chosen `pred-dir`.
