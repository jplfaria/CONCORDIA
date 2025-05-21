# Benchmarking

This directory contains the end-to-end benchmarking workflow for CONCORDIA:

- `datasets/` – Gold and unlabeled CSVs for benchmarks.
- `scripts/` – Helper scripts for running and processing benchmarks.
- `results/` – Generated outputs (ignored by Git).

## Usage

There are two main ways to run benchmarks:

### 1. Using `run_custom_benchmarks.sh` (Recommended for full suite)

The `eval/scripts/run_custom_benchmarks.sh` script is the recommended way to run a comprehensive benchmark suite. It iterates through a predefined set of models (e.g., `gpt4o`, `gpto3mini`) and prompt versions, executing them in `zero-shot` mode by default.

Outputs are saved to a timestamped directory within `eval/results/`, with individual subfolders for each model's predictions.

**To run the custom benchmark suite (from the project root):**
```bash
bash eval/scripts/run_custom_benchmarks.sh
```
Consult the script itself for the latest list of models, prompts, and configurable options.

### 2. Using `benchmark_runner.py` (For specific configurations)

For running specific combinations of datasets, modes, prompts, and models, you can use `eval/scripts/benchmark_runner.py` directly.

**Example:**
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

After generating benchmark predictions (typically using `run_custom_benchmarks.sh`), use `eval/evaluate_suite.py` to compare predictions against the gold standard, calculate metrics, and produce visualizations.

**Example evaluation command (from the project root):**
```bash
python eval/evaluate_suite.py \
  --gold eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --pred-dir eval/results/your_benchmark_run_timestamp_dir \
  --pattern "**/*.csv" \
  --out eval/results/your_benchmark_run_timestamp_dir/evaluation_output \
  --plot
```
Replace `your_benchmark_run_timestamp_dir` with the actual name of the directory created by the benchmark script (e.g., `benchmark_run_20250521_011119`).

Further details on evaluation metrics and plots can be found in the main project [Benchmarking Workflow documentation](../../docs/benchmarking.md).
