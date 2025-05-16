# Benchmarking

This directory contains the end-to-end benchmarking workflow for CONCORDIA:

- `datasets/` – Gold and unlabeled CSVs for benchmarks.
- `scripts/` – Helper scripts for running and processing benchmarks.
- `results/` – Generated outputs (ignored by Git).

## Usage

```bash
python eval/scripts/benchmark_runner.py \
  --data eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --modes zero-shot vote \
  --prompts v1.0 v2.1 v3.0 v3.0-CoT v3.1 v3.1-CoT \
  --hint both \
  --llm-model gpt4o \
  --out-dir eval/results
```
