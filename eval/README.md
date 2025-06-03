# Benchmarking

This directory contains the end-to-end benchmarking workflow for CONCORDIA:

- `datasets/` – Gold and unlabeled CSVs for benchmarks.
- `scripts/` – Helper scripts for running and processing benchmarks.
- `results/` – Main directory for generated benchmark outputs (ignored by Git).
- `example_outputs/` – Contains an example of a full benchmark run output, including evaluation results (tracked by Git). This serves as a reference for the expected structure.

## Label System

CONCORDIA uses a **6-class ontology** for relationship classification:

- **Exact**: Wording differences only (capitalization, punctuation, accepted synonyms)
- **Synonym**: Same biological entity but nomenclature or EC renumbering changes
- **Broader**: B is more generic; loses specificity present in A
- **Narrower**: B adds specificity (extra EC digits, substrate, sub-unit, phage part)
- **Different**: Unrelated proteins or functions (includes pathway neighbors or fused variants)
- **Uninformative**: Neither term provides enough functional information

*Note: The "Related" label from the previous 7-class system is automatically mapped to "Different" for backward compatibility during evaluation.*

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
  --prompts v1.0 v3.2 v3.2-CoT \
  --hint both \
  --llm-model gpt4o \
  --out-dir eval/results
```

### Available Templates

Current available templates for benchmarking:
- `v1.0`: Basic template with all labels listed
- `v1.1-general`: Microbial genome curator context
- `v1.1-enzyme`: Enzymology specialist context
- `v1.1-phage`: Bacteriophage protein expert context
- `v3.2`: Latest comprehensive template with detailed heuristics (default)
- `v3.2-CoT`: Chain-of-thought version of v3.2

### Performance Optimizations

For faster benchmarking, consider enabling performance features:

```bash
# Enable metrics to track performance
export CONCORDIA_METRICS=true

# Use larger batch sizes for faster processing
python eval/scripts/benchmark_runner.py \
  --data eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --modes zero-shot \
  --prompts v3.2 \
  --batch-size 64 \
  --llm-batch-size 8 \
  --llm-model gpt4o
```

## Evaluating Results

After generating benchmark predictions (typically using `run_custom_benchmarks.sh`), use `eval/evaluate_suite.py` to compare predictions against the gold standard, calculate metrics, and produce visualizations.

The main `eval/results/` directory (where `run_custom_benchmarks.sh` saves its output) is ignored by Git to prevent clutter from multiple test runs. However, a complete example of a benchmark run's output, including the `evaluation_output` subdirectory with plots and metrics, can be found in `eval/example_outputs/`. This directory is tracked by Git and serves as a reference.

Key points for `evaluate_suite.py`:
- If `--pred-dir` is not specified, the script will attempt to automatically use the latest `benchmark_run_*` directory found in `eval/results/`.
- The default pattern for finding prediction files (if `--pattern` is not specified) is `"*_predictions.csv"`. The example below uses a more general pattern.
- For advanced use cases with custom CSV formats, arguments like `--gold-s1-col`, `--pred-rel-col`, etc., are available to specify column names. Run `python eval/evaluate_suite.py --help` for details.
- **Important**: The evaluation automatically handles the "Related" → "Different" mapping for backward compatibility.

#### CSV Column Name Conventions for Evaluation

For seamless integration with `evaluate_suite.py` and `benchmark_runner.py`, your gold standard CSV files (in `eval/datasets/`) should ideally use the following column headers for the core data:

*   `annotation_a`: The first sentence or text item to compare.
*   `annotation_b`: The second sentence or text item to compare.
*   `relationship_label`: The gold standard relationship label between `annotation_a` and `annotation_b`.

If your gold standard CSV uses these headers, `benchmark_runner.py` will propagate them to its prediction output files. Consequently, `evaluate_suite.py` will be able to process these files without requiring additional column name arguments.

**Handling Non-Standard Column Names:**

If your gold standard CSV files use different column names, you **must** inform `evaluate_suite.py` using the following arguments. The script will then expect the corresponding prediction files (generated by `benchmark_runner.py`) to also contain these specified column names for the sentence pairs.

*   `--gold-s1-col "YOUR_GOLD_SENTENCE1_COLUMN_NAME"`
*   `--gold-s2-col "YOUR_GOLD_SENTENCE2_COLUMN_NAME"`
*   `--gold-rel-col "YOUR_GOLD_RELATIONSHIP_COLUMN_NAME"` (default is `relationship_label`)
*   `--pred-s1-col "YOUR_PRED_SENTENCE1_COLUMN_NAME"`
*   `--pred-s2-col "YOUR_PRED_SENTENCE2_COLUMN_NAME"`
*   `--pred-rel-col "YOUR_PRED_RELATIONSHIP_COLUMN_NAME"` (default is `relationship`)

**Example:** If your gold standard and prediction files use `old_annotation` for the first sentence and `new_annotation` for the second, and the gold relationship is `gold_relation` while the prediction relationship is `predicted_relation`, your command would include:

```bash
python eval/evaluate_suite.py \
  --gold path/to/your/gold.csv \
  --pred-dir path/to/your/predictions/ \
  --gold-s1-col "old_annotation" \
  --gold-s2-col "new_annotation" \
  --gold-rel-col "gold_relation" \
  --pred-s1-col "old_annotation" \
  --pred-s2-col "new_annotation" \
  --pred-rel-col "predicted_relation" \
  --plot
```
Using the standard column names (`annotation_a`, `annotation_b`, `relationship_label`) in your `eval/datasets/` files is recommended to simplify the evaluation command.

**Example evaluation command (from the project root):**
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

## Evaluation Metrics

The evaluation suite automatically handles the 6-class label system and provides comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged metrics
- **Matthews Correlation Coefficient (MCC)**: Balanced performance measure
- **Confusion Matrix**: Visual representation of classification performance

All metrics are automatically adjusted for the "Related" → "Different" mapping to ensure consistent evaluation across different dataset versions.

Further details on evaluation metrics and plots can be found in the main project [Benchmarking Workflow documentation](../docs/benchmarking.md).
