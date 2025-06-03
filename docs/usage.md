# Usage

## CLI

Use the `concord` command to annotate pairs in a file.

> **Note:** On startup, the CLI checks Argo Gateway API connectivity and warns if unreachable.

```bash
# Annotate a CSV file with zero-shot LLM mode
concord data/pairs.csv \
  --mode zero-shot \
  --output results.csv

# Local mode (fast cosine similarity only)
concord data/pairs.csv --mode local --output local_results.csv

# Vote mode with specific prompt template
concord data/pairs.csv --mode vote --prompt-ver v3.2 --output vote_results.csv

# Specify columns and custom delimiter
concord data/pairs.tsv --mode zero-shot --col-a term1 --col-b term2 --sep '\t'

# Enable performance metrics
CONCORDIA_METRICS=true concord data/pairs.csv --mode zero-shot --verbose

# Preload embedding model for faster processing
concord data/pairs.csv --mode zero-shot --preload --batch-size 64
```

## CLI Options (Grouped)

### Input Options
- `file`: Input CSV/TSV file (positional argument)
- `--text-a TEXT`: First annotation text (for direct pair comparison)
- `--text-b TEXT`: Second annotation text (for direct pair comparison)
- `--col-a TEXT`: Column name for first annotation
- `--col-b TEXT`: Column name for second annotation

### Configuration Options
- `--cfg TEXT`: YAML config file (default: config.yaml)
- `--mode TEXT`: Annotation mode: local|zero-shot|vote|rac
- `--prompt-ver TEXT`: Prompt template version (e.g., v1.0, v3.2, v3.2-CoT)

### LLM Options
- `--llm-model TEXT`: LLM model override (e.g., gpt4o, gpto3mini)
- `--llm-batch-size INT`: Batch size for LLM calls (default: 1)
- `--llm-stream/--no-llm-stream`: Force streaming mode on/off
- `--llm-debug`: Enable LLM debugging

### Processing Options
- `--batch-size INT`: Processing batch size (default: 32)
- `--device TEXT`: Device for embeddings (cpu/cuda, default: cpu)
- `--sim-hint`: Include similarity hints in prompts
- `--preload`: Preload embedding model to reduce first-call latency

### Output Options
- `--output TEXT`: Output CSV file path
- `--overwrite`: Overwrite existing output file
- `--sep TEXT`: CSV delimiter
- `--verbose/-v`: Show detailed evidence and explanations

### Utility Options
- `--log-level TEXT`: Logging level (DEBUG/INFO/WARNING/ERROR, default: INFO)
- `--log-file TEXT`: Log to file in addition to console
- `--list-templates`: List available prompt templates

## Label System

CONCORDIA uses a **6-class ontology** for relationship classification:

- **Exact**: Wording differences only (capitalization, punctuation, accepted synonyms)
- **Synonym**: Same biological entity but nomenclature or EC renumbering changes
- **Broader**: B is more generic; loses specificity present in A
- **Narrower**: B adds specificity (extra EC digits, substrate, sub-unit, phage part)
- **Different**: Unrelated proteins or functions (includes pathway neighbors or fused variants)
- **Uninformative**: Neither term provides enough functional information

*Note: The "Related" label from the previous 7-class system is automatically mapped to "Different" for backward compatibility.*

## Available Templates

```bash
# List all available templates
concord --list-templates
```

Current templates include:
- `v1.0`: Basic template with all labels listed
- `v1.1-general`: Microbial genome curator context
- `v1.1-enzyme`: Enzymology specialist context
- `v1.1-phage`: Bacteriophage protein expert context
- `v3.2`: Latest comprehensive template with detailed heuristics (default)
- `v3.2-CoT`: Chain-of-thought version of v3.2

## Evaluation

After generating predictions (e.g., from a benchmark run as described in the [Benchmarking Workflow](benchmarking.md)), evaluate them against the gold standard using `eval/evaluate_suite.py`.

```bash
python eval/evaluate_suite.py \
  --gold eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --pred-dir eval/results/your_benchmark_run_timestamp_dir \
  --pattern "**/*.csv" \
  --out eval/results/your_benchmark_run_timestamp_dir/evaluation_output \
  --plot
```
Replace `your_benchmark_run_timestamp_dir` with the specific output directory of your benchmark run.

## Output Files

Vote mode outputs include:
- `relationship`: The classification result
- `evidence`: Explanation for the result 
- `duo_conflict`: Boolean indicating if there was disagreement
- `votes`: Semicolon-separated list of individual votes

All modes include:
- `similarity_Pubmedbert`: Cosine similarity score (when calculated)
- `evidence`: Evidence or reasoning for the classification

## Performance Features

### Metrics Collection
```bash
# Enable detailed performance metrics (disabled by default)
export CONCORDIA_METRICS=true
concord data/pairs.csv --mode zero-shot

# Metrics include timing, counters, and memory usage
```

### Configuration Caching
Configuration files are automatically cached to avoid repeated YAML parsing. Cache is based on file modification time.

### Batch Processing
```bash
# Control processing and LLM batch sizes independently
concord data/pairs.csv --batch-size 64 --llm-batch-size 8 --mode zero-shot
```

## Python API

```python
from concord.pipeline import run_pair, run_file

# Single pair
label, sim, evidence = run_pair(
    'Entity A', 'Entity B', Path('config.yaml')
)
print(label, sim, evidence)

# Batch file
output_path = run_file(
    Path('data/pairs.csv'),
    Path('config.yaml'),
    col_a='term1',
    col_b='term2',
    overwrite=True
)
print(f"Results written to {output_path}")

# Enable metrics programmatically
from concord.metrics import enable_metrics, get_metrics

enable_metrics()
# ... perform operations ...
metrics = get_metrics()
metrics.log_summary()