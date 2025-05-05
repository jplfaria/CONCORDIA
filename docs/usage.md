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

# Specify columns and custom delimiter
concord data/pairs.tsv --mode sim-hint --col-a term1 --col-b term2 --sep '\t'
```

### Global Options

```
--cfg TEXT           YAML config path (default: config.yaml)
--mode TEXT          local | zero-shot | sim-hint | vote
--llm-model TEXT     Override gateway model (e.g. gpto3mini)
--prompt-ver TEXT    Freeze a specific prompt version
--output TEXT        Destination CSV
--overwrite          Overwrite existing output file
--batch-size INT     Batch size for processing (default: 32)
--device TEXT        Device for embedding model (cpu/cuda)
--preload            Preload embedding model to reduce first latency
--log-level TEXT     Logging level (DEBUG/INFO/WARNING/ERROR)
--log-file TEXT      Log to file in addition to console
--list-templates     List available prompt templates
--verbose, -v        Show detailed evidence and explanations
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
