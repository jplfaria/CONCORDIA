# CONCORDIA
*CONcordance of Curated & Original Raw Descriptions In Annotations*

A toolkit for annotation concordance and entity relationship classification using embeddings and LLMs.

## Features
- **gateway-check**: Argo Gateway API connectivity check on startup with prod/dev endpoint fallback
- **local**: PubMedBERT embeddings → cosine similarity → heuristic labels
- **zero-shot**: Single LLM call with optional similarity hints
- **vote**: Multiple LLM calls with majority vote (with vote tracking)
- **rac** (Beta): Retrieval-Augmented Classification with example memory
- **fallback**: Safe local fallback on errors
- Template-driven prompt management with versioned external templates (v1.x, v2, v2.1, v3.0, v3.1)
- Ad-hoc mode for quick two-sentence comparisons (without requiring a CSV file)
- **list-templates**: List available prompt templates
- **batch processing**: Control both file chunking and LLM batch sizes
- **verbose**: Show detailed evidence and explanations

## Installation

### Using Poetry (recommended)
```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install          # install dependencies & CLI entry-point
poetry shell            # activate the virtual environment
```

### Alternative via pip
```bash
pip install concordia
```

### Syncing Local Dependencies
If you've installed additional Python packages in your environment, you can compare them with Poetry-managed dependencies:
```bash
# export current environment packages
pip freeze > env-requirements.txt

# export Poetry-managed requirements
poetry export -f requirements.txt --without-hashes > poetry-requirements.txt

# view differences
diff env-requirements.txt poetry-requirements.txt
```
Manually add any missing packages to `pyproject.toml` under `[tool.poetry.dependencies]` and run `poetry update`.

## Quickstart
**CLI**
```bash
# Simplified command structure (single invocation)
concord example_data/annotations_test.csv --mode zero-shot --llm-model gpt4o
concord example_data/annotations_test.csv --mode local --output local.csv
concord example_data/annotations_test.csv --mode vote --output results_vote.csv
concord example_data/annotations_test.csv --mode rac --output results_rac.csv

# Direct text comparison (no CSV required)
concord --text-a "Entity A" --text-b "Entity B" --mode zero-shot

# List available templates
concord --list-templates

# Control batch processing
concord example_data/annotations_test.csv --batch-size 32 --llm-batch-size 12
```

**Python**
```python
from concord.pipeline import run_pair, run_file
label, sim, evidence = run_pair("Entity A", "Entity B", "config.yaml")
print(label, sim, evidence)
```

## Evaluation
After generating predictions (e.g., from a benchmark run), evaluate them against the gold standard using `eval/evaluate_suite.py`.
For detailed instructions on running benchmark suites and evaluation, see the [Benchmarking Workflow](docs/benchmarking.md).

Example evaluation command:
```bash
python eval/evaluate_suite.py \
  --gold eval/datasets/Benchmark_subset__200_pairs_v1.csv \
  --pred-dir eval/results/your_benchmark_run_timestamp_dir \
  --pattern "**/*.csv" \
  --out eval/results/your_benchmark_run_timestamp_dir/evaluation_output \
  --plot
```
Replace `your_benchmark_run_timestamp_dir` with the specific output directory of your benchmark run.

## Configuration (`config.yaml`)
```yaml
engine:
  mode: zero-shot        # local | zero-shot | vote | rac
  sim_hint: false       # Optional: prefix similarity hint to prompts

llm:
  model: gpt4o          # use without hyphens
  stream: false
  user: ${ARGO_USER}

local:
  model_id: NeuML/pubmedbert-base-embeddings
  device: cpu           # cpu or cuda

# RAC mode settings (Beta)
rac:
  example_limit: 3      # Number of examples to include in prompts
  similarity_threshold: 0.6  # Minimum similarity to include example
  auto_store: true      # Auto-save classifications to vector store

data_dir: "./data"      # Where to store the vector database
```

### Configuration Fields
- `engine.mode`: select mode (`local`, `zero-shot`, `vote`, `rac`)
- `engine.sim_hint`: boolean flag to prefix cosine similarity hint to LLM prompts (default: false)
- `engine.sim_threshold`: similarity threshold for local mode (default: 0.98)
- `engine.vote_temps`: list of temperatures for vote mode LLM calls (default: `[0.8, 0.2, 0.0]`)
- `llm.model`: Gateway model name (e.g. `gpt4o`, `gpt35`, `gpto3mini`)
- `llm.stream`: `true` to use streaming `/streamchat/` endpoint
- `llm.user`: Argo Gateway username (via `ARGO_USER`)
- `llm.api_key`: Argo Gateway API key (via `ARGO_API_KEY`)
- `prompt_ver`: explicit prompt version to use (overrides config `prompt_ver` and bucket routing)
- `local.model_id`: embedding model ID (PubMedBERT or SPECTER2)
- `local.device`: device for embeddings (`cpu` or `cuda`)
- `local.batch_size`: batch size for file processing
- `rac.example_limit`: number of similar examples to retrieve (for RAC mode)
- `rac.similarity_threshold`: minimum similarity score for examples (0-1)
- `rac.auto_store`: whether to automatically store successful classifications
- `data_dir`: directory for storing vector database and other data

## RAC Mode (Beta)

The Retrieval-Augmented Classification (RAC) mode is currently in beta development. This mode enhances classification by retrieving similar previously classified examples and including them in the prompt for context.

### Current Limitations

RAC mode currently has several limitations being actively worked on:

1. **All Classifications Get Stored**: Currently, all successful LLM classifications are stored in the vector database if `auto_store` is enabled, regardless of quality or accuracy.

2. **Planned Improvements**:
   - Human validation before storing examples
   - Confidence thresholds from the LLM responses
   - Selective storage based on specific characteristics or patterns
   - Improved embedding methods for better similarity matching

### Using RAC Mode

```bash
# First time setup - create data directory
mkdir -p data

# Run with RAC mode (will build up examples over time)
concord example_data/annotations_test.csv --mode rac --output results_rac.csv
```

## Documentation
```bash
mkdocs serve
```
Published site: https://<org>.github.io/concordia/

## Environment Variables
- `ARGO_USER`: ANL login for Argo Gateway (required)
- `ARGO_API_KEY`: API key for private Argo Gateway (optional)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Testing
Run all tests via `pytest`:
```bash
pytest
```

## Development
We enforce formatting and linting with pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## License
Apache-2.0