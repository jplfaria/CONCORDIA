# CONCORDIA
*CONcordance of Curated & Original Raw Descriptions In Annotations*

A toolkit for biomedical entity-relationship classification using embeddings and LLMs.

## Features
- **gateway-check**: Argo Gateway API connectivity check on startup
- **local**: PubMedBERT embeddings → cosine similarity → heuristic labels
- **zero-shot**: Single LLM call without similarity hints
- **sim-hint**: Prefix similarity hint to LLM prompt
- **vote**: Three LLM calls at different temperatures, majority vote
- **fallback**: Safe local fallback on errors
- Template-driven prompt management with versioned external templates
- **list-templates**: List available prompt templates
- **verbose**: Show detailed evidence and explanations

## Installation
```bash
pip install concordia
# or from source
git clone <repo_url>
cd CONCORDIA
pip install -e .
```

## Quickstart
**CLI**
```bash
concord data/pairs.csv --mode zero-shot --output results.csv
concord data/pairs.csv --mode local --output local.csv
concord data/pairs.csv --mode sim-hint --output results_simhint.csv
concord data/pairs.csv --mode vote --output results_vote.csv
concord --list-templates
```

**Python**
```python
from concord.pipeline import run_pair, run_file
label, sim, evidence = run_pair("Entity A", "Entity B", "config.yaml")
print(label, sim, evidence)
```

## Configuration (`config.yaml`)
```yaml
engine:
  mode: zero-shot

llm:
  model: gpt4o       # use without hyphens
  stream: false
  user: ${ARGO_USER}

local:
  model_id: NeuML/pubmedbert-base-embeddings
```

### Configuration Fields
- `engine.mode`: select mode (`local`, `zero-shot`, `sim-hint`, `vote`, `fallback`)
- `llm.model`: Gateway model name (e.g. `gpt4o`)
- `llm.stream`: `true` to use streaming `/streamchat/` endpoint
- `llm.user`: Argo Gateway username (via `ARGO_USER`)
- `llm.api_key`: Argo Gateway API key (via `ARGO_API_KEY`)
- `local.model_id`: embedding model ID (PubMedBERT or SPECTER2)
- `local.device`: device for embeddings (`cpu` or `cuda`)
- `local.batch_size`: batch size for local mode

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