# CONCORDIA
*CONcordance of Curated & Original Raw Descriptions In Annotations*

A toolkit for annotation concordance and entity relationship classification using embeddings and LLMs.

## Features
- **gateway-check**: Argo Gateway API connectivity check on startup
- **local**: PubMedBERT embeddings → cosine similarity → heuristic labels
- **zero-shot**: Single LLM call with optional similarity hints
- **vote**: Multiple LLM calls with majority vote (with vote tracking)
- **fallback**: Safe local fallback on errors
- Template-driven prompt management with versioned external templates (v1.x, v2, v2.1)
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
# Simplified command structure (single invocation)
concord data/pairs.csv --mode zero-shot --output results.csv
concord data/pairs.csv --mode local --output local.csv
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
  sim_hint: false      # Optional: prefix similarity hint to prompts

llm:
  model: gpt4o        # use without hyphens
  stream: false
  user: ${ARGO_USER}

local:
  model_id: NeuML/pubmedbert-base-embeddings
```

### Configuration Fields
- `engine.mode`: select mode (`local`, `zero-shot`, `vote`)
- `engine.sim_hint`: boolean flag to prefix cosine similarity hint to LLM prompts (default: false)
- `engine.sim_threshold`: similarity threshold for local mode (default: 0.98)
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