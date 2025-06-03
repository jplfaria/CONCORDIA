# API Reference
# CONCORDIA Codebase Structure

## Core Modules

- **cli.py**: Streamlined command-line interface with grouped options and Typer app
- **pipeline.py**: File and pair annotation pipeline with optimized processing
- **modes.py**: Simplified annotation modes (local, zero-shot, vote, rac)
- **embedding.py**: Optimized embedding model loading and similarity calculation
- **constants.py**: Global constants and configuration defaults
- **utils.py**: Shared utilities with configuration caching and performance helpers
- **metrics.py**: Optional performance tracking and metrics collection

## LLM Modules

- **llm/argo_gateway.py**: Client for Argo Gateway API with "Related" → "Different" mapping
- **llm/prompts.py**: Simplified prompt template management with external template support
- **llm/template_store.py**: Template loading and storage utilities
- **llm/prompt_builder.py**: Builds prompts with placeholders
- **llm/prompt_buckets.py**: Bucket management for targeted templates with few-shot examples

## Label System

CONCORDIA uses a **6-class ontology**:
```python
LABEL_SET = {
    "Exact",
    "Synonym", 
    "Broader",
    "Narrower",
    "Uninformative",
    "Different"
}
```

*Note: "Related" has been removed and is automatically mapped to "Different" for backward compatibility.*

## API Reference

### concord.cli

`concord` – Streamlined single command CLI entrypoint with grouped options:

- **Input options**: file, text-a, text-b, col-a, col-b
- **Configuration options**: cfg, mode, prompt-ver
- **LLM options**: llm-model, llm-batch-size, llm-stream, llm-debug
- **Processing options**: batch-size, device, sim-hint, preload
- **Output options**: output, overwrite, sep, verbose
- **Utility options**: log-level, log-file, list-templates

### concord.pipeline

- `run_pair(a: str, b: str, cfg_path: Path)` – Annotate single pair, returns (label, similarity, evidence)
- `run_file(file_path: Path, cfg_path: Path, ...)` – Annotate file in batch, returns output file path

### concord.modes

Simplified annotation modes:
- `annotate_local(a, b, cfg)` – Similarity-only mode using embeddings
- `annotate_zero_shot(a, b, cfg)` – Single LLM call with optional similarity hint
- `annotate_vote(a, b, cfg)` – Multiple LLM calls with majority vote
- `annotate_rac(a, b, cfg)` – Retrieval-augmented classification (Beta)
- `annotate_fallback(a, b, cfg, err)` – Fallback to similarity on errors

Each mode returns a dict with:
- `label`: The classification result (from 6-class system)
- `similarity`: Cosine similarity value (if calculated)
- `evidence`: Explanation for the result
- `conflict`: Boolean indicating if there was disagreement
- `votes`: List of individual votes (vote mode only)

### concord.embedding

Optimized embedding operations:
- `embed_sentence(text, cfg)` – Get vector embedding for text
- `batch_embed(texts, cfg, batch_size)` – Batch embed multiple texts
- `cosine_sim(vec1, vec2)` – Calculate cosine similarity
- `similarity(text1, text2, cfg)` – One-step similarity between texts
- `preload_model(cfg)` – Preload model to reduce first-call latency
- `clear_cache()` – Free memory from cached model

### concord.metrics

Optional performance tracking (controlled by `CONCORDIA_METRICS` environment variable):
- `enable_metrics()` – Enable metrics collection globally
- `disable_metrics()` – Disable metrics collection globally
- `is_metrics_enabled()` – Check if metrics are enabled
- `get_metrics()` – Get global metrics collector
- `timed(name)` – Decorator to time function execution
- `evaluate_gold_standard(predictions_path, gold_path, ...)` – Evaluate against gold standard

### concord.utils

Performance-optimized utilities:
- `load_yaml_config(path, use_cache=True)` – Load YAML with optional caching
- `enable_config_cache()` – Enable configuration caching
- `disable_config_cache()` – Disable configuration caching
- `clear_config_cache()` – Clear configuration cache
- `with_retries(max_retries, backoff_factor)` – Retry decorator
- `validate_config(config)` – Validate configuration dictionary
- `validate_template(template, raise_error=True)` – Validate template placeholders

### concord.llm.prompts

Simplified template management:
- `get_prompt_template(cfg, ver=None, bucket_pair=None)` – Get prompt template
- `list_available_templates()` – List all available templates
- `choose_bucket(a, b)` – Determine bucket for template routing
- `save_template(version, content)` – Save template to file
- `build_annotation_prompt(a, b, template)` – Fill placeholders in template

Constants:
- `PROMPT_VER = "v3.2"` – Default prompt version
- `LABEL_SET` – Set of valid classification labels

### concord.llm.argo_gateway

- `ArgoGatewayClient` – Client for Argo Gateway with "Related" alias mapping
- `llm_label(a, b, client, cfg, template, with_note=False)` – Get LLM classification

## Templates

Available prompt templates:
- `v1.0`: Basic template with all labels listed
- `v1.1-general`: Microbial genome curator context
- `v1.1-enzyme`: Enzymology specialist context  
- `v1.1-phage`: Bacteriophage protein expert context
- `v3.2`: Latest comprehensive template with detailed heuristics (default)
- `v3.2-CoT`: Chain-of-thought version of v3.2

Templates are loaded from external files in `concord/llm/templates/` directory.

## Configuration Files

- **config.yaml**: Main configuration file containing engine settings, LLM parameters, and embedding configuration.

### Key Configuration Sections
```yaml
engine:
  mode: zero-shot
  sim_hint: false

llm:
  model: gpt4o
  stream: false

embedding:
  model_id: NeuML/pubmedbert-base-embeddings
  device: cpu
  batch_size: 32

prompt_ver: v3.2
```

## Environment Variables

- `CONCORDIA_METRICS`: Enable metrics collection (true/false, default: false)
- `ARGO_USER`: Argo Gateway username
- `ARGO_API_KEY`: Argo Gateway API key
