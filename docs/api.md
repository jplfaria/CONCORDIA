# API Reference
# CONCORDIA Codebase Structure

## Core Modules

- **cli.py**: Command-line interface with Typer app
- **pipeline.py**: File and pair annotation pipeline
- **modes.py**: Different annotation modes (local, zero-shot, vote)
- **embedding.py**: Embedding model loading and similarity calculation
- **constants.py**: Global constants and configuration defaults

## LLM Modules

- **llm/argo_gateway.py**: Client for Argo Gateway API
- **llm/prompts.py**: Prompt template management
- **llm/template_store.py**: Template loading and storage
- **llm/prompt_builder.py**: Builds prompts with placeholders
- **llm/prompt_buckets.py**: Bucket management for targeted templates

## I/O Modules

- **io/loader.py**: Data loading utilities

## API Reference

### concord.cli

`concord` – Single command CLI entrypoint for all annotation tasks.

### concord.pipeline

- `run_pair(a: str, b: str, cfg_path: Path)` – Annotate single pair, returns (label, similarity, evidence)
- `run_file(file_path: Path, cfg_path: Path, ...)` – Annotate file in batch, returns output file path

### concord.modes

- `annotate_local(a, b, cfg)` – Similarity-only mode using embeddings
- `annotate_zero_shot(a, b, cfg)` – Single LLM call with optional similarity hint
- `annotate_vote(a, b, cfg)` – Multiple LLM calls with majority vote
- `annotate_fallback(a, b, cfg, err)` – Fallback to similarity on errors

Each mode returns a dict with:
- `label`: The classification result
- `similarity`: Cosine similarity value (if calculated)
- `evidence`: Explanation for the result
- `conflict`: Boolean indicating if there was disagreement
- `votes`: List of individual votes (vote mode only)

### concord.embedding

- `embed_sentence(text, cfg)` – Get vector embedding for text
- `batch_embed(texts, cfg, batch_size)` – Batch embed multiple texts
- `cosine_sim(vec1, vec2)` – Calculate cosine similarity
- `similarity(text1, text2, cfg)` – One-step similarity between texts
- `preload_model(cfg)` – Preload model to reduce first-call latency
- `clear_cache()` – Free memory from cached model

### concord.llm.template_store

- `get_prompt_template(cfg, ver=None, bucket_pair=None)` – Get prompt template
- `list_available_templates()` – List all available templates

## Templates

Available prompt templates:
- `v1.0`, `v1.1-general`, etc. – Original templates
- `v2` – Updated template with required labels and markers
- `v2.1` – Further improved template with better validation

## Configuration Files

- **config.yaml**: Main configuration file containing engine settings, LLM parameters, and local model configuration.
