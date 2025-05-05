# API Reference

## concord.cli

`concord` – CLI entrypoint for file and pair annotation.

## concord.pipeline

- `run_pair(a: str, b: str, cfg_path: Path)` – Annotate single pair via CLI.
- `run_file(file_path: Path, cfg_path: Path, ...)` – Annotate file in batch.

## concord.modes

- `annotate_local(a, b, cfg)`
- `annotate_zero_shot(a, b, cfg)`
- `annotate_sim_hint(a, b, cfg)`
- `annotate_vote(a, b, cfg)`
- `annotate_fallback(a, b, cfg, err)`

## concord.embedding

- `embed_sentence(text, cfg)`
- `batch_embed(texts, cfg, batch_size)`
- `cosine_sim(vec1, vec2)`
- `similarity(text1, text2, cfg)`
- `preload_model(cfg)`
- `clear_cache()`

## concord.llm.template_store

- `get_prompt_template(cfg, ver=None, bucket_pair=None)`
- `list_available_templates()`

## concord.llm.prompt_builder

- `build_annotation_prompt(a, b, template)`

## concord.llm.argo_gateway

- `ArgoGatewayClient` – client for Argo Gateway API
- `ArgoGatewayClient.ping()` – check if the Argo Gateway API is reachable (returns bool)
- `llm_label(a, b, client, cfg=None, template=None, with_note=False)` – wrapper to build prompt, call LLM, and parse the response

## concord.utils

Utility functions and helpers.
