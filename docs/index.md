# CONCORDIA Documentation

Welcome to the CONCORDIA documentation site.

CONCORDIA is a toolkit for annotation concordance and entity relationship classification using embeddings and LLMs with a **6-class ontology system**.

## Quick Navigation

- **[Usage Guide](usage.md)**: CLI and Python API examples with performance features
- **[API Reference](api.md)**: Detailed module and class documentation
- **[Benchmarking](benchmarking.md)**: End-to-end benchmarking workflow and evaluation

## Key Features

- **6-class relationship ontology**: Exact, Synonym, Broader, Narrower, Different, Uninformative
- **Multiple processing modes**: local, zero-shot, vote, rac (retrieval-augmented)
- **Performance optimizations**: Configuration caching, optional metrics, embedding preloading
- **Comprehensive templating**: v3.2 default with external template files
- **Evaluation suite**: Automated benchmarking and metrics calculation

## Performance Optimizations

CONCORDIA includes several performance features:
- **Optional metrics collection** (via `CONCORDIA_METRICS` environment variable)
- **Configuration caching** for faster repeated loads
- **Embedding model preloading** to reduce first-call latency
- **Batch processing controls** for optimal throughput
