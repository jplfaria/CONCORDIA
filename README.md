# CONCORDIA  
*CONcordance of Curated & Original Raw Descriptions In Annotations*

Concordia compares two functional-annotation sources—old vs new, RAST vs UniProt, manual vs AI—and writes a tidy table with  

* **`similarity_Pubmedbert`** – cosine similarity from the PubMedBERT sentence-embedding model  
* **`label`** – a one-word judgement from an LLM (GPT-4o or gpto3mini) drawn from a 7-class ontology  
* **`evidence`** – detailed reasoning from the LLM explaining the relationship classification

You can choose from **six processing modes**:

| Mode | What happens |
|------|--------------|
| **llm** | LLM only — cheapest if you already trust the model |
| **local** | PubMedBERT embeddings → cosine → heuristic label (no LLM) |
| **dual** | Embeddings **and** LLM (baseline) |
| **simhint** | Same as **dual** **plus** the cosine similarity is prefixed to the prompt as a weak prior |
| **bucket** | Choose template based on embedding similarity bucket |
| **duo** | Runs the LLM 3 times with different temperatures and takes a majority vote for more robust results |

---

## Installation
```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install          # installs deps + CLI
poetry shell            # activate the virtualenv
```

### Environment Setup
Concordia requires setting the ARGO_USER environment variable for accessing the Argo Gateway API:

```bash
export ARGO_USER=<your-ANL-username>
```

---

## Quick-start recipes

| Goal | Command |
|------|---------|
| CSV with default LLM (gpto3mini → dev) | `python -m concord.cli concord example_data/annotations_test.csv` |
| CSV with GPT-4o | `python -m concord.cli concord example_data/annotations_test.csv --llm-model gpt4o` |
| TSV with detailed evidence | `python -m concord.cli concord example_data/annotations_test.tsv --llm-model gpt4o --verbose` |
| Embed-only (no LLM) | `python -m concord.cli concord example_data/annotations_test.csv --mode local` |
| Dual (baseline) | `python -m concord.cli concord example_data/annotations_test.csv --mode dual` |
| Similarity-hint | `python -m concord.cli concord example_data/annotations_test.csv --mode simhint` |
| Template buckets | `python -m concord.cli concord example_data/annotations_test.csv --mode bucket` |
| Voting-based consensus | `python -m concord.cli concord example_data/annotations_test.csv --mode duo` |
| Two ad-hoc strings | `python -m concord.cli concord --text-a "RecA" --text-b "DNA recombinase A"` |
| List available templates | `python -m concord.cli concord --list-templates` |
| **Overwrite** existing output | `python -m concord.cli concord example_data/test.csv --output results.csv --overwrite` |
| Debug with verbose logging | `python -m concord.cli concord example_data/test.csv --log-level DEBUG` |

---

## Accepted input formats

| Extension | Loader | Note |
|-----------|--------|------|
| `.csv`           | `pandas.read_csv(sep=',')` | default |
| `.tsv` / `.tab`  | `pandas.read_csv(sep='\t')`| or `--sep "\t"` |
| `.json`          | `pandas.read_json()`       | list-of-objects **or** column-orient |

If you do **not** pass `--col-a / --col-b`, the **first two textual columns that don't end with `id`** are taken.

---

## Minimal sample CSV
```csv
annotation_a,annotation_b
DNA repair protein RecA,Recombinase A
Hypothetical protein,Uncharacterized protein
```
Any extra columns (e.g. `gene_id`) are preserved in the output.

---

## CLI options

| Flag | Description |
|------|-------------|
| **`FILE`**             | Input table (`.csv`, `.tsv`, `.json`) |
| `--text-a / --text-b`  | Compare two strings instead of a file |
| `--mode`               | `llm` (default) \| `local` \| `dual` \| `simhint` \| `bucket` \| `duo` |
| `--llm-model`          | Gateway LLM (`gpto3mini`, `gpt4o`) - Note: use without hyphens |
| `--retry`              | Automatic blank-reply retries (default 5) |
| `--overwrite`          | Overwrite existing output instead of appending |
| `--output`             | Destination path (file-mode only) |
| `--cfg`                | Alternate YAML config |
| `--col-a / --col-b`    | Explicit annotation columns |
| `--sep`                | Custom delimiter for text files (e.g. `"\t"`) |
| `--verbose / -v`       | Show detailed evidence and explanations |
| `--log-level`          | Set logging level (DEBUG/INFO/WARNING/ERROR) |
| `--log-file`           | Log to file in addition to console |
| `--list-templates`     | List available prompt templates |
| `--prompt-ver`         | Specify template version to use (e.g., "v1.3-test") |
| `--batch-size`         | Batch size for processing (default: 32) |
| `--device`             | Device for embedding model (cpu/cuda) |

---

## Modes in detail

| Mode | Pipeline | Extra columns |
|------|----------|---------------|
| **llm**    | Skip embeddings; every pair → LLM               | `label`, `evidence` |
| **local**  | PubMedBERT embeddings → cosine → heuristic      | `similarity_Pubmedbert`, `label` |
| **dual**   | Embeddings **and** LLM for every row            | `similarity_Pubmedbert`, `label`, `evidence` |
| **simhint**| Same as **dual**, but cosine similarity is sent to the LLM prompt | `similarity_Pubmedbert`, `label`, `evidence` |
| **bucket** | Choose template based on embedding similarity bucket | `similarity_Pubmedbert`, `label`, `evidence` |
| **duo**    | Runs the LLM 3 times with different temperatures and uses majority vote | `label`, `evidence`, `conflict` |

> **Embedding model** [`NeuML/pubmedbert-base-embeddings`](https://huggingface.co/NeuML/pubmedbert-base-embeddings) (Apache-2.0)

---

## 7-label ontology

| Label | Meaning |
|-------|---------|
| **Exact**        | Same function; wording/punctuation only |
| **Synonym**      | Semantically equivalent paraphrase |
| **Broader**      | A ⊃ B (A more general) |
| **Narrower**     | A ⊂ B (A more specific) |
| **Related**      | Same pathway / complex / family but not parent–child |
| **Uninformative**| Placeholder or extremely generic |
| **Different**    | No functional overlap |

### Alias system — why you rarely see **Unknown**

Older checkpoints reply with tokens like *Identical* or *Partial*.  
`llm_label()` holds a tiny alias map so such answers are remapped automatically; genuine blanks are the only source of `Unknown`.

---

## Prompting — Templates and Customization

All prompt text lives in **`concord/llm/prompts.py`** and external template files in **`concord/llm/templates/`**.  
Edit the few-shot examples or definitions to experiment; no other code must change.

Available templates can be listed with:
```bash
python -m concord.cli concord --list-templates
```

You can specify a template version with:
```bash
python -m concord.cli concord example_data/test.csv --prompt-ver "v1.3-test"
```

---

## Config snapshot (`concord/config.yaml`)
```yaml
engine:
  mode: llm

llm:
  model: gpt4o            # Use without hyphens
  stream: false
  user: ${ARGO_USER}      # Must set ARGO_USER environment variable

local:
  model_id: NeuML/pubmedbert-base-embeddings
```

---

## Viewing Detailed Evidence

By default, CONCORDIA provides abbreviated evidence for concise output. To see the full biological context:

```bash
python -m concord.cli concord example_data/test.csv --verbose
```

This shows detailed reasoning behind each classification, which is particularly valuable for complex biomedical entity relationships.

---

## Debugging Argo Gateway API Connections

For troubleshooting API connections:

1. Ensure the ARGO_USER environment variable is set
2. Use the correct model name format (e.g., "gpt4o" without hyphen)
3. Run with DEBUG logging level:
   ```bash
   python -m concord.cli concord --text-a "Test" --text-b "Test entity" --log-level DEBUG
   ```

The `debug_llm_call.py` utility script can also be used to test direct API connections.

---

## Progress, recovery & overwrite

* Output is **appended row-by-row** – abort with Ctrl-C and rerun; finished pairs are skipped.  
  Add `--overwrite` to **replace** an existing file instead.  
* Live progress bar example:
  ```
  Processing 73%|██████████████▋ | 730/1000 [00:14<00:05, 49.2it/s]
  ```
  gpto3mini ≈ 0.10 s per pair; GPT-4o ≈ 1.0 s per pair; embeddings ≈ 1 ms per string on Apple M-series.

---

## FAQ

| Q | A |
|---|---|
| **Why keep similarity in LLM mode?** | Free sanity check (~ 2 ms) |
| **Still see "Unknown"?** | Reply didn't start with any known token; tweak alias or prompt |
| **Where are model weights?** | Hugging Face cache (`~/.cache/huggingface`) |
| **Crash recovery?** | Append-only output; rerun resumes automatically |
| **Overwrite output?** | Use `--overwrite` to replace an existing file |
| **How to get detailed explanations?** | Use `--verbose` flag |
| **API errors?** | Ensure ARGO_USER is set and model names don't have hyphens |
| **What model should I use?** | gpto3mini is faster, gpt4o provides better biological context |

---

*Happy concording! – Stars ⭐, issues 🐞, and PRs 💡 welcome.*