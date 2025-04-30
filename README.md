# CONCORDIA  
*CONcordance of Curated & Original Raw Descriptions In Annotations*

Concordia compares two functional-annotation sources—old vs new, RAST vs UniProt, manual vs AI—and writes a tidy table with  

* **`similarity_Pubmedbert`** – cosine similarity from the PubMedBERT sentence-embedding model  
* **`label`** – a one-word judgement from an LLM (`o3-mini` by default) drawn from a 7-class ontology  
* **`note`** – the LLM’s ultra-short reason (blank when the heuristic label is used)

You can choose from **four processing modes**:

| Mode | What happens |
|------|--------------|
| **llm** | LLM only — cheapest if you already trust the model |
| **local** | PubMedBERT embeddings → cosine → heuristic label (no LLM) |
| **dual** | Embeddings **and** LLM (baseline) |
| **simhint** | Same as **dual** **plus** the cosine similarity is prefixed to the prompt as a weak prior |

---

## Installation
```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install          # installs deps + CLI
poetry shell            # activate the virtualenv
```

---

## Quick-start recipes

| Goal | Command |
|------|---------|
| CSV with default LLM (o3-mini → dev) | `concord example_data/annotations_test.csv` |
| TSV with GPT-4o (prod)               | `concord example_data/annotations_test.tsv --llm-model gpt4o` |
| Embed-only (no LLM)                  | `concord …csv --mode local` |
| Dual (baseline)                      | `concord …csv --mode dual` |
| Similarity-hint                      | `concord …csv --mode simhint` |
| Two ad-hoc strings                   | `concord --text-a "RecA" --text-b "DNA recombinase A"` |
| **Overwrite** existing output        | add `--force` |

---

## Accepted input formats

| Extension | Loader | Note |
|-----------|--------|------|
| `.csv`           | `pandas.read_csv(sep=',')` | default |
| `.tsv` / `.tab`  | `pandas.read_csv(sep='\t')`| or `--sep "\t"` |
| `.json`          | `pandas.read_json()`       | list-of-objects **or** column-orient |

If you do **not** pass `--col-a / --col-b`, the **first two textual columns that don’t end with `id`** are taken.

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
| `--mode`               | `llm` (default) | `local` | `dual` | `simhint` |
| `--llm-model`          | Gateway LLM (`gpto3mini`, `gpt4o`, …) |
| `--retry`              | Automatic blank-reply retries (default 5) |
| `--force`              | Overwrite existing output instead of appending |
| `--output`             | Destination path (file-mode only) |
| `--cfg`                | Alternate YAML config |
| `--col-a / --col-b`    | Explicit annotation columns |
| `--sep`                | Custom delimiter for text files (e.g. `"\t"`) |

---

## Modes in detail

| Mode | Pipeline | Extra columns |
|------|----------|---------------|
| **llm**    | Skip embeddings; every pair → LLM               | `label`, `note` |
| **local**  | PubMedBERT embeddings → cosine → heuristic      | `similarity_Pubmedbert`, `label` |
| **dual**   | Embeddings **and** LLM for every row            | `similarity_Pubmedbert`, `label`, `note` |
| **simhint**| Same as **dual**, but cosine similarity is sent to the LLM prompt | `similarity_Pubmedbert`, `label`, `note` |

> **Embedding model** [`NeuML/pubmedbert-base-embeddings`](https://huggingface.co/NeuML/pubmedbert-base-embeddings) (Apache-2.0)

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

## Prompting — tweak in one place

All prompt text lives in **`concord/llm/prompts.py`**.  
Edit the few-shot examples or definitions to experiment; no other code must change.

---

## Config snapshot (`concord/config.yaml`)
```yaml
engine:
  mode: llm

llm:
  model: gpto3mini        # auto-routes to apps-dev
  stream: false
  user: ${ARGO_USER}

local:
  model_id: NeuML/pubmedbert-base-embeddings
```
*(Leave `env` unset — o-series → apps-dev, GPT-4* → apps-prod.)*

---

## Progress, recovery & overwrite

* Output is **appended row-by-row** – abort with Ctrl-C and rerun; finished pairs are skipped.  
  Add `--force` to **replace** an existing file instead.  
* Live progress bar example:
  ```
  Processing 73%|██████████████▋ | 730/1000 [00:14<00:05, 49.2it/s]
  ```
  o3-mini ≈ 0.10 s per pair; embeddings ≈ 1 ms per string on Apple M-series.

---

## FAQ

| Q | A |
|---|---|
| **Why keep similarity in LLM mode?** | Free sanity check (~ 2 ms) |
| **Still see “Unknown”?** | Reply didn’t start with any known token; tweak alias or prompt |
| **Where are model weights?** | Hugging Face cache (`~/.cache/huggingface`) |
| **Crash recovery?** | Append-only output; rerun resumes automatically |
| **Overwrite output?** | Use `--force` to replace an existing file |

---

*Happy concording! – Stars ⭐, issues 🐞, and PRs 💡 welcome.*