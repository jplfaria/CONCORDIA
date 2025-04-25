
# CONCORDIA  
*CONcordance of Curated & Original Raw Descriptions In Annotations*

Concordia compares two functional-annotation sources—old vs new, RAST vs UniProt, manual vs AI—and streams a tidy file with a PubMedBERT cosine-similarity score **plus** a one-word **LLM label** drawn from a 7-class ontology.

---

## Installation
```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install          # deps + CLI
poetry shell            # activate venv
```

---

## Quick-start recipes

| Purpose | Command |
|---------|---------|
| CSV with **default LLM** (o3-mini → apps-dev) | `concord example_data/annotations_test.csv` |
| TSV with **GPT-4o** (prod) | `concord example_data/annotations_test.tsv --llm-model gpt4o` |
| **Embed-only**, no LLM | `concord …csv --mode local` |
| **Dual** (embed + LLM) | `concord …csv --mode dual` |
| Two ad-hoc strings | `concord --text-a "RecA" --text-b "DNA recombinase A"` |

---

## Accepted input formats

| Extension | Loader | Note |
|-----------|--------|------|
| `.csv` | `read_csv(sep=",")` | default |
| `.tsv` / `.tab` | `read_csv(sep="\t")` | or `--sep "\t"` |
| `.json` | `read_json()` | list-of-objects **or** column-orient |

If you do *not* pass `--col-a / --col-b`, the **first two textual columns that do not end with “id”** are used.

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
| **`FILE`** | Input table (`.csv`, `.tsv`, `.json`). |
| `--text-a / --text-b` | Compare two strings instead of a file. |
| `--mode` | `llm` (default) | `local` | `dual` |
| `--llm-model` | Gateway LLM (`gpto3mini`, `gpt4o`, …). |
| `--output` | Destination path (file-mode only). |
| `--cfg` | Alternate YAML config. |
| `--col-a / --col-b` | Explicit annotation columns. |
| `--sep` | Custom delimiter for text files. |

---

## Modes

| Mode | What happens | Extra columns |
|------|--------------|---------------|
| **llm**  | Skip embeddings; every pair → LLM | `label`, `note` |
| **local**| PubMedBERT embeddings → cosine → heuristic label | `similarity_Pubmedbert`, `label` |
| **dual** | Embeddings **and** LLM for every row | `similarity_Pubmedbert`, `label`, `note` |

> **Embedding model** `NeuML/pubmedbert-base-embeddings` (Apache-2.0).

---

## Output columns

| Name | Description |
|------|-------------|
| *(all originals)* | Copied unchanged |
| `similarity_Pubmedbert` | Cosine similarity (null in pure LLM) |
| `label` | Final label (from table below) |
| `note` | Very short LLM reason (blank if heuristic) |

### 7-label ontology

| Label | Meaning |
|-------|---------|
| **Exact** | Same function; wording/punctuation only |
| **Synonym** | Semantically equivalent paraphrase |
| **Broader** | A ⊃ B (A more general) |
| **Narrower** | A ⊂ B (A more specific) |
| **Related** | Same pathway / complex / family but not parent–child |
| **Uninformative** | Placeholder or extremely generic |
| **Different** | No functional overlap |

---

## Alias system – why you rarely see “Unknown”

Older prompts taught GPT models to answer with legacy words like *Identical* or *Partial*.  
`llm_label()` now contains a **one-line alias map**:

```python
_ALIAS = {"Identical": "Exact",
          "Partial":   "Related",
          "Equivalent": "Synonym"}
label = _ALIAS.get(label, label)
```

Any reply starting with those legacy tokens is transparently remapped to the new 7-label set, so you almost never get `Unknown`. Only genuinely empty or malformed replies remain `Unknown`, signalling a real problem.

---

## Prompting – tweak in one place

All prompt logic lives in **`concord/llm/prompts.py`**.  
Edit the few-shot block or definitions to experiment; no other code needs to change.

---

## Config snapshot (`concord/config.yaml`)
```yaml
engine:
  mode: llm

llm:
  model: gpto3mini
  stream: false
  user: ${ARGO_USER}

local:
  model_id: NeuML/pubmedbert-base-embeddings
```
*(Leave `env` unset—o-series → apps-dev, GPT-4* → apps-prod.)*

---

## Progress & recovery

* Output is **appended row-by-row** – kill & rerun and processed pairs are skipped.  
* Live progress bar:
```
Processing 73%|██████████████▋ | 730/1000 [00:14<00:05, 49.2it/s]
```
o3-mini ≈ 100 ms/pair; embeddings ≈ 1 ms/string on M-series Mac.

---

## FAQ

| Q | A |
|---|---|
| **Why keep similarity in LLM mode?** | Free sanity check (~2 ms). |
| **Still see “Unknown”?** | Means model’s first token wasn’t in 7-label set *and* not in alias map; tweak alias or prompt. |
| **Where are weights?** | Hugging Face cache (`~/.cache/huggingface`). |
| **Crash recovery?** | Output is append-only; rerun resumes automatically. |

---

*Happy concording! – Stars ⭐, issues 🐞, and PRs 💡 welcome.*