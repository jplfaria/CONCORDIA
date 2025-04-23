# CONCORDIA  
*CONcordance of Curated & Original Raw Descriptions In Annotations*

CONCORDIA is a lightweight engine that compares **any two functional-annotation sources**—old vs. new, RAST vs. UniProt, manual vs. automated, etc.—and exports a tidy CSV with similarity scores and one-word labels.

---

## Installation

```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install      # install deps + CLI script
poetry shell        # activate the virtual-env
```

---

## Quick-start recipes

| Purpose | Command |
|---------|---------|
| Compare a CSV with **default LLM** (o3-mini → apps-dev) | `concord example_data/changes.subset.csv` |
| Same CSV with **GPT-4o** (prod) | `concord example_data/changes.subset.csv --llm-model gpt4o` |
| **Embed-only**, no LLM | `concord example_data/changes.subset.csv --mode local` |
| **Hybrid** (embed first, LLM only for ambiguous pairs) | `concord example_data/changes.subset.csv --mode hybrid` |
| Write to **custom file** | `concord example_data/changes.subset.csv --output results.csv` |
| **Ad-hoc strings** (LLM) | `concord --text-a "DNA repair protein RecA" --text-b "Recombinase A"` |
| Ad-hoc strings, **hybrid** | `concord --text-a "DNA ligase" --text-b "NAD-dependent ligase" --mode hybrid` |

---

## Minimal sample CSV

```csv
old_annotation,new_annotation
DNA repair protein RecA,Recombinase A
Hypothetical protein,Uncharacterized protein
```

Any extra columns are preserved in the output.

---

## CLI options

| Flag | Description |
|------|-------------|
| **`csv`** | Path to CSV. If it contains columns `old_annotation` and `new_annotation` they’re used; otherwise the first two text columns are taken. |
| `--text-a / --text-b` | Compare two free-text strings instead of a CSV. |
| `--mode` | `llm` (default) | `local` | `hybrid` |
| `--llm-model` | Override gateway model (`gpto3mini`, `gpt4o`, …). |
| `--output` | Destination CSV (file-mode only). |
| `--cfg` | Alternate YAML config (default `concord/config.yaml`). |
| `--col-a / --col-b` | Explicitly name the annotation columns in the CSV. |

---

## Output fields

| Column | Meaning |
|--------|---------|
| All original columns | Carried through unchanged. |
| `similarity` | Cosine similarity of PubMedBERT embeddings (`null` in pure LLM mode). |
| `label` | One-word relation (see table below). |
| `note` | Very short reason returned by the LLM (blank in local mode). |

### Label definitions

| Label | Meaning |
|-------|---------|
| **Identical** | Function descriptions are effectively the same (minor wording differences). |
| **Synonym** | Same meaning via clear synonym / alternative phrasing. |
| **Partial** | Overlap in meaning; one description is broader/narrower. |
| **New** | Descriptions represent different or unrelated functions. |
| **Unknown** | LLM returned an empty reply (rare). |

### Label logic

* **local** – heuristic on similarity  
  * > 0.90 → Identical  
  * 0.60 – 0.90 → Partial  
  * < 0.60 → New
* **llm** – first token from Argo Gateway response.  
* **hybrid** – local rule when similarity outside gray zone; otherwise LLM.

---

## Config snapshot (`concord/config.yaml`)

```yaml
engine:
  mode: hybrid
  hybrid_thresholds:
    lower: 0.60
    upper: 0.85

llm:
  model: gpto3mini          # auto-routes to apps-dev
  stream: false
  user: ${ARGO_USER}        # export ARGO_USER=<your-login>

local:
  model_id: NeuML/pubmedbert-base-embeddings
```

*(Leave `env` unset—client auto-selects: o-series → dev, GPT-4* → prod.)*

---

## Progress & performance

During embedding phases (local/hybrid) you’ll see a **tqdm** bar:

```
Processing  73%|██████████████▋   | 730/1000 [00:14<00:05, 49.2it/s]
```

o3-mini calls take roughly 100–150 ms per pair on the lab network; GPT-4o is similar.

---

## FAQ

| Q | A |
|---|---|
| **Why keep `similarity` in `llm` mode?** | It’s nearly free (~2 ms) and helps QA the LLM label. |
| **What does `Unknown` mean?** | Gateway returned an empty string—retry or switch model. |
| **How do I tune thresholds?** | Edit `hybrid_thresholds` in `config.yaml` or pass a custom file via `--cfg`. |
| **Where are the model weights?** | Not in the repo—sentence-transformers downloads them on first run to your Hugging Face cache. |

---

*Happy concording! Issues & PRs welcome.*